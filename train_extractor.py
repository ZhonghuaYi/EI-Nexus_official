from typing import Dict
import argparse
import os
import datetime

import wandb
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from rich import pretty, print

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler
import numpy as np
import cv2 as cv

from datasets.MVSEC import fetch_mvsec_dataloader
from datasets.EC import fetch_ec_dataloader
from core.modules import build_model
from core.loss import build_losses
from core.geometry.gt_generation import gt_matches_from_pose_depth
from core.geometry.wrappers import Camera, Pose

from core.metrics.keypoints_metrics import Repeatability, ValidDescriptorsDistance
from core.metrics.matching_metrics import (
    MeanMatchingAccuracy,
    MatchingRatio,
    RelativePoseEstimation,
    HomographyEstimation,
)

from utils.optimizers import build_optimizer
from utils.schedulers import build_scheduler
from utils.common import (
    setup,
    count_parameters,
    get_envs,
    set_cuda_devices,
    parallel_model,
)
from utils.logger import Logger

from val_extractor import val_model_by_loss


def wandb_init(cfg: DictConfig):
    if cfg.wandb.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.login(key=cfg.wandb.key)
    wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cfg.wandb.name,
        notes=cfg.wandb.notes,
        tags=cfg.wandb.tags,
        # settings=wandb.Settings(code_dir="./"),
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, "conf.yaml"))
    # wandb.run.log_code("./")


@hydra.main(version_base=None, config_path="configs", config_name="train_EDM_stage1")
def main(cfg):
    # set up the environment
    setup(
        cfg.setup.seed,
        cfg.setup.cudnn_enabled,
        cfg.setup.allow_tf32,
        cfg.setup.num_threads,
    )

    # set up the cuda devices and the distributed training
    cuda_available = torch.cuda.is_available()
    device = None
    if cuda_available and cfg.setup.device == "cuda":
        # set_cuda_devices(OmegaConf.to_object(cfg.setup.gpus))
        device = torch.device("cuda")
        rank, local_rank, world_size = get_envs()
        if local_rank != -1:  # DDP distriuted mode
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist.init_process_group(
                backend="nccl" if dist.is_nccl_available() else "gloo",
                init_method="env://",
                rank=local_rank,
                world_size=world_size,
            )
    else:
        device = torch.device("cpu")

    # set up the logger
    logger = Logger(
        cfg.experiment,
        cfg.logger.status_freq,
        cfg.logger.files_to_backup,
        cfg.logger.dirs_to_backup,
    )
    OmegaConf.save(config=cfg, f=os.path.join(logger.log_dir, "conf.yaml"))

    # wandb init
    if rank in [-1, 0]:
        wandb_init(cfg)
    logger.log_info(f"Wandb: {cfg.wandb}\n", rank)

    # print the configuration
    logger.log_info(
        f"Experiment name: [bold yellow]{cfg.experiment}[/bold yellow]", rank
    )
    logger.log_info(f"Logger: \n {cfg.logger}", rank)
    logger.log_info(f"Setup: \n {cfg.setup}", rank)
    logger.log_info(f"DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}, {device}", rank)

    # set up the dataset
    dataset_name = cfg.dataset.name
    train_dataloader = None
    val_dataloader = None
    logger.log_info(f"Dataset: [bold yellow]{dataset_name}[/bold yellow]", rank)
    if dataset_name == "mvsec":
        train_dataloader = fetch_mvsec_dataloader(
            cfg.dataset, "train", logger, rank, world_size
        )
        val_dataloader = fetch_mvsec_dataloader(
            cfg.dataset, "val", logger, rank, world_size
        )
    elif dataset_name == "ec":
        train_dataloader = fetch_ec_dataloader(
            cfg.dataset, "train", logger, rank, world_size
        )
        val_dataloader = fetch_ec_dataloader(
            cfg.dataset, "val", logger, rank, world_size
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    # set up the model
    model_name = cfg.model.name
    logger.log_info(f"\nModel: [bold yellow]{model_name}[/bold yellow]", rank)
    model = build_model(cfg.model, device, logger)
    logger.log_info(f"Model initialized. Parameters: {count_parameters(model)}", rank)

    # set up the optimizer
    logger.log_info(
        f"\nOptimizer: [bold yellow]{cfg.train.optimizer.type}[/bold yellow]", rank
    )
    logger.log_info(f"{cfg.train.optimizer[cfg.train.optimizer.type]}", rank)
    optimizer = build_optimizer(cfg.train.optimizer, model.parameters())

    # set up the scheduler
    logger.log_info(
        f"Scheduler: [bold yellow]{cfg.train.scheduler.type}[/bold yellow]", rank
    )
    logger.log_info(f"{cfg.train.scheduler[cfg.train.scheduler.type]}", rank)
    scheduler = build_scheduler(cfg.train.scheduler, optimizer)

    # resume from checkpoint if needed
    if cfg.resume:
        logger.log_info(
            f"[bold yellow]Resuming from checkpoint: {cfg.resume}[/bold yellow]\n", rank
        )
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # set up the loss function
    keypoints_loss_type = cfg.train.loss.keypoints_loss.type
    descriptors_loss_type = cfg.train.loss.descriptors_loss.type
    logger.log_info(
        f"Losses:\n"
        f"- keypoints_loss: [bold yellow]{keypoints_loss_type}[/bold yellow] -- {cfg.train.loss.keypoints_loss[keypoints_loss_type]}\n"
        f"- descriptors_loss: [bold yellow]{descriptors_loss_type}[/bold yellow] -- {cfg.train.loss.descriptors_loss[descriptors_loss_type]}\n"
        f"- feature_loss: [bold yellow]{cfg.train.loss.feature_loss.type}[/bold yellow] -- {cfg.train.loss.feature_loss}\n",
        rank,
    )
    losses = build_losses(cfg.train.loss)
    keypoints_loss = losses["keypoints_loss"]
    descriptors_loss = losses["descriptors_loss"]
    feature_loss = losses["feature_loss"]
    matcher_loss = losses["matcher_loss"]

    # set up the metrics
    R_1 = Repeatability("repeatability@1", distance_thresh=1, ordering="yx")
    R_3 = Repeatability("repeatability@3", distance_thresh=3, ordering="yx")
    VVD = ValidDescriptorsDistance("VVD", distance_thresh_list=[1, 3], ordering="yx")
    MMA_1 = MeanMatchingAccuracy("MMA@1", threshold=1, ordering="yx")
    MMA_3 = MeanMatchingAccuracy("MMA@3", threshold=3, ordering="yx")
    MR = MatchingRatio("MR")
    HE = HomographyEstimation("HE", correctness_thresh=[3, 5, 10])
    RPE = RelativePoseEstimation("RPE", pose_thresh=[5, 10, 20])

    # parallelize the model if needed
    model = parallel_model(model, device, rank, local_rank)

    # use mixed precision if needed
    logger.log_info(
        f"[bold yellow]Use Mixed Precision: {cfg.setup.mixed_precision}[/bold yellow]",
        rank,
    )
    if cfg.setup.mixed_precision:
        scaler = GradScaler(enabled=True)

    # start training
    torch.cuda.empty_cache()
    logger.log_info(f"Training epochs: {cfg.train.epochs}", rank)
    logger.log_info(f"Start training", rank)
    for epoch in range(cfg.train.epochs):

        logger.log_info(f"[bold yellow]Epoch {epoch}[/bold yellow]", rank)
        model.train()
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in pbar:
            # load data
            data0, _, _, _ = batch

            events = data0["events_rep"].to(device)
            image = data0["image"].to(device)
            # events mask
            events_mask = (data0["events_image"].to(device)) > 0
            # events_mask = torch.ones_like(data0['events_image']).to(device) > 0
            image_mask = (data0["depth_mask"].to(device)) > 0

            # forward
            optimizer.zero_grad()
            if cfg.setup.mixed_precision:
                with torch.cuda.amp.autocast():
                    raise NotImplementedError

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # get the predictions
                events_feats, image_feats, matches = model(events, image, events_mask, image_mask)
                # calculate the loss on keypoints
                keypoints_loss_value, kpts_loss_info = keypoints_loss(
                    events_feats, image_feats, events_mask
                )
                # calculate the loss on descriptors
                descriptors_loss_value, descriptors_loss_info = descriptors_loss(
                    events_feats, image_feats, events_mask
                )
                # calculate the feature loss
                feature_loss_value, feature_loss_info = feature_loss(
                    events_feats, image_feats, events_mask
                )
                # calculate the total loss
                loss = keypoints_loss_value + descriptors_loss_value + feature_loss_value

                # get metrics
                b = events.shape[0]
                true_homography = torch.eye(3).repeat(b, 1, 1).to(device)
                # true_relative_pose = torch.eye(4).repeat(b, 1, 1).to(device)
                r_1 = R_1.update_batch(
                    events_feats["sparse_positions"],
                    image_feats["sparse_positions"],
                    events.shape[-2:],
                    image.shape[-2:],
                    true_homography,
                )
                r_3 = R_3.update_batch(
                    events_feats["sparse_positions"],
                    image_feats["sparse_positions"],
                    events.shape[-2:],
                    image.shape[-2:],
                    true_homography,
                )
                vvd = VVD.update_batch(
                    events_feats["sparse_positions"],
                    image_feats["sparse_positions"],
                    events_feats["sparse_descriptors"],
                    image_feats["sparse_descriptors"],
                    events.shape[-2:],
                    image.shape[-2:],
                    true_homography,
                )
                mma_1 = MMA_1.update_batch(
                    matches["matched_kpts0"], matches["matched_kpts1"], true_homography
                )
                mma_3 = MMA_3.update_batch(
                    matches["matched_kpts0"], matches["matched_kpts1"], true_homography
                )
                mr = MR.update_batch(
                    matches["matched_kpts0"],
                    matches["matched_kpts1"],
                    events_feats["sparse_positions"],
                    image_feats["sparse_positions"],
                )
                he = HE.update_batch(events_feats['image_size'], matches['matched_kpts0'], matches['matched_kpts1'], true_homography)
                # rpe = RPE.update_batch(matches['matched_kpts0'], matches['matched_kpts1'], data0['K'], data0['K'], true_relative_pose)
                all_metrics = {
                    **r_1,
                    **r_3,
                    **vvd,
                    **mma_1,
                    **mma_3,
                    **mr,
                    **he,
                    # **rpe
                }

                # log
                loss_info = {
                    **kpts_loss_info,
                    **descriptors_loss_info,
                    **feature_loss_info,
                    "total_loss": loss.detach().item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                loss_info.update(all_metrics)
                if rank in [-1, 0]:
                    logger.write_status(loss_info)
                    wandb.log(loss_info)

                loss.backward()
                optimizer.step()

        # validation
        if rank in [-1, 0] and (epoch + 1) % cfg.train.val_freq == 0:
            logger.log_info(f"[bold yellow]Validation in Epoch {epoch}[/bold yellow]")
            model.eval()

            # validation
            metrics_dict = val_model_by_loss(
                model,
                val_dataloader,
                keypoints_loss,
                descriptors_loss,
                matcher_loss,
                device,
                cfg,
            )

            logger.write_results(metrics_dict)
            wandb.log(metrics_dict)

        # save checkpoint
        if rank in [-1, 0] and (epoch + 1) % cfg.train.checkpoint_freq == 0:
            logger.log_info(
                f"[bold yellow]Saving checkpoint in Epoch {epoch}[/bold yellow]"
            )
            checkpoint_path = os.path.join(
                f"{logger.log_dir}/checkpoints", f"checkpoint_{epoch}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                checkpoint_path,
            )
            logger.log_info(f"Checkpoint saved in {checkpoint_path}")

        # update the scheduler
        scheduler.step()

    # save the final model
    if rank in [-1, 0]:
        logger.log_info(f"[bold yellow]Saving final model[/bold yellow]")
        current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        PATH = f"{logger.log_dir}/{cfg.experiment}_{current_time}.pth"
        torch.save(model.state_dict(), PATH)
        torch.save(model.state_dict(), os.path.join(f"{logger.log_dir}", f"final.pth"))
        logger.log_info(f"[bold yellow]Model saved: {PATH}[/bold yellow]")

    # close the log
    if rank in [-1, 0]:
        wandb.finish()
        logger.close()

    # destroy process
    if world_size > 1 and rank == 0:
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    main()
