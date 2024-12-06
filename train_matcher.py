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
from core.modules import build_model
from core.loss import build_losses
from core.geometry.gt_generation import gt_matches_from_pose_depth
from core.geometry.wrappers import Camera, Pose

from core.metrics.matching_metrics import (
    MeanMatchingAccuracy,
    MatchingRatio,
    RelativePoseEstimation,
    HomographyEstimation,
)

from utils.optimizers import build_optimizer
from utils.schedulers import build_scheduler
from utils.common import setup, count_parameters, get_envs, set_cuda_devices, parallel_model
from utils.logger import Logger

from val_matcher import val_model


def wandb_init(cfg: DictConfig):
    if cfg.wandb.dryrun:
        os.environ['WANDB_MODE'] = 'dryrun'
    
    wandb.login(key=cfg.wandb.key)
    wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cfg.wandb.name,
        notes=cfg.wandb.notes,
        tags=cfg.wandb.tags,
        # settings=wandb.Settings(code_dir="./"),
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, 'conf.yaml'))
    # wandb.run.log_code("./")

    
@hydra.main(version_base=None, config_path='configs', config_name='train_EDM_stage2')
def main(cfg):
    # set up the environment
    setup(cfg.setup.seed, cfg.setup.cudnn_enabled, cfg.setup.allow_tf32, cfg.setup.num_threads)
    
    # set up the cuda devices and the distributed training
    cuda_available = torch.cuda.is_available()
    device = None
    if cuda_available and cfg.setup.device == 'cuda':
        # set_cuda_devices(OmegaConf.to_object(cfg.setup.gpus))
        device = torch.device('cuda')
        rank, local_rank, world_size = get_envs()
        if local_rank != -1:  # DDP distriuted mode
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
            dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo",
                                    init_method='env://', rank=local_rank, world_size=world_size)
    else:
        device = torch.device('cpu')
    
    # set up the logger
    logger = Logger(cfg.experiment, cfg.logger.status_freq, cfg.logger.files_to_backup, cfg.logger.dirs_to_backup)
    OmegaConf.save(config=cfg, f=os.path.join(logger.log_dir, 'conf.yaml'))
        
    # wandb init
    if rank in [-1, 0]:
        wandb_init(cfg)
    logger.log_info(f'Wandb: {cfg.wandb}\n', rank)
    
    # print the configuration
    logger.log_info(f'Experiment name: [bold yellow]{cfg.experiment}[/bold yellow]', rank)
    logger.log_info(f'Logger: \n {cfg.logger}', rank)
    logger.log_info(f'Setup: \n {cfg.setup}', rank)
    logger.log_info(f"DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}, {device}", rank)
    
    # set up the dataset
    dataset_name = cfg.dataset.name
    train_dataloader = None
    val_dataloader = None
    logger.log_info(f'Dataset: [bold yellow]{dataset_name}[/bold yellow]', rank)
    if dataset_name == 'mvsec':
        train_dataloader = fetch_mvsec_dataloader(cfg.dataset, 'train', logger, rank, world_size)
        val_dataloader = fetch_mvsec_dataloader(cfg.dataset, 'val', logger, rank, world_size)
    else:
        raise NotImplementedError(f'Dataset {dataset_name} not implemented')
    
    # set up the model
    model_name = cfg.model.name
    logger.log_info(f'\nModel: [bold yellow]{model_name}[/bold yellow]', rank)
    model = build_model(cfg.model, device, logger)
    logger.log_info(f'Model initialized. Parameters: {count_parameters(model)}', rank)
    
    # set up the optimizer
    logger.log_info(f'\nOptimizer: [bold yellow]{cfg.train.optimizer.type}[/bold yellow]', rank)
    logger.log_info(f'{cfg.train.optimizer[cfg.train.optimizer.type]}', rank)
    optimizer = build_optimizer(cfg.train.optimizer, model.parameters())
    
    # set up the scheduler
    logger.log_info(f'Scheduler: [bold yellow]{cfg.train.scheduler.type}[/bold yellow]', rank)
    logger.log_info(f'{cfg.train.scheduler[cfg.train.scheduler.type]}', rank)
    scheduler = build_scheduler(cfg.train.scheduler, optimizer)
    
    # resume from checkpoint if needed
    if cfg.resume:
        logger.log_info(f'[bold yellow]Resuming from checkpoint: {cfg.resume}[/bold yellow]\n', rank)
        checkpoint = torch.load(cfg.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # # set up the loss function
    # matcher_loss_type = cfg.train.loss.matcher_loss.type
    # logger.log_info(f'- matcher_loss: [bold yellow]{matcher_loss_type}[/bold yellow] -- {cfg.train.loss.matcher_loss[matcher_loss_type]}\n', rank)
    # _, _, matcher_loss = build_losses(cfg.train.loss)
    
    # set up the metrics
    RPE = RelativePoseEstimation("RPE", pose_thresh=[5, 10, 20])
    
    # parallelize the model if needed
    model = parallel_model(model, device, rank, local_rank)
    
    # use mixed precision if needed
    logger.log_info(f'[bold yellow]Use Mixed Precision: {cfg.setup.mixed_precision}[/bold yellow]', rank)
    if cfg.setup.mixed_precision:
        scaler = GradScaler(enabled=True)
    
    # start training
    torch.cuda.empty_cache()
    logger.log_info(f'Training epochs: {cfg.train.epochs}', rank)
    logger.log_info(f'Start training', rank)
    for epoch in range(cfg.train.epochs):
        
        logger.log_info(f'[bold yellow]Epoch {epoch}[/bold yellow]', rank)
        model.train()
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch in pbar:
            # load data
            data0, data1, T_0to1, T_1to0 = batch

            events = data0['events_rep'].to(device)
            image = data1['image'].to(device)
            # events mask
            events_mask = (data0['events_image'].to(device)) > 0
            # events_mask = torch.ones_like(data0['events_image']).to(device) > 0
            
            T_0to1 = T_0to1.to(device)
            T_1to0 = T_1to0.to(device)
            camera0 = Camera.from_calibration_matrix(data0['K'].float().to(device))
            camera1 = Camera.from_calibration_matrix(data1['K'].float().to(device))

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
                events_feats, image_feats, matches = model(events, image, events_mask)
                # calculate the ground truth
                gt = gt_matches_from_pose_depth(
                    kp0=matches['input_feats0']['sparse_positions'],
                    kp1=matches['input_feats1']['sparse_positions'],
                    camera0=camera0,
                    camera1=camera1,
                    depth0=data0['depth'].to(device),
                    depth1=data1['depth'].to(device),
                    T_0to1=Pose.from_4x4mat(T_0to1),
                    T_1to0=Pose.from_4x4mat(T_1to0),
                )
                gt = {f"gt_{k}": v.to(device) for k, v in gt.items()}
                # for lightglue
                losses, metrics = model.matcher.matcher.loss(matches, gt)
                # calculate the total loss
                loss = losses['total'].mean()

                # get metrics
                # rpe = RPE.update_batch(
                #     matches["matched_kpts0"],
                #     matches["matched_kpts1"],
                #     data0["K"],
                #     data1["K"],
                #     T_0to1,
                # )
                # all_metrics = {
                #     **rpe,
                # }

                # log
                loss_info = {
                    "total_loss": loss.detach().item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                }
                # loss_info.update(all_metrics)
                if rank in [-1, 0]:
                    logger.write_status(loss_info)
                    wandb.log(loss_info)

                loss.backward()
                optimizer.step()
        
        # validation
        if rank in [-1, 0] and (epoch + 1) % cfg.train.val_freq == 0:
            logger.log_info(f'[bold yellow]Validation in Epoch {epoch}[/bold yellow]')
            model.eval()
            
            # validation
            metrics_dict = val_model(model, val_dataloader, None, device, epoch)
            
            logger.write_results(metrics_dict)
            wandb.log(metrics_dict)
        
        # save checkpoint
        if rank in [-1, 0] and (epoch + 1) % cfg.train.checkpoint_freq == 0:
            logger.log_info(f'[bold yellow]Saving checkpoint in Epoch {epoch}[/bold yellow]')
            checkpoint_path = os.path.join(f'{logger.log_dir}/checkpoints', f'checkpoint_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            logger.log_info(f'Checkpoint saved in {checkpoint_path}')
        
        # update the scheduler
        scheduler.step()

    # save the final model
    if rank in [-1, 0]:
        logger.log_info(f'[bold yellow]Saving final model[/bold yellow]')
        current_time = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
        PATH = f'{logger.log_dir}/{cfg.experiment}_{current_time}.pth'
        torch.save(model.state_dict(), PATH)
        torch.save(model.state_dict(), os.path.join(f'{logger.log_dir}', f'final.pth'))
        logger.log_info(f'[bold yellow]Model saved: {PATH}[/bold yellow]')

    # During training, record a data point in this way
    # step=epoch records the x value of the curves, data records the y values
    # 'data' is a dict. Each key creates a figure with that as the title
    # wandb.log(step=1, data={'loss': 114.514})

    # close the log
    if rank in [-1, 0]:
        wandb.finish()
        logger.close()
    
    # destroy process
    if world_size > 1 and rank == 0:
        dist.destroy_process_group()
    
    return 0


if __name__ == '__main__':
    main()
