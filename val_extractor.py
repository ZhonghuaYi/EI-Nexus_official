from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from core.geometry.gt_generation import gt_matches_from_pose_depth
from core.geometry.wrappers import Camera, Pose
from omegaconf import DictConfig
from utils.common import Padder
from core.metrics.keypoints_metrics import Repeatability, ValidDescriptorsDistance
from core.metrics.matching_metrics import (
    MeanMatchingAccuracy,
    MatchingRatio,
    HomographyEstimation,
    RelativePoseEstimation,
)


def padding_data(data: Dict, padder: Padder) -> Dict:
    (
        data["events_rep"],
        data["image"],
        data["depth"],
        data["depth_mask"],
        data["events_image"],
    ) = padder.pad(
        data["events_rep"],
        data["image"],
        data["depth"],
        data["depth_mask"],
        data["events_image"],
    )
    return data


def unpad_data(data: Dict, padder: Padder) -> Dict:
    (
        data["events_rep"],
        data["image"],
        data["depth"],
        data["depth_mask"],
        data["events_image"],
    ) = padder.unpad(
        data["events_rep"],
        data["image"],
        data["depth"],
        data["depth_mask"],
        data["events_image"],
    )
    return data


def unpad_feats(feats: Dict, padder: Padder) -> Dict:
    feats["score"], feats["nms"], feats["normalized_descriptors"] = padder.unpad(
        feats["score"], feats["nms"], feats["normalized_descriptors"]
    )
    return feats


@torch.no_grad()
def val_model_by_loss(
    model: nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    keypoints_loss,
    descriptors_loss,
    matcher_loss,
    device: str,
    cfg: DictConfig,
):
    pbar = tqdm(val_dataloader, total=len(val_dataloader))

    # set up the metrics
    R_1 = Repeatability("repeatability@1", distance_thresh=1)
    R_3 = Repeatability("repeatability@3", distance_thresh=3)
    VVD = ValidDescriptorsDistance("VVD", distance_thresh_list=[1, 3], ordering="yx")
    MMA_1 = MeanMatchingAccuracy("MMA@1", threshold=1)
    MMA_3 = MeanMatchingAccuracy("MMA@3", threshold=3)
    MR = MatchingRatio("MR")
    HE = HomographyEstimation("HE", correctness_thresh=[3, 5, 10])
    RPE = RelativePoseEstimation("RPE", pose_thresh=[5, 10, 20])

    metrics_dict = {}
    padder = None
    for batch in pbar:
        # load data
        data0, _, _, _ = batch

        events = data0["events_rep"].to(device)
        image = data0["image"].to(device)
        # events mask
        events_mask = (data0["events_image"].to(device)) > 0
        # events_mask = torch.ones_like(data0['events_image']).to(device) > 0

        # get the predictions
        events_feats, image_feats, matches = model(events, image, events_mask)

        # get metrics
        b = events.shape[0]
        true_homography = torch.eye(3).repeat(b, 1, 1).to(device)
        true_relative_pose = torch.eye(4).repeat(b, 1, 1).to(device)
        r_1 = R_1.update_one(
            events_feats["sparse_positions"][0],
            image_feats["sparse_positions"][0],
            events.shape[-2:],
            image.shape[-2:],
            true_homography[0],
        )
        r_3 = R_3.update_one(
            events_feats["sparse_positions"][0],
            image_feats["sparse_positions"][0],
            events.shape[-2:],
            image.shape[-2:],
            true_homography[0],
        )
        vvd = VVD.update_one(
            events_feats["sparse_positions"][0],
            image_feats["sparse_positions"][0],
            events_feats["sparse_descriptors"][0],
            image_feats["sparse_descriptors"][0],
            events.shape[-2:],
            image.shape[-2:],
            true_homography[0],
        )
        mma_1 = MMA_1.update_one(
            matches["matched_kpts0"][0], matches["matched_kpts1"][0], true_homography[0]
        )
        mma_3 = MMA_3.update_one(
            matches["matched_kpts0"][0], matches["matched_kpts1"][0], true_homography[0]
        )
        mr = MR.update_one(
            matches["matched_kpts0"][0],
            matches["matched_kpts1"][0],
            events_feats["sparse_positions"][0],
            image_feats["sparse_positions"][0],
        )
        he = HE.update_one(
            events_feats["image_size"][0],
            matches["matched_kpts0"][0],
            matches["matched_kpts1"][0],
            true_homography[0],
        )
        rpe = RPE.update_one(
            matches["matched_kpts0"][0],
            matches["matched_kpts1"][0],
            data0["K"][0],
            data0["K"][0],
            true_relative_pose[0],
        )

        all_metrics = {**vvd,}

        # calculate the loss on keypoints
        keypoints_loss_value, kpts_loss_info = keypoints_loss(
            events_feats, image_feats, events_mask, padder=padder
        )
        # calculate the loss on descriptors
        descriptors_loss_value, descriptors_loss_info = descriptors_loss(
            events_feats, image_feats, events_mask
        )
        # calculate the total loss
        loss = keypoints_loss_value + descriptors_loss_value

        # get the VAL info
        
        loss_info = {}
        loss_info = {
            **kpts_loss_info,
            **descriptors_loss_info,
            "VAL_loss": loss.detach().item(),
        }
        loss_info.update(all_metrics)
        loss_info = {f"VAL_{k}": v for k, v in loss_info.items()}
        for k, v in loss_info.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().item()
            if k in metrics_dict:
                metrics_dict[k].append(v)
            else:
                metrics_dict[k] = [v]

    for k, v in metrics_dict.items():
        v = np.array(v).astype(np.float32)
        v = v[np.isfinite(v)]
        metrics_dict[k] = v.mean()

    he_auc = HE.compute_all_auc()
    for k in he_auc:
        metrics_dict[f"VAL_{HE.metric_name}@{k}_auc"] = he_auc[f"{k}"]
    rpe_auc = RPE.compute_all_auc()
    for k in rpe_auc:
        metrics_dict[f"VAL_{RPE.metric_name}@{k}_auc"] = rpe_auc[f"{k}"]

    return metrics_dict


if __name__ == "__main__":
    from core.modules.EIM import EIM
    from torch.utils.data import DataLoader
    from datasets.MVSEC import MVSECDataset
    from omegaconf import OmegaConf
    from hydra import compose, initialize

    with initialize(version_base=None, config_path="configs", job_name="test_app"):
        cfg = compose(config_name="train_EDM")

    # cfg = OmegaConf.load('configs/train_EDM.yaml')
    # print(cfg)

    model_cfg = cfg.model
    model = EIM(model_cfg, "cuda")
    ckpt_path = "runs/Mar20_16-28-59_mvsec-E+MNN-vgg-superpointv1-1+1-mse+mae-0.15s-16c-normalized_desc-no_e_aug/final.pth"
    model.load_state_dict(torch.load(ckpt_path))

    dataset_cfg = cfg.dataset
    dataset = MVSECDataset(dataset_cfg, is_train=False, use_aug=False)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    from core.loss import build_losses

    keypoints_loss, descriptors_loss, matcher_loss = build_losses(cfg.train.loss)

    metrics_dict = val_model_by_loss(
        model, data_loader, keypoints_loss, descriptors_loss, matcher_loss, "cuda", cfg
    )
