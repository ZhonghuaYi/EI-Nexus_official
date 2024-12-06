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
from core.metrics.keypoints_metrics import Repeatability
from core.metrics.matching_metrics import MeanMatchingAccuracy, MatchingRatio, HomographyEstimation, RelativePoseEstimation


def padding_data(data: Dict, padder: Padder) -> Dict:
    data['events_rep'], data['image'], data['depth'], data['depth_mask'], data['events_image'] = padder.pad(data['events_rep'],
                                                                                                        data['image'],
                                                                                                        data['depth'],
                                                                                                        data[
                                                                                                            'depth_mask'],
                                                                                                        data[
                                                                                                            'events_image'])
    return data


def unpad_data(data: Dict, padder: Padder) -> Dict:
    data['events_rep'], data['image'], data['depth'], data['depth_mask'], data['events_image'] = padder.unpad(
        data['events_rep'], data['image'], data['depth'], data['depth_mask'], data['events_image'])
    return data


def unpad_feats(feats: Dict, padder: Padder) -> Dict:
    feats['score'], feats['nms'], feats['normalized_descriptors'] = padder.unpad(feats['score'], feats['nms'],
                                                                                 feats['normalized_descriptors'])
    return feats


@torch.no_grad()
def val_model(
        model: nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        matcher_loss,
        device: str,
        cfg: DictConfig,
):
    pbar = tqdm(val_dataloader, total=len(val_dataloader))
    
    # set up the metrics
    RPE = RelativePoseEstimation("RPE", pose_thresh=[5, 10, 20])
    
    metrics_dict = {}
    for batch in pbar:

        # load data
        data0, data1, T_0to1, T_1to0 = batch


        events = data0['events_rep'].to(device)
        image = data1['image'].to(device)
        # events mask
        events_mask = (data0['events_image'].to(device)) > 0
        # events_mask = torch.ones_like(data0['events_image']).to(device) > 0
        
        # get the predictions
        events_feats, image_feats, matches = model(events, image, events_mask)
        
        T_0to1 = T_0to1.to(device)
        T_1to0 = T_1to0.to(device)
        camera0 = Camera.from_calibration_matrix(data0['K'].float().to(device))
        camera1 = Camera.from_calibration_matrix(data1['K'].float().to(device))
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
        
        losses, metrics = model.matcher.matcher.loss(matches, gt)

        # get metrics
        true_relative_pose = T_0to1
        rpe = RPE.update_one(matches['matched_kpts0'],
                            matches['matched_kpts1'],
                            data0['K'][0],
                            data0['K'][0],
                            true_relative_pose[0])
        
        all_metrics = {
            **metrics,
            **rpe
        }

        # calculate the total loss
        loss = losses['total'].mean()

        # get the VAL info
        loss_info = {
            'VAL_loss': loss.detach().item(),
        }
        loss_info.update(all_metrics)
        loss_info = {f'VAL_{k}': v for k, v in loss_info.items()}
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
    
    rpe_auc = RPE.compute_all_auc()
    for k in rpe_auc:
        metrics_dict[f'VAL_{RPE.metric_name}@{k}_auc'] = rpe_auc[f'{k}']
    
    return metrics_dict
