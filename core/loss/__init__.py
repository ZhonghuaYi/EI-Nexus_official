from typing import Any, Tuple
from omegaconf import DictConfig

import torch.nn as nn

from .extractor_loss import ScoreLoss, LogitsLoss, DescriptorsLoss, FeatureLoss
from .matcher_loss import MNNLoss, NLLLoss


class Pass(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, *args, **kwargs) -> Any:
        return None


def build_losses(config: DictConfig) -> Tuple:
    """
    Args:
        config (DictConfig): the config.
        
    Returns:
        keypoints_loss (nn.Module): the keypoints loss.
        descriptors_loss (nn.Module): the descriptors loss.
        matcher_loss (nn.Module): the matcher loss.
    """
    
    keypoints_loss = Pass() 
    descriptors_loss = Pass()
    matcher_loss = Pass()
    
    if config.keypoints_loss.type == 'ScoreLoss':
        keypoints_loss = ScoreLoss(**config.keypoints_loss.ScoreLoss)
    elif config.keypoints_loss.type == 'LogitsLoss':
        keypoints_loss = LogitsLoss(
            weight=config.keypoints_loss.LogitsLoss.weight,
            mode=config.keypoints_loss.LogitsLoss.mode,
            cell_size=config.keypoints_loss.LogitsLoss.cell_size
        )
        
    if config.feature_loss.type == 'FeatureLoss':
        feature_loss = FeatureLoss(
            **config.feature_loss.FeatureLoss
        )
    
    if config.descriptors_loss.type == 'DescriptorsLoss':
        descriptors_loss = DescriptorsLoss(
            **config.descriptors_loss.DescriptorsLoss
        )
    
    if config.matcher_loss.type == 'MNNLoss':
        matcher_loss = MNNLoss(
            weight=config.matcher_loss.MNNLoss.weight
        )
    elif config.matcher_loss.type == 'NLLLoss':
        matcher_loss = NLLLoss(
            weight=config.matcher_loss.NLLLoss.weight,
            nll_balancing=config.matcher_loss.NLLLoss.nll_balancing
        )
    
    return {
        'keypoints_loss': keypoints_loss,
        'descriptors_loss': descriptors_loss,
        'feature_loss': feature_loss,
        'matcher_loss': matcher_loss
    }
    
