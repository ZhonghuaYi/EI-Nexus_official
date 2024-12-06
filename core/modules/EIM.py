from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

from .Extractors import EventKeypointsExtractor, ImageKeypointsExtractor
from .Matchers import Matcher


class EIM(nn.Module):
    def __init__(self, config: DictConfig, device: str='cuda', logger=None) -> None:
        """
        Args:
            config: DictConfig of module config, defined in configs/model/EDM.yaml
            device: device to use
        """
        super().__init__()
        self.device = device
        self.config = config
        self.logger = logger
        
        self.event_extractor = EventKeypointsExtractor(config, logger, device=device)
        self.image_extractor = ImageKeypointsExtractor(config, logger, device=device)
        self.matcher = Matcher(config, logger, device=device)
        
        if config.pretrain_stage1.model_path is not None:
            m = torch.load(config.pretrain_stage1.model_path, map_location=device)
            event_extractor_dict = {k[16:]: v for k, v in m.items() if 'event_extractor' in k}
            self.event_extractor.load_state_dict(event_extractor_dict)
            
            if logger is not None:
                logger.log_info(f"Loaded pretrain_stage1 model from {config.pretrain_stage1.model_path}")
        
        if config.pretrain_stage2.model_path is not None:
            m = torch.load(config.pretrain_stage2.model_path, map_location=device)
            matcher_dict = {k[8:]: v for k, v in m.items() if 'matcher' in k}
            self.matcher.load_state_dict(matcher_dict)
            if logger is not None:
                logger.log_info(f"Loaded pretrain_stage2 model from {config.pretrain_stage2.model_path}")
        
    def forward(self, events: torch.Tensor, image: torch.Tensor, events_mask: torch.Tensor=None, image_mask: torch.Tensor=None) -> Tuple[dict, dict, dict]:
        """
        Args:
            events (torch.Tensor): the input events.
            image (torch.Tensor): the input image.
            
        Returns:
            events_feats (dict): a dict containing:
                - image_size (torch.Size): the original image size.
                - backbone_feats (torch.Tensor): [B, feat_channels, H/8, W/8] the backbone features.
                - logits (torch.Tensor): [B, 65, H/8, W/8] the logits of keypoints.
                - raw_descriptors (torch.Tensor): [B, descriptor_dim, H/8, W/8] the raw descriptors.
                - probability (torch.Tensor): [B, 65, H/8, W/8] the probability of keypoints.
                - score (torch.Tensor): [B, 1, H, W] the score of keypoints.
                - nms (torch.Tensor): [B, H, W] the nms of keypoints.
                - coarse_descriptors (torch.Tensor): [B, descriptor_dim, H/8, W/8] the coarse descriptors.
                - normalized_descriptors (torch.Tensor): [B, descriptor_dim, H, W] the normalized descriptors.
                - dense_descriptors (torch.Tensor): [B, H*W, descriptor_dim] the dense descriptors.
                - sparse_descriptors (Tuple): [B, N_i, descriptor_dim] the sparse descriptors.
                - sparse_positions (Tuple): [B, N_i, 3] the positions of sparse keypoints.
                - dense_positions (torch.Tensor): [B, H*W, 3] the positions of dense keypoints.
            image_feats (dict): a dict containing:
                - image_size (torch.Size): the original image size.
                - backbone_feats (torch.Tensor): [B, feat_channels, H/8, W/8] the backbone features.
                - logits (torch.Tensor): [B, 65, H/8, W/8] the logits of keypoints.
                - raw_descriptors (torch.Tensor): [B, descriptor_dim, H/8, W/8] the raw descriptors.
                - probability (torch.Tensor): [B, 65, H/8, W/8] the probability of keypoints.
                - score (torch.Tensor): [B, 1, H, W] the score of keypoints.
                - nms (torch.Tensor): [B, H, W] the nms of keypoints.
                - coarse_descriptors (torch.Tensor): [B, descriptor_dim, H/8, W/8] the coarse descriptors.
                - normalized_descriptors (torch.Tensor): [B, descriptor_dim, H, W] the normalized descriptors.
                - dense_descriptors (torch.Tensor): [B, H*W, descriptor_dim] the dense descriptors.
                - sparse_descriptors (Tuple): [B, N_i, descriptor_dim] the sparse descriptors.
                - sparse_positions (Tuple): [B, N_i, 3] the positions of sparse keypoints.
                - dense_positions (torch.Tensor): [B, H*W, 3] the positions of dense keypoints.
            matches (dict): a dict containing:
                - matches0: (B, N) tensor of indices of matches in desc1
                - matches1: (B, M) tensor of indices of matches in desc0
                - matching_scores0: (B, N) tensor of matching scores
                - matching_scores1: (B, M) tensor of matching scores
                - matched_kpts0: (B, N, 3) tensor of matched keypoints in image 0
                - matched_kpts1: (B, M, 3) tensor of matched keypoints in image 1
                - similarity: (B, N, M) tensor of descriptor similarity
                - log_assignment: (B, N+1, M+1) tensor of log assignment matrix
        """
        events_feats = self.event_extractor(events, events_mask)
        image_feats = self.image_extractor(image, image_mask)

        if self.matcher.matcher is not None:
            matches = self.matcher(events_feats, image_feats)
        else:
            matches = None
        
        return events_feats, image_feats, matches
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    pass
