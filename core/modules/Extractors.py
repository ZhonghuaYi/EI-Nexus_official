import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_extractors.superpoint_extractor import SuperPointv1
from .image_extractors.silk_extractor import SiLKModel

from .event_extractors.EventExtractors import VGGExtractor, VGGExtractorNP

from rich import pretty, print
from omegaconf import DictConfig


def threshhold_applier(points_heatmap: torch.Tensor, points_indices: torch.Tensor, th: float):
    """
    Args:
        points_heatmap (torch.Tensor): [B, N, C], the heatmap of candidate points.
        points_indices (torch.Tensor): [B, N, 2], the indices (x, y) of candidate points.
        th (float): the threshhold to select keypoints.
    
    Returns:
        keypoints_heatmap (torch.Tensor): [B, N, 1], the heatmap of keypoints.
        keypoints_indices (torch.Tensor): [B, N, 1], the indices of keypoints.
    """
    _, N, C = points_heatmap.shape
    
    pass

    return 


class EventKeypointsExtractor(nn.Module):
    
    def __init__(self, config: DictConfig, logger=None, device: str='cuda'):
        super().__init__()
        self.config = config.event_extractor
        
        self.extractor = None
        self.device = device
        self.freeze = self.config.freeze
        
        self.extractor_type = self.config.type
        self.representation = None
        
        if self.extractor_type == 'vgg':
            extractor = VGGExtractor(
                in_channels=self.config.vgg.in_channels,
                feat_channels=self.config.vgg.feat_channels,
                descriptor_dim=self.config.vgg.descriptor_dim,
                nms_radius=self.config.vgg.nms_radius,
                detection_threshold=self.config.vgg.detection_threshold,
                detection_top_k=self.config.vgg.detection_top_k,
                remove_borders=self.config.vgg.remove_borders,
                ordering=self.config.vgg.ordering,
                descriptor_scale_factor=self.config.vgg.descriptor_scale_factor,
                learnable_descriptor_scale_factor=self.config.vgg.learnable_descriptor_scale_factor,
                use_batchnorm=self.config.vgg.use_batchnorm,
            )
        
        elif self.extractor_type == 'vgg_np':
            extractor = VGGExtractorNP(
                in_channels=self.config.vgg_np.in_channels,
                feat_channels=self.config.vgg_np.feat_channels,
                descriptor_dim=self.config.vgg_np.descriptor_dim,
                nms_radius=self.config.vgg_np.nms_radius,
                detection_threshold=self.config.vgg_np.detection_threshold,
                detection_top_k=self.config.vgg_np.detection_top_k,
                remove_borders=self.config.vgg_np.remove_borders,
                ordering=self.config.vgg_np.ordering,
                descriptor_scale_factor=self.config.vgg_np.descriptor_scale_factor,
                learnable_descriptor_scale_factor=self.config.vgg_np.learnable_descriptor_scale_factor,
                use_batchnorm=self.config.vgg_np.use_batchnorm,
                padding=self.config.vgg_np.padding,
            )
            
        else:
            raise ValueError(f'Unsupported extractor type: {self.extractor_type}')
        
        self.extractor = extractor
        self.extractor.to(device)
        
        if self.freeze:
            for param in self.extractor.parameters():
                param.requires_grad = False
            self.extractor.eval()
        else:
            self.extractor.train()
        
        params = sum(p.numel() for p in self.extractor.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.extractor.parameters())
        
        if logger is not None:
            logger.log_info(f'[bold red]EventKeypointsExtractor[/bold red]:'
                            f' - type: [bold yellow]{self.config.type}[/bold yellow]'
                            f' - freeze: {self.config.freeze}'
                            f' - params: {params / 1e6:.2f}M/ {params}'
                            f' - all_params: {all_params / 1e6:.2f}M/ {all_params}')
            logger.log_info(self.config[self.config.type])

    def forward(self, events: torch.Tensor, score_mask: torch.Tensor=None):
        """
        Args:
            events (torch.Tensor): the input events.
            score_mask (torch.Tensor): the mask of events.
        
        Returns:
            feats (dict): a dict containing:
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
        """
        if self.freeze:
            with torch.no_grad():
                feats = self.extractor(events, score_mask)
        else:
            feats = self.extractor(events, score_mask)
        
        return feats
    

class ImageKeypointsExtractor(nn.Module):
    """ 
    Base class for image keypoints detector.
    """
    def __init__(self, config: DictConfig, logger, device: str='cuda') -> None:
        super().__init__()
        
        self.config = config.image_extractor
        
        self.extractor = None
        self.freeze = self.config.freeze
        self.device = device
        
        self.feature_type = self.config.type
        if self.feature_type == 'superpointv1':
            extractor = SuperPointv1(
                descriptor_dim=self.config.superpointv1.descriptor_dim,
                nms_radius=self.config.superpointv1.nms_radius,
                detection_threshold=self.config.superpointv1.detection_threshold,
                detection_top_k=self.config.superpointv1.detection_top_k,
                ordering=self.config.superpointv1.ordering,
                remove_borders=self.config.superpointv1.remove_borders,
                descriptor_scale_factor=self.config.superpointv1.descriptor_scale_factor,
                learnable_descriptor_scale_factor=self.config.superpointv1.learnable_descriptor_scale_factor,
            )
        elif self.feature_type == 'silk':
            extractor = SiLKModel(
                device=self.device,
                **self.config.silk,
            )
        else:
            raise ValueError(f'Unsupported feature type: {self.feature_type}')

        self.extractor = extractor
        self.extractor.to(device)
        
        if self.freeze:
            for param in self.extractor.parameters():
                param.requires_grad = False
            self.extractor.eval()
        else:
            self.extractor.train()
        
        params = sum(p.numel() for p in self.extractor.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.extractor.parameters())
        
        if logger is not None:
            logger.log_info(f'[bold red]ImageKeypointsExtractor[/bold red]:'
                            f' - type: [bold yellow]{self.config.type}[/bold yellow]'
                            f' - freeze: {self.config.freeze}'
                            f' - params: {params / 1e6:.2f}M/ {params}'
                            f' - all_params: {all_params / 1e6:.2f}M/ {all_params}')
            logger.log_info(self.config[self.config.type])

    def forward(self, image: torch.Tensor, mask=None) -> dict:
        """ 
        extract features from image.
        
        Args:
            image (torch.Tensor): [B, C, H, W] the input image.
        
        Returns:
            feats (dict): a dict containing:
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
        
        """
        assert image.dim() == 4, f'Expected 4D tensor, got {image.dim()}D tensor instead.'
        
        if self.freeze:
            with torch.no_grad():
                feats = self.extractor(image, mask)
        else:
            feats = self.extractor(image, mask)
        
        return feats
    
    
    
