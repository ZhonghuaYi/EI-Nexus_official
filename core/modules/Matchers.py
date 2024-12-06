from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .matchers.MNN import NearestNeighborMatcher
from .matchers.lightglue import LightGlue

from rich import pretty, print
from omegaconf import DictConfig


class Matcher(nn.Module):
    def __init__(self, config: DictConfig, logger=None, device: str='cuda') -> None:
        super().__init__()
        self.config = config.matcher

        self.matcher = None
        self.freeze = self.config.freeze
        self.max_points_num = self.config.max_points_num
        self.pad_mode = self.config.pad_mode
        self.desc_scale_factor = self.config.desc_scale_factor
        
        self.matcher_type = self.config.type
        
        if self.matcher_type == 'MNN':
            self.matcher = NearestNeighborMatcher(
                ratio_thresh=self.config.MNN.ratio_thresh,
                distance_thresh=self.config.MNN.distance_thresh,
                mutual_check=True,
            )
        elif self.matcher_type == 'LightGlue':
            self.matcher = LightGlue(
                conf=self.config.LightGlue,
            )
        elif self.matcher_type is None:
            self.matcher = None
        else:
            raise NotImplementedError

        if self.matcher is not None:
            self.matcher.to(device)
            
            if self.freeze:
                for param in self.matcher.parameters():
                    param.requires_grad = False
                self.matcher.eval()
            else:
                self.matcher.train()
            
            params = sum(p.numel() for p in self.matcher.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.matcher.parameters())
            
            if logger is not None:
                logger.log_info(f'[bold red]Matcher[/bold red]:'
                                f' - type: [bold yellow]{self.config.type}[/bold yellow]'
                                f' - freeze: {self.config.freeze}'
                                f' - params: {params / 1e6:.2f}M/ {params}'
                                f' - all_params: {all_params / 1e6:.2f}M/ {all_params}')
                logger.log_info(self.config[self.config.type])
        else:
            if logger is not None:
                logger.log_info(f'[bold red]Matcher[/bold red]:'
                                f' - type: [bold yellow]{self.config.type}[/bold yellow]'
                                f' - freeze: {self.config.freeze}')
    
    def pad_sparse_positions_to_length(self, sparse_positions, length, image_size=None):
        """
        Pad the sparse positions to the given length.
        
        Args:
            sparse_positions (torch.Tensor): the sparse positions. [N, 3]
            length (int): the length to pad to.
            image_size (Tuple[int, int]): the image size. [W, H]
        
        Returns:
            sparse_positions (torch.Tensor): the padded sparse positions. [length, 3]
        """
        if len(sparse_positions) < length:
            random_sample_len = length - len(sparse_positions)
            if self.pad_mode == 'zeros':
                random_positions = torch.zeros(random_sample_len, 3)
            elif self.pad_mode == 'random':
                if image_size is None:
                    image_size = sparse_positions[:, 0].max(), sparse_positions[:, 1].max()
                if isinstance(image_size, list):
                    image_size = image_size[0]
                random_positions = torch.rand(random_sample_len, 2, device=sparse_positions.device) * torch.tensor(image_size).float()
                random_scores = torch.zeros(random_sample_len, 1, device=sparse_positions.device)
                random_positions = torch.cat([random_positions, random_scores], dim=1)
            else:
                raise NotImplementedError(f'Unknown mode: {self.pad_mode}')
            
            random_positions = random_positions.to(sparse_positions.device)
            sparse_positions = torch.cat([sparse_positions, random_positions], dim=0)
        elif len(sparse_positions) > length:
            sparse_positions = sparse_positions[:length, ...]
        
        return sparse_positions
    
    def pad_sparse_descriptors_to_length(self, sparse_descriptors, length):
        """
        Pad the sparse descriptors to the given length.
        
        Args:
            sparse_descriptors (torch.Tensor): the sparse descriptors. [N, C]
            length (int): the length to pad to.
            mode (str): the mode to pad. 'zeros' or 'random'
        """
        if len(sparse_descriptors) < length:
            random_sample_len = length - len(sparse_descriptors)
            if self.pad_mode == 'zeros':
                random_descriptors = torch.zeros(random_sample_len, sparse_descriptors.shape[1])
            elif self.pad_mode == 'random':
                random_descriptors = torch.randn(random_sample_len, sparse_descriptors.shape[1])
            else:
                raise NotImplementedError(f'Unknown mode: {self.pad_mode}')

            # normalize the random descriptors
            random_descriptors = F.normalize(random_descriptors, dim=1) * self.desc_scale_factor
            random_descriptors = random_descriptors.to(sparse_descriptors.device)
            
            sparse_descriptors = torch.cat([sparse_descriptors, random_descriptors], dim=0)
        elif len(sparse_descriptors) > length:
            sparse_descriptors = sparse_descriptors[:length, ...]
            
        return sparse_descriptors
    
    def pad_sparse_feats_to_length(self, feats, length):
        sparse_positions = feats['sparse_positions']
        sparse_descriptors = feats['sparse_descriptors']
        image_size = feats['image_size'][::-1]
        out_sparse_postions = []
        out_sparse_descriptors = []
        for i in range(len(sparse_positions)):
            sparse_positions_i = self.pad_sparse_positions_to_length(sparse_positions[i], length, image_size)
            sparse_descriptors_i = self.pad_sparse_descriptors_to_length(sparse_descriptors[i], length)
            out_sparse_postions.append(sparse_positions_i)
            out_sparse_descriptors.append(sparse_descriptors_i)
        feats['sparse_positions'] = out_sparse_postions
        feats['sparse_descriptors'] = out_sparse_descriptors
        return feats
    
    def stack_sparse_feats(self, feats):
        sparse_positions = torch.stack(feats['sparse_positions'], dim=0)
        sparse_descriptors = torch.stack(feats['sparse_descriptors'], dim=0)
        feats['sparse_positions'] = sparse_positions
        feats['sparse_descriptors'] = sparse_descriptors
        return feats
        
    def forward(self, feats0: torch.Tensor, feats1: torch.Tensor, *args, **kargs) -> Dict:
        """
        Args:
            feats0 (torch.Tensor): the input features.
            feats1 (torch.Tensor): the input features.
        Returns:
            dict[str, Tensor]: a dictionary of matching results, including:
                - matches0: (B, N) tensor of indices of matches in desc1
                - matches1: (B, M) tensor of indices of matches in desc0
                - matching_scores0: (B, N) tensor of matching scores
                - matching_scores1: (B, M) tensor of matching scores
                - matched_kpts0: (B, N, 3) tensor of matched keypoints in image 0
                - matched_kpts1: (B, M, 3) tensor of matched keypoints in image 1
                - similarity: (B, N, M) tensor of descriptor similarity
                - log_assignment: (B, N+1, M+1) tensor of log assignment matrix
        """
        
        if self.freeze:
            with torch.no_grad():
                if self.matcher is None:
                    return {
                        'matches0': None,
                        'matches1': None,
                        'matching_scores0': None,
                        'matching_scores1': None,
                        'similarity': None,
                        'log_assignment': None
                    }
                elif self.matcher_type == 'MNN' or self.matcher_type == 'LightGlue':
                    out_dict = {
                        'matches0': [],
                        'matches1': [],
                        'matching_scores0': [],
                        'matching_scores1': [],
                        'matched_kpts0': [],
                        'matched_kpts1': [],
                        # 'similarity': [],
                        'log_assignment': []
                    }
                    # feats0 = self.pad_sparse_feats_to_length(feats0, self.max_points_num)
                    # feats1 = self.pad_sparse_feats_to_length(feats1, self.max_points_num)
                    for i in range(len(feats0['sparse_positions'])):
                        feats0_i = {}
                        feats1_i = {}
                        for k in feats0.keys():
                            feats0_i[k] = feats0[k][i][None, ...]
                        for k in feats1.keys():
                            feats1_i[k] = feats1[k][i][None, ...]
                        out_dict_i = self.matcher(feats0_i, feats1_i)
                        
                        out_dict = {k: out_dict[k] + [out_dict_i[k]] for k in out_dict}
                    
                    return out_dict
        else:
            if self.matcher is None:
                return {
                    'matches0': None,
                    'matches1': None,
                    'matching_scores0': None,
                    'matching_scores1': None,
                    'similarity': None,
                    'log_assignment': None
                }
            elif self.matcher_type == 'MNN' or self.matcher_type == 'LightGlue':
                feats0 = self.pad_sparse_feats_to_length(feats0, self.max_points_num)
                feats1 = self.pad_sparse_feats_to_length(feats1, self.max_points_num)
                feats0 = self.stack_sparse_feats(feats0)
                feats1 = self.stack_sparse_feats(feats1)
                matches = self.matcher(feats0, feats1)
                matches['input_feats0'] = feats0
                matches['input_feats1'] = feats1
                return matches
