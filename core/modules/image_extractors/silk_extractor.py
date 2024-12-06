# This file is modified from the original file in the following repository: https://github.com/facebookresearch/silk

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
import os
from os import PathLike
from copy import deepcopy

import numpy as np
import skimage.io as io
import torch
import torch.nn as nn


# from silk.cli.image_pair_visualization import create_img_pair_visual, save_image

from .silk.backbones.silk.silk import SiLKVGG as SiLK
from .silk.backbones.superpoint.vgg import ParametricVGG

from .silk.config.model import load_model_from_checkpoint
from ..utils.detector_util import (
    logits_to_prob,
    depth_to_space,
    prob_map_to_points_map,
    prob_map_to_positions_with_prob,
    get_dense_positions,
)
from ..utils.descriptor_util import (
    normalize_descriptors,
    get_dense_descriptors,
    sparsify_full_resolution_descriptors,
    sparsify_low_resolution_descriptors,
    upsample_descriptors,
)
from ..utils.util import Padder
# from silk.models.silk import matcher

# CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../../assets/models/silk/analysis/alpha/pvgg-4.ckpt")
# CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "silk/pvgg-4.ckpt")
# DEVICE = "cuda:0"

# SILK_NMS = 4  # NMS radius, 0 = disabled
# SILK_BORDER = 0  # remove detection on border, 0 = disabled
# SILK_THRESHOLD = 0.5  # keypoint score thresholding, if # of keypoints is less than provided top-k, then will add keypoints to reach top-k value, 1.0 = disabled
# SILK_TOP_K = 1024  # minimum number of best keypoints to output, could be higher if threshold specified above has low value
# SILK_DEFAULT_OUTPUT = (  # outputs required when running the model
#     "dense_positions",
#     "normalized_descriptors",
#     "probability",
# )
# SILK_SCALE_FACTOR = 1.41  # scaling of descriptor output, do not change
# SILK_BACKBONE = ParametricVGG(
#     use_max_pooling=False,
#     padding=0,
#     normalization_fn=[torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)],
# )
# SILK_MATCHER = matcher(postprocessing="ratio-test", threshold=0.6)
# SILK_MATCHER = matcher(postprocessing="double-softmax", threshold=0.6, temperature=0.1)
# SILK_MATCHER = matcher(postprocessing="none")


# def load_images(*paths, as_gray=True):
#     images = np.stack([io.imread(path, as_gray=as_gray) for path in paths])
#     images = torch.tensor(images, device=DEVICE, dtype=torch.float32)
#     if not as_gray:
#         images = images.permute(0, 3, 1, 2)
#         images = images / 255.0
#     else:
#         images = images.unsqueeze(1)  # add channel dimension
#     return images


class SiLKModel(nn.Module):
    def __init__(self, 
                 device,
                 padding: int,
                 nms_radius: int=4,
                 detection_top_k: int=2048,
                 detection_threshold: float=0.0005, 
                 remove_borders: int=4,
                 ordering: str = "yx",
                 descriptor_scale_factor: float = 1.0,
                 learnable_descriptor_scale_factor: bool = False,) -> None:
        super().__init__()
        
        self.device = device
        self.padding = padding
        self.nms_radius = nms_radius
        self.detection_top_k = detection_top_k
        self.detection_threshold = detection_threshold
        self.remove_borders = remove_borders
        self.ordering = ordering
        self.cell_size = 1
        
        self.descriptor_scale_factor = nn.parameter.Parameter(
            torch.tensor(descriptor_scale_factor),
            requires_grad=learnable_descriptor_scale_factor,
        )
        
        self.CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "silk/pvgg-4.ckpt")
        self.SILK_SCALE_FACTOR = 1.41
        self.SILK_BACKBONE = ParametricVGG(
            use_max_pooling=False,
            padding=self.padding,
            normalization_fn=[torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)],
            )
        
        self.default_outputs = (
            "features",
            "logits",
            "raw_descriptors",
        )
        
        self.model = self.get_model()
        
    def filter_sparse_feats(self, positions, descs, image_shape):
        h, w = image_shape
        assert len(positions) == len(descs)
        new_positions = []
        new_descs = []
        
        for i in range(len(positions)):
            position = positions[i]
            desc = descs[i]
            if self.ordering == 'yx':
                x_valid = (position[:, 1] >= 0) & (position[:, 1] < w)
                y_valid = (position[:, 0] >= 0) & (position[:, 0] < h)
            elif self.ordering == 'xy':
                x_valid = (position[:, 0] >= 0) & (position[:, 0] < w)
                y_valid = (position[:, 1] >= 0) & (position[:, 1] < h)
            valid = x_valid & y_valid 
            new_positions.append(position[valid])
            new_descs.append(desc[valid])
        
        return new_positions, new_descs 
        
    def mapping_positions(self, positions: torch.Tensor) -> torch.Tensor:
        
        if isinstance(positions, tuple):
            return tuple(self.mapping_positions(p) for p in positions)
        
        assert self.padding in (0, 1)
        if self.padding == 0:
            positions[..., 0] = positions[..., 0] + 9.
            positions[..., 1] = positions[..., 1] + 9.
        
        return positions
        
    def get_model(self):
        # load model
        model = SiLK(
            in_channels=1,
            backbone=deepcopy(self.SILK_BACKBONE),
            detection_threshold=self.detection_threshold,
            detection_top_k=self.detection_top_k,
            nms_dist=self.nms_radius,
            border_dist=self.remove_borders,
            default_outputs=self.default_outputs,
            descriptor_scale_factor=self.SILK_SCALE_FACTOR,
            padding=self.padding,
        )
        model = load_model_from_checkpoint(
            model,
            checkpoint_path=self.CHECKPOINT_PATH,
            state_dict_fn=lambda x: {k[len("_mods.model.") :]: v for k, v in x.items()},
            device=self.device,
            freeze=True,
            eval=True,
        )
        return model
        
    def forward(self, image, *args, **kwargs):
        image = image / 255.0
        image_size = image.shape[-2:]
        
        # init padder
        padder = Padder(image.shape, self.cell_size)

        # pad the x
        image = padder.pad(image)[0]
        padded_size = image.shape[-2:]
        
        out_value = self.model(image)
        out_dict = {self.default_outputs[i]: out_value[i] for i in range(len(self.default_outputs))}
        
        features = out_dict['features']
        logits = out_dict['logits']
        raw_descriptors = out_dict['raw_descriptors']
        
        # detector postprocess
        probability = logits_to_prob(logits, channel_dim=1)
        score = depth_to_space(probability, cell_size=self.cell_size)
        nms = prob_map_to_points_map(
            score,
            prob_thresh=self.detection_threshold,
            nms_dist=self.nms_radius,
            border_dist=self.remove_borders,
            use_fast_nms=True,
            top_k=self.detection_top_k,
        )
        positions = prob_map_to_positions_with_prob(
            nms, threshold=0.0, ordering=self.ordering
        )
        dense_positions = get_dense_positions(
            score, ordering=self.ordering
        )  # in (y, x) order
        
        # descriptor postprocess
        normalized_descriptors = normalize_descriptors(
            raw_descriptors, scale_factor=self.descriptor_scale_factor, normalize=True
        )
        dense_descriptors = get_dense_descriptors(normalized_descriptors)
        sparse_descriptors = sparsify_full_resolution_descriptors(
            raw_descriptors,
            positions,
            scale_factor=self.descriptor_scale_factor,
            normalize=True,
        )
        
        # unpad the results
        score, nms, normalized_descriptors = padder.unpad(score, nms, normalized_descriptors)
        positions = padder.unpad_positions(positions, self.ordering)
        # sparse_positions = positions
        dense_positions = padder.unpad_positions(dense_positions, ordering=self.ordering)
        
        # # filter the postions and descriptors
        positions, sparse_descriptors = self.filter_sparse_feats(positions, sparse_descriptors, image_size)
        sparse_positions = positions
        dense_positions, dense_descriptors = self.filter_sparse_feats(dense_positions, dense_descriptors, image_size)

        positions = self.mapping_positions(positions)
        sparse_positions = positions
        dense_positions = self.mapping_positions(dense_positions)

        image_size_list = [torch.tensor(image_size, device=image.device)] * image.shape[0]
        
        out_dict = {
            "image_size": image_size_list,
            "backbone_feats": features,
            "logits": logits,
            "raw_descriptors": raw_descriptors,
            "probability": probability,
            "score": score,
            "nms": nms,
            "normalized_descriptors": normalized_descriptors,
            "dense_descriptors": dense_descriptors,
            "sparse_descriptors": sparse_descriptors,
            "sparse_positions": sparse_positions,
            "dense_positions": dense_positions,
        }
        
        return out_dict
    
    
if __name__ == "__main__":
    def draw_keypoints(image, keypoints):
        import cv2 as cv
        for i in range(len(keypoints)):
            x, y = keypoints[i]
            cv.circle(image, (int(x), int(y)), 1, (0, 255, 0), 1)
            
        return image
    
    
    
    # import cv2 as cv
    # image = torch.tensor(cv.imread('../../../data/Caltech101/train/accordion/image_0001.jpg', 0))
    # image_gray = cv.imread('../../../data/Caltech101/train/accordion/image_0001.jpg', 1)
    # image = image.unsqueeze(0).unsqueeze(0).float()
    
    # # image1 = torch.tensor(cv.imread('../../../data/Caltech101/train/accordion/image_0001.jpg', 0))
    # # image1 = image1.unsqueeze(0).unsqueeze(0).float()
    
    # images = image
    # images = images.to('cuda')
    
    # model = SiLKModel()
    
    # out = model(images)
    
    # image0 = image_gray.copy()
    # kpts0 = out['sparse_positions'][0][:, :2].cpu().detach().numpy()[:, ::-1]
    # kpts0 += 9.
    # kp_img = draw_keypoints(image0, kpts0)
    
    # import matplotlib.pyplot as plt
    
    
    # for k, v in out.items():
    #     print(k, v.shape if isinstance(v, torch.Tensor) else type(v))
    # print('padding: 0-0')
    # print(out["sparse_positions"][0][0])
    # print(out["sparse_descriptors"][0][0])
    # print('probability shape: ', out['probability'].shape)
    # print('dense_positions shape: ', out['dense_positions'].shape)
    # print('dense_positions begin: ', out['dense_positions'][0][0])
    # print('dense_positions end: ', out['dense_positions'][0][-1])

