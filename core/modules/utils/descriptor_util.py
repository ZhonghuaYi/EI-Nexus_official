# From SiLK library https://github.com/facebookresearch/silk

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils functions for the magicpoint model.
"""

import math
from typing import Iterable, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, resize


def normalize_descriptors(raw_descriptors, scale_factor=1.0, normalize=True):
    if normalize:
        return scale_factor * F.normalize(
            raw_descriptors,
            p=2,
            dim=1,
        )  # L2 normalization
    return scale_factor * raw_descriptors


def upsample_descriptors(raw_descriptors, image_size, scale_factor: float = 1.0):
    upsampled_descriptors = resize(
        raw_descriptors,
        image_size,
        interpolation=InterpolationMode.BILINEAR,
    )
    return normalize_descriptors(upsampled_descriptors, scale_factor)


def get_dense_descriptors(normalized_descriptors):
    dense_descriptors = normalized_descriptors.reshape(
        normalized_descriptors.shape[0],
        normalized_descriptors.shape[1],
        -1,
    )
    dense_descriptors = dense_descriptors.permute(0, 2, 1)
    return dense_descriptors
    

def sparsify_full_resolution_descriptors(
    raw_descriptors,
    positions,
    scale_factor: float = 1.0,
    normalize: bool = True,
):
    sparse_descriptors = []
    for i, pos in enumerate(positions):
        pos = pos[:, :2]
        pos = pos.floor().long()

        descriptors = raw_descriptors[i, :, pos[:, 0], pos[:, 1]].T

        # L2 normalize the descriptors
        descriptors = normalize_descriptors(
            descriptors,
            scale_factor,
            normalize,
        )

        sparse_descriptors.append(descriptors)
    return tuple(sparse_descriptors)


def sparsify_low_resolution_descriptors(
        raw_descriptors,
        positions,
        image_size,
        scale_factor: float = 1.0,
        normalize: bool = True,
    ):
    image_size = torch.tensor(
        image_size,
        dtype=torch.float,
        device=raw_descriptors.device,
    )
    sparse_descriptors = []

    for i, pos in enumerate(positions):
        pos = pos[:, :2]
        n = pos.shape[0]

        # handle edge case when no points has been detected
        if n == 0:
            desc = raw_descriptors[i]
            fdim = desc.shape[0]
            sparse_descriptors.append(
                torch.zeros(
                    (n, fdim),
                    dtype=desc.dtype,
                    device=desc.device,
                )
            )
            continue

        # revert pixel centering for grad sample
        pos = pos - 0.5

        # normalize to [-1. +1] & prepare for grid sample
        pos = 2.0 * (pos / (image_size - 1)) - 1.0
        pos = pos[:, [1, 0]]
        pos = pos[None, None, ...]

        # process descriptor output by interpolating into descriptor map using 2D point locations\
        # note that grid_sample takes coordinates in x, y order (col, then row)
        descriptors = raw_descriptors[i][None, ...]
        descriptors = F.grid_sample(
            descriptors,
            pos,
            mode="bilinear",
            align_corners=False,
        )
        descriptors = descriptors.view(-1, n).T

        # L2 normalize the descriptors
        descriptors = normalize_descriptors(descriptors, scale_factor, normalize)

        sparse_descriptors.append(descriptors)
    return sparse_descriptors


def upsample_descriptors(raw_descriptors, image_size, scale_factor: float = 1.0):
    upsampled_descriptors = resize(
        raw_descriptors,
        image_size,
        interpolation=InterpolationMode.BILINEAR,
        antialias=None
    )
    return normalize_descriptors(upsampled_descriptors, scale_factor)


