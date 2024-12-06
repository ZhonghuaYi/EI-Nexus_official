# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

# Adapted by Remi Pautrat, Philipp Lindenberger

from typing import Callable, List, Optional, Tuple, Union
import torch
import kornia
from kornia.color import rgb_to_grayscale
from torch import nn


import cv2
from pathlib import Path
import numpy as np

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


def read_image(path: Path, grayscale: bool = True) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str = "max",
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    image = read_image(path)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)


class ImagePreprocessor:

    def __init__(
        self,
        resize=None,
        side="long",
        interpolation="bilinear",
        align_corners=None,
        antialias=True,
    ) -> None:
        super().__init__()
        self.resize = resize
        self.side = side
        self.interpolation = interpolation
        self.align_corners = align_corners
        self.antialias = antialias

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        if self.resize is not None:
            img = kornia.geometry.transform.resize(
                img,
                self.resize,
                side=self.side,
                antialias=self.antialias,
                align_corners=self.align_corners,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        return img, scale


def select_top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def pad_to_length(
    x,
    length: int,
    pad_dim: int = -2,
    mode: str = "zeros",  # zeros, ones, random, random_c
    bounds: Tuple[int] = (None, None),
):
    shape = list(x.shape)
    d = x.shape[pad_dim]
    assert d <= length
    if d == length:
        return x
    shape[pad_dim] = length - d

    low, high = bounds

    if mode == "zeros":
        xn = torch.zeros(*shape, device=x.device, dtype=x.dtype)
    elif mode == "ones":
        xn = torch.ones(*shape, device=x.device, dtype=x.dtype)
    elif mode == "random":
        low = low if low is not None else x.min()
        high = high if high is not None else x.max()
        xn = torch.empty(*shape, device=x.device).uniform_(low, high)
    elif mode == "random_c":
        low, high = bounds  # we use the bounds as fallback for empty seq.
        xn = torch.cat(
            [
                torch.empty(*shape[:-1], 1, device=x.device).uniform_(
                    x[..., i].min() if d > 0 else low,
                    x[..., i].max() if d > 0 else high,
                )
                for i in range(shape[-1])
            ],
            dim=-1,
        )
    else:
        raise ValueError(mode)
    return torch.cat([x, xn], dim=pad_dim)


def pad_and_stack(
    sequences: List[torch.Tensor],
    length: Optional[int] = None,
    pad_dim: int = -2,
    **kwargs,
):
    if length is None:
        length = max([x.shape[pad_dim] for x in sequences])

    y = torch.stack([pad_to_length(x, length, pad_dim, **kwargs) for x in sequences], 0)
    return y


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def top_k_keypoints(keypoints, scores, k):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0, sorted=True)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


class SuperPointv1(nn.Module):
    def __init__(
        self,
        descriptor_dim=256,
        nms_radius=4,
        detection_top_k: int = 2048,
        detection_threshold: float = 0.0005,
        remove_borders: int = 4,
        ordering: str = "yx",
        descriptor_scale_factor: float = 1.0,
        learnable_descriptor_scale_factor: bool = False,
    ):
        super().__init__()

        self.descriptor_dim = descriptor_dim
        self.nms_radius = nms_radius
        self.detection_threshold = detection_threshold
        self.remove_borders = remove_borders
        self.detection_top_k = detection_top_k
        self.cell_size = 8

        assert ordering in ("xy", "yx")
        self.ordering = ordering

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.descriptor_dim, kernel_size=1, stride=1, padding=0
        )

        url = "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth"  # noqa
        self.load_state_dict(torch.hub.load_state_dict_from_url(url))

        self.descriptor_scale_factor = nn.parameter.Parameter(
            torch.tensor(descriptor_scale_factor),
            requires_grad=learnable_descriptor_scale_factor,
        )

    def filter_sparse_feats(self, positions, descs, image_shape):
        h, w = image_shape
        assert len(positions) == len(descs)
        new_positions = []
        new_descs = []

        for i in range(len(positions)):
            position = positions[i]
            desc = descs[i]
            if self.ordering == "yx":
                x_valid = (position[:, 1] >= 0) & (position[:, 1] < w)
                y_valid = (position[:, 0] >= 0) & (position[:, 0] < h)
            elif self.ordering == "xy":
                x_valid = (position[:, 0] >= 0) & (position[:, 0] < w)
                y_valid = (position[:, 1] >= 0) & (position[:, 1] < h)
            valid = x_valid & y_valid
            new_positions.append(position[valid])
            new_descs.append(desc[valid])

        return new_positions, new_descs

    def forward(self, image: torch.Tensor, score_mask=None) -> dict:
        """Compute keypoints, scores, descriptors for image

        Args:
            img (torch.Tensor): [B, C, H, W] the input image.

        Returns:
            a dict containing:
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

        assert (
            image.dim() == 4
        ), f"Expected 4D tensor, got {image.dim()}D tensor instead."

        image /= 255.0

        image_size = image.shape[-2:]
        if image.shape[1] == 3:
            image = rgb_to_grayscale(image)

        # init padder
        padder = Padder(image.shape, self.cell_size)

        # pad the x and mask
        image = padder.pad(image)[0]
        if score_mask is not None:
            score_mask = padder.pad(score_mask)[0]

        padded_size = image.shape[-2:]
        # Shared Encoder
        x = self.relu(self.conv1a(image))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        feats = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(feats))
        logits = self.convPb(cPa)

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(feats))
        raw_descriptors = self.convDb(cDa)

        # detector postprocess
        probability = logits_to_prob(logits, channel_dim=1)
        score = depth_to_space(probability, cell_size=self.cell_size)
        if score_mask is not None:
            score[~score_mask] = 0.
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

        # descriptor postprocess
        coarse_descriptors = normalize_descriptors(
            raw_descriptors, scale_factor=self.descriptor_scale_factor, normalize=True
        )  # [B, C, H/8, W/8]
        sparse_descriptors = sparsify_low_resolution_descriptors(
            raw_descriptors,
            positions,
            padded_size,
            scale_factor=self.descriptor_scale_factor,
            normalize=True,
        )
        upsampled_descriptors = upsample_descriptors(
            raw_descriptors, padded_size, scale_factor=self.descriptor_scale_factor
        )  # [B, C, H, W]
        normalized_descriptors = upsampled_descriptors  # [B, C, H, W]
        dense_descriptors = get_dense_descriptors(normalized_descriptors)  # [B, H*W, C]
        # sparse_positions = positions
        dense_positions = get_dense_positions(score, ordering=self.ordering)

        # unpad the results
        score, nms, normalized_descriptors = padder.unpad(
            score, nms, normalized_descriptors
        )
        positions = padder.unpad_positions(positions, self.ordering)
        # sparse_positions = positions
        dense_positions = padder.unpad_positions(
            dense_positions, ordering=self.ordering
        )

        # filter the postions and descriptors
        positions, sparse_descriptors = self.filter_sparse_feats(
            positions, sparse_descriptors, image_size
        )
        sparse_positions = positions
        dense_positions, dense_descriptors = self.filter_sparse_feats(
            dense_positions, dense_descriptors, image_size
        )

        image_size_list = [torch.tensor(image_size, device=x.device)] * x.shape[0]
        out_dict = {
            "image_size": image_size_list,
            "backbone_feats": feats,
            "logits": logits,
            "raw_descriptors": raw_descriptors,
            "probability": probability,
            "score": score,
            "nms": nms,
            "coarse_descriptors": coarse_descriptors,
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

    import cv2 as cv

    image = torch.tensor(cv.imread("./frame_0010.png", 0))
    image_gray = cv.imread("./frame_0010.png", 1)
    image = image.unsqueeze(0).unsqueeze(0).float()

    image1 = torch.tensor(cv.imread("./frame_0010.png", 0))
    image1 = image1.unsqueeze(0).unsqueeze(0).float()

    images = torch.cat([image, image1], dim=0)
    images = images.to("cuda")

    superpoint = SuperPointv1().to("cuda").eval()
    out_dict = superpoint(images)

    image0 = image_gray.copy()
    kpts0 = out_dict["sparse_positions"][0][:, :2].cpu().detach().numpy()[:, ::-1]
    kp_img = draw_keypoints(image0, kpts0)

    import matplotlib.pyplot as plt

    plt.imshow(kp_img)
    plt.show()

    # for k, v in out_dict.items():
    #     if isinstance(v, torch.Tensor):
    #         print(k, v.shape)
    #     elif isinstance(v, torch.Size):
    #         print(k, v)
    #     else:
    #         print(k, type(v))

    # keypoints = out_dict['keypoints'][0]
    # keypoints_scores = out_dict['keypoint_scores'][0]
    # keypoints_desc = out_dict['keypoint_descriptors'][0]
    # print((keypoints_scores > 0.005).sum())
    # print(keypoints[0])
    # print(keypoints_desc[0, :10])

    # from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED

    # print('---official superpoint---')
    # image = load_image('test.png')
    # print(image.shape)
    # extractor = SuperPoint(max_num_keypoints=2048).eval()
    # feats0 = extractor.extract(image)
    # desc = feats0['descriptors'][0]
    # kpts = feats0['keypoints'][0]
    # kpts_scores = feats0['keypoint_scores'][0]
    # print((kpts_scores > 0.005).sum())
    # print(kpts[0])
    # print(desc[0, :10])
