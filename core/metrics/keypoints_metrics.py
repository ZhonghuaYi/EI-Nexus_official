from typing import Dict
import torch
import torch.nn as nn

from ..modules.utils.detector_util import prob_map_to_points_map
from .util import keep_true_points, warp_points


@torch.no_grad()
def detection_metric(
    pred_score,
    gt_score,
    pred_nms,
    gt_nms,
    event_mask,
):

    pred_kpts_map = pred_nms > 0
    gt_kpts_map = gt_nms > 0

    if pred_score.ndim == 4:
        pred_score = pred_score.squeeze(1)
    if gt_score.ndim == 4:
        gt_score = gt_score.squeeze(1)
    if event_mask.ndim == 4:
        event_mask = event_mask.squeeze(1)

    min_kpts = min(pred_kpts_map.sum().float(), gt_kpts_map.sum().float())
    if min_kpts > 0:
        repeatability = (pred_kpts_map & gt_kpts_map).sum().float() / min_kpts.float()
    else:
        repeatability = torch.tensor(0.0)

    pred_avg_probs = pred_score[event_mask].mean()
    pred_avg_pred_probs = pred_score[pred_kpts_map].mean()
    pred_avg_gt_probs = pred_score[gt_kpts_map].mean()
    gt_avg_probs = gt_score[event_mask].mean()
    gt_avg_gt_probs = gt_score[gt_kpts_map].mean()
    gt_avg_pred_probs = gt_score[pred_kpts_map].mean()

    out = {
        "repeatability": repeatability.item(),
        "pred_avg_probs": pred_avg_probs.item(),
        "pred_avg_pred_probs": pred_avg_pred_probs.item(),
        "pred_avg_gt_probs": pred_avg_gt_probs.item(),
        "gt_avg_probs": gt_avg_probs.item(),
        "gt_avg_gt_probs": gt_avg_gt_probs.item(),
        "gt_avg_pred_probs": gt_avg_pred_probs.item(),
    }

    return out


class Repeatability:
    def __init__(self, name, distance_thresh=3, ordering="xy") -> None:
        self.distance_thresh = distance_thresh
        self.metric_name = name
        self.ordering = ordering
        assert self.ordering in ["xy", "yx"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def update_one(self, points1, points2, img1_shape, img2_shape, homography) -> Dict:
        """

        Args:
            points1 (torch.Tensor): [N, 2] tensor
            points2 (torch.Tensor): [M, 2] tensor
            img_shape (Tuple[int, int]): (H, W)
            homography (torch.Tensor): [3, 3] tensor
        """

        out_dict = {}

        points1 = points1.to(self.device)
        points2 = points2.to(self.device)

        assert homography.shape == (3, 3)

        if self.ordering == "xy":
            points1 = points1.T[[0, 1]]  # [2, N]
            points2 = points2.T[[0, 1]]  # [2, M]
        else:
            points1 = points1.T[[1, 0]]  # [2, N]
            points2 = points2.T[[1, 0]]  # [2, M]

        homography = homography.to(self.device).float()

        # only keep points from warped image if they would be present in original image
        points2, _ = keep_true_points(points2, torch.linalg.inv(homography), img1_shape)

        # only keep points from original image if they would be present in warped image
        points1, _ = keep_true_points(points1, homography, img2_shape)

        # warp the original image output_points with the true homography
        true_warped_pred_points = warp_points(points1, homography)

        # need to transpose to properly compute repeatability
        points2 = points2.T
        true_warped_pred_points = true_warped_pred_points.T

        # get the number of predicted points in both images in the pair
        original_num = true_warped_pred_points.shape[0]
        warped_num = points2.shape[0]

        # compute the norm
        assert true_warped_pred_points.shape[1] == 2
        assert points2.shape[1] == 2
        norm = torch.linalg.norm(
            true_warped_pred_points[:, :2].unsqueeze(1) - points2[:, :2].unsqueeze(0),
            dim=2,
        )

        # count the number of points with norm distance less than distance_thresh
        count1 = 0
        count2 = 0

        if original_num != 0:
            min1 = torch.min(norm, 0).values
            count1 = torch.sum(min1 <= self.distance_thresh)
        if warped_num != 0:
            min2 = torch.min(norm, 1).values
            count2 = torch.sum(min2 <= self.distance_thresh)
        if original_num + warped_num > 0:
            repeatability = (count1 + count2) / (original_num + warped_num)
            out_dict[self.metric_name] = repeatability.item()

        return out_dict

    @torch.no_grad()
    def update_batch(
        self, points1, points2, img1_shape, img2_shape, homography
    ) -> Dict:
        """

        Args:
            points1 (Tuple): [B, N, 2] tensor
            points2 (Tuple): [B, M, 2] tensor
            img_shape (Tuple[int, int]): (H, W)
            homography (torch.Tensor): [B, 3, 3] tensor
        """

        out_dict = {}

        assert len(points1) == len(points2) == len(homography)

        repeatability_list = []
        for i in range(len(points1)):
            one_out_dict = self.update_one(
                points1[i], points2[i], img1_shape, img2_shape, homography[i]
            )
            if self.metric_name in one_out_dict:
                repeatability_list.append(one_out_dict[self.metric_name])

        out_dict[self.metric_name] = torch.tensor(repeatability_list).mean().item()
        return out_dict


class ValidDescriptorsDistance:
    def __init__(self, name, distance_thresh_list, ordering="xy") -> None:
        self.distance_thresh_list = distance_thresh_list
        self.metric_name = name
        self.ordering = ordering
        assert self.ordering in ["xy", "yx"]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def update_one(
        self, points1, points2, desc1, desc2, img1_shape, img2_shape, homography
    ) -> Dict:
        """

        Args:
            points1 (torch.Tensor): [N, 2] tensor
            points2 (torch.Tensor): [M, 2] tensor
            desc1 (torch.Tensor): [N, D] tensor
            desc2 (torch.Tensor): [M, D] tensor
            img_shape (Tuple[int, int]): (H, W)
            homography (torch.Tensor): [3, 3] tensor
        """

        out_dict = {}

        points1 = points1.to(self.device)
        points2 = points2.to(self.device)
        desc1 = desc1.to(self.device)
        desc2 = desc2.to(self.device)

        assert homography.shape == (3, 3)

        if self.ordering == "yx":
            points1 = points1.T[[0, 1]]  # [2, N]
            points2 = points2.T[[0, 1]]  # [2, M]
        else:
            points1 = points1.T[[1, 0]]  # [2, N]
            points2 = points2.T[[1, 0]]  # [2, M]

        homography = homography.to(self.device).float()

        # only keep points from warped image if they would be present in original image
        points2, mask2 = keep_true_points(
            points2, torch.linalg.inv(homography), img1_shape
        )
        desc2 = desc2[mask2, :]

        # only keep points from original image if they would be present in warped image
        points1, mask1 = keep_true_points(points1, homography, img2_shape)
        desc1 = desc1[mask1, :]

        # warp the original image output_points with the true homography
        true_warped_pred_points = warp_points(points1, homography)

        # need to transpose to properly compute repeatability
        points2 = points2.T
        true_warped_pred_points = true_warped_pred_points.T

        # compute the repeatability

        # get the number of predicted points in both images in the pair
        original_num = true_warped_pred_points.shape[0]
        warped_num = points2.shape[0]

        # compute the norm
        assert true_warped_pred_points.shape[1] == 2
        assert points2.shape[1] == 2
        norm = torch.linalg.norm(
            true_warped_pred_points[:, :2].unsqueeze(1) - points2[:, :2].unsqueeze(0),
            dim=2,
        )

        for distance_thresh in self.distance_thresh_list:
            # count the number of points with norm distance less than distance_thresh
            count1 = 0
            count2 = 0
            repeatability = torch.tensor(0.0)
            valid_distance = torch.tensor(0.0)
            angle = torch.tensor(0.0)

            if original_num != 0 and warped_num != 0:
                min1, index1 = torch.min(norm, 1)
                valid_points1 = min1 <= distance_thresh
                # out_dict['valid_points1'] = valid_points1.cpu().numpy()
                valid_index1 = index1[valid_points1]
                valid_desc1 = desc1[valid_points1]
                valid_desc2 = desc2[valid_index1]
                distance1 = torch.linalg.norm(valid_desc1 - valid_desc2, dim=1)
                angle1 = torch.acos(
                    torch.sum(valid_desc1 * valid_desc2, dim=1)
                    / (
                        torch.linalg.norm(valid_desc1, dim=1)
                        * torch.linalg.norm(valid_desc2, dim=1)
                    )
                )
                angle1 = torch.rad2deg(angle1)
                count1 = torch.sum(valid_points1)

                min2, index2 = torch.min(norm, 0)
                valid_points2 = min2 <= distance_thresh
                # out_dict['valid_points2'] = valid_points2.cpu().numpy()
                valid_index2 = index2[valid_points2]
                valid_desc2 = desc2[valid_points2]
                valid_desc1 = desc1[valid_index2]
                distance2 = torch.linalg.norm(valid_desc1 - valid_desc2, dim=1)
                angle2 = torch.acos(
                    torch.sum(valid_desc1 * valid_desc2, dim=1)
                    / (
                        torch.linalg.norm(valid_desc1, dim=1)
                        * torch.linalg.norm(valid_desc2, dim=1)
                    )
                )
                angle2 = torch.rad2deg(angle2)
                count2 = torch.sum(valid_points2)

                repeatability = (count1 + count2) / (original_num + warped_num)
                valid_distance = (torch.sum(distance1) + torch.sum(distance2)) / (
                    count1 + count2
                )
                angle = (torch.sum(angle1) + torch.sum(angle2)) / (count1 + count2)

            out_dict[f"{self.metric_name}_Repeatability@{distance_thresh}"] = (
                repeatability.item()
            )
            out_dict[f"{self.metric_name}_ValidDistance@{distance_thresh}"] = (
                valid_distance.item()
            )
            out_dict[f"{self.metric_name}_Angle@{distance_thresh}"] = angle.item()

        return out_dict

    @torch.no_grad()
    def update_batch(
        self, points1, points2, desc1, desc2, img1_shape, img2_shape, homography
    ) -> Dict:
        """

        Args:
            points1 (Tuple): [B, N, 2] tensor
            points2 (Tuple): [B, M, 2] tensor
            desc1 (Tuple): [B, N, D] tensor
            desc2 (Tuple): [B, M, D] tensor
            img_shape (Tuple[int, int]): (H, W)
            homography (torch.Tensor): [B, 3, 3] tensor
        """

        out_dict = {}

        assert (
            len(points1) == len(points2) == len(desc1) == len(desc2) == len(homography)
        )

        for i in range(len(points1)):
            one_out_dict = self.update_one(
                points1[i],
                points2[i],
                desc1[i],
                desc2[i],
                img1_shape,
                img2_shape,
                homography[i],
            )
            for key, value in one_out_dict.items():
                out_dict[key] = out_dict.get(key, []) + [value]

        for k in out_dict.keys():
            out_dict[k] = torch.tensor(out_dict[k]).mean().item()

        return out_dict
