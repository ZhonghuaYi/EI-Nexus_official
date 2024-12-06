from typing import List, Dict, Any, Tuple
import torch
import numpy as np
import cv2
from .util import warp_points


def compute_auc(errors, thresholds) -> Dict:
    if isinstance(errors, list):
        errors = np.array(errors)

    # remove inf or nan
    errors = errors[np.isfinite(errors)].astype(np.float32)

    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]

    aucs = {}
    for thres in thresholds:
        last_index = np.searchsorted(errors, thres)
        rec = np.r_[recall[:last_index], recall[last_index - 1]]
        err = np.r_[errors[:last_index], thres]
        aucs[f"{thres}"] = np.trapz(rec, x=err) / thres
    return aucs


class MatchingRatio:
    def __init__(self, name) -> None:
        self.metric_name = name

    def update_one(
        self,
        matched_keypoints1: torch.Tensor,
        matched_keypoints2: torch.Tensor,
        keypoints1: torch.Tensor,
        keypoints2: torch.Tensor,
    ):
        out_dict = {}
        keypoints_length = min(len(keypoints1), len(keypoints2))

        assert len(matched_keypoints1) == len(matched_keypoints2)
        matched_keypoints_length = len(matched_keypoints1)

        ratio = matched_keypoints_length / (keypoints_length + 1e-8)

        out_dict[self.metric_name] = ratio

        return out_dict

    def update_batch(
        self,
        matched_keypoints1: List[torch.Tensor],
        matched_keypoints2: List[torch.Tensor],
        keypoints1: List[torch.Tensor],
        keypoints2: List[torch.Tensor],
    ):
        out_dict = {}
        assert (
            len(matched_keypoints1)
            == len(matched_keypoints2)
            == len(keypoints1)
            == len(keypoints2)
        )

        MR_list = []
        for i in range(len(matched_keypoints1)):
            one_out_dict = self.update_one(
                matched_keypoints1[i],
                matched_keypoints2[i],
                keypoints1[i],
                keypoints2[i],
            )
            if self.metric_name in one_out_dict:
                MR_list.append(one_out_dict[self.metric_name])

        out_dict[self.metric_name] = torch.tensor(MR_list).mean().item()

        return out_dict


class MeanMatchingAccuracy:
    def __init__(self, name, threshold=3, ordering="yx") -> None:
        self.metric_name = name
        self._threshold = threshold
        self._cum_acc = 0
        self._n_matches = 0
        self._n_points = 0
        self._ordering = ordering

        assert ordering in {"xy", "yx"}
        self.to_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def good_matches_mask(
        self,
        matched_keypoints,
        warped_matched_keypoints,
        homography,
        threshold,
        ordering,
    ):
        assert ordering in {"yx", "xy"}

        if ordering == "xy":
            matched_keypoints = matched_keypoints[:, :2]
            warped_matched_keypoints = warped_matched_keypoints[:, :2]
        else:
            matched_keypoints = matched_keypoints[:, [1, 0]]
            warped_matched_keypoints = warped_matched_keypoints[:, [1, 0]]

        true_warped_keypoints = warp_points(
            matched_keypoints.T,
            homography,
        ).T

        mask_good = ((true_warped_keypoints - warped_matched_keypoints) ** 2).sum(
            dim=1
        ).sqrt() <= threshold

        return mask_good

    @torch.no_grad()
    def update_one(
        self,
        matched_keypoints: torch.Tensor,
        warped_matched_keypoints: torch.Tensor,
        true_homography: torch.Tensor,
    ) -> Dict:
        out_dict = {}
        assert len(matched_keypoints) == len(warped_matched_keypoints)

        true_homography = true_homography.to(self.to_device).float()
        matched_keypoints = matched_keypoints.to(self.to_device)
        warped_matched_keypoints = warped_matched_keypoints.to(self.to_device)

        if matched_keypoints.numel() == 0 or warped_matched_keypoints.numel() == 0:
            acc = 0.0
        else:
            mask_good = self.good_matches_mask(
                matched_keypoints,
                warped_matched_keypoints,
                true_homography,
                self._threshold,
                self._ordering,
            )

            if mask_good.numel() > 0:
                acc = mask_good.float().mean().item()
            else:
                acc = 0.0

        out_dict[self.metric_name] = acc
        return out_dict

    def update_batch(
        self,
        matched_keypoints: List[torch.Tensor],
        warped_matched_keypoints: List[torch.Tensor],
        true_homographies: List[torch.Tensor],
    ):
        out_dict = {}
        assert (
            len(matched_keypoints)
            == len(warped_matched_keypoints)
            == len(true_homographies)
        )

        MMA_list = []
        for i in range(len(matched_keypoints)):
            if matched_keypoints[i].numel() == 0:
                continue
            one_out_dict = self.update_one(
                matched_keypoints[i],
                warped_matched_keypoints[i],
                true_homographies[i],
            )
            if self.metric_name in one_out_dict:
                MMA_list.append(one_out_dict[self.metric_name])

        out_dict[self.metric_name] = torch.tensor(MMA_list).mean().item()

        return out_dict


class HomographyEstimation:
    def __init__(self, name, correctness_thresh, ordering="yx") -> None:
        self.metric_name = name
        self.to_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        assert type(correctness_thresh) in (list, tuple)
        self.correctness_thresh = correctness_thresh
        self.ordering = ordering
        self.error_list = []

        assert ordering in {"xy", "yx"}

    def estimate_homography(
        self, matched_keypoints1, matched_keypoints2, ordering="yx"
    ):
        assert len(matched_keypoints1) == len(matched_keypoints2)
        assert matched_keypoints1.shape[1] in (2, 3)
        if matched_keypoints1.shape[1] == 3:
            matched_keypoints1 = matched_keypoints1[:, :2]
            matched_keypoints2 = matched_keypoints2[:, :2]

        if len(matched_keypoints1) < 4:
            print("Not enough points to estimate homography")
            return None, None

        if ordering == "yx":
            matched_keypoints1 = matched_keypoints1[:, [1, 0]]
            matched_keypoints2 = matched_keypoints2[:, [1, 0]]

        matched_keypoints1 = matched_keypoints1.detach().cpu().numpy()
        matched_keypoints2 = matched_keypoints2.detach().cpu().numpy()

        # estimate the homography
        homography, mask = cv2.findHomography(
            matched_keypoints1,
            matched_keypoints2,
            method=cv2.RANSAC,
        )
        if homography is None:
            print("\nHomography is None while trying to recover pose.\n")
            return None, None

        return torch.tensor(homography, device=self.to_device), mask

    def compute_all_auc(self):
        return compute_auc(self.error_list, self.correctness_thresh)

    @torch.no_grad()
    def update_one(
        self,
        img_shape: Tuple[int, int],
        matched_keypoints1: torch.Tensor,
        matched_keypoints2: torch.Tensor,
        true_homography: torch.Tensor,
    ):
        out_dict = {}

        pred_homography, inliers = self.estimate_homography(
            matched_keypoints1, matched_keypoints2, ordering=self.ordering
        )

        if pred_homography is None:
            correctness = 0.0
            errors = np.inf

            for i in range(len(self.correctness_thresh)):
                out_dict[f"{self.metric_name}@{self.correctness_thresh[i]}_ratio"] = (
                    correctness
                )
            out_dict[self.metric_name + "_errors"] = errors
            out_dict[self.metric_name + "_inliers"] = 0.0
            self.error_list.append(errors)

        else:

            true_homography = true_homography.to(self.to_device).float()
            pred_homography = pred_homography.to(self.to_device).float()

            corners = torch.tensor(
                [
                    [0, 0, 1],
                    [img_shape[1] - 1, 0, 1],
                    [0, img_shape[0] - 1, 1],
                    [img_shape[1] - 1, img_shape[0] - 1, 1],
                ],
                dtype=torch.float32,
                device=self.to_device,
            )

            # apply the true homography and the estimated homography to the corners
            real_warped_corners = torch.mm(
                corners, torch.transpose(true_homography, 0, 1)
            )
            real_warped_corners = (
                real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            )

            warped_corners = torch.mm(corners, torch.transpose(pred_homography, 0, 1))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]

            mean_dist = torch.mean(
                torch.linalg.norm(real_warped_corners - warped_corners, dim=1)
            )

            # considered correct if mean distance is below a given threshold
            correctness = mean_dist <= torch.tensor(
                self.correctness_thresh, device=self.to_device, dtype=torch.float32
            )

            for i in range(len(self.correctness_thresh)):
                out_dict[f"{self.metric_name}@{self.correctness_thresh[i]}_ratio"] = (
                    correctness[i].float().cpu().numpy()
                )
            out_dict[self.metric_name + "_errors"] = mean_dist.float().cpu().numpy()
            out_dict[self.metric_name + "_inliers"] = inliers.mean().item()
            self.error_list.append(mean_dist.float().item())

        return out_dict

    @torch.no_grad()
    def update_batch(
        self,
        img_shapes: List[Tuple[int, int]],
        matched_keypoints1: List[torch.Tensor],
        matched_keypoints2: List[torch.Tensor],
        true_homographies: List[torch.Tensor],
    ):
        out_dict = {}

        self.error_list = []

        assert (
            len(matched_keypoints1) == len(matched_keypoints2) == len(true_homographies)
        )

        for i in range(len(matched_keypoints1)):
            one_out_dict = self.update_one(
                img_shapes[i],
                matched_keypoints1[i],
                matched_keypoints2[i],
                true_homographies[i],
            )
            for k, v in one_out_dict.items():
                if k in out_dict.keys():
                    out_dict[k].append(v)
                else:
                    out_dict[k] = [v]

        # calculate auc
        auc = self.compute_all_auc()

        for k in out_dict.keys():
            out_dict[k] = np.array(out_dict[k]).mean()
        for k in self.correctness_thresh:
            out_dict[f"{self.metric_name}@{k}_auc"] = auc[f"{k}"]

        return out_dict


class RelativePoseEstimation:
    def __init__(
        self, name, pose_thresh, ransac_thresh=1.0, ransac_conf=0.999, ordering="yx"
    ) -> None:
        self.metric_name = name
        self.to_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.pose_thresh = pose_thresh
        self.ransac_thresh = ransac_thresh
        self.ransac_conf = ransac_conf
        self.ordering = ordering
        self.error_list = []

        assert ordering in {"xy", "yx"}

    def estimate_pose(
        self,
        matched_keypoints1: torch.Tensor,
        matched_keypoints2: torch.Tensor,
        K0,
        K1,
        thresh,
        conf,
        ordering="yx",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the relative pose between two cameras from a set of matched keypoints.

        Args:
            matched_keypoints1: torch.Tensor, shape (N, 2)
                The matched keypoints in the first image.
            matched_keypoints2: torch.Tensor, shape (N, 2)
                The matched keypoints in the second image.
            K0: torch.Tensor, shape (3, 3)
                The camera matrix of the first camera.
            K1: torch.Tensor, shape (3, 3)
                The camera matrix of the second camera.
            thresh: float
                The RANSAC threshold to use.
            conf: float
                The RANSAC confidence to use.
            ordering: str
                The ordering of the keypoints. Either "xy" or "yx".

        Returns:
            (R, t, mask): tuple
                The rotation matrix R and translation vector t that transform points from the first camera to the second camera.
                The mask is a boolean array that indicates which points were inliers.
        """
        assert len(matched_keypoints1) == len(matched_keypoints2)
        assert matched_keypoints1.shape[1] in (2, 3)
        if matched_keypoints1.shape[1] == 3:
            matched_keypoints1 = matched_keypoints1[:, :2]
            matched_keypoints2 = matched_keypoints2[:, :2]

        if len(matched_keypoints1) < 5:
            print("Not enough points to estimate pose")
            return None

        if ordering == "yx":
            matched_keypoints1 = matched_keypoints1[:, [1, 0]]
            matched_keypoints2 = matched_keypoints2[:, [1, 0]]

        matched_keypoints1 = matched_keypoints1.detach().cpu().numpy()
        matched_keypoints2 = matched_keypoints2.detach().cpu().numpy()
        K0 = K0.detach().cpu().numpy()
        K1 = K1.detach().cpu().numpy()

        # normalize keypoints
        matched_keypoints1 = (matched_keypoints1 - K0[[0, 1], [2, 2]][None]) / K0[
            [0, 1], [0, 1]
        ][None]
        matched_keypoints2 = (matched_keypoints2 - K1[[0, 1], [2, 2]][None]) / K1[
            [0, 1], [0, 1]
        ][None]

        # normalize ransac threshold
        ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

        # compute pose with cv2
        E, mask = cv2.findEssentialMat(
            matched_keypoints1,
            matched_keypoints2,
            np.eye(3),
            threshold=ransac_thr,
            prob=conf,
            method=cv2.RANSAC,
        )
        if E is None:
            print("\nE is None while trying to recover pose.\n")
            return None

        # recover pose from E
        best_num_inliers = 0
        ret = None
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(
                _E, matched_keypoints1, matched_keypoints2, np.eye(3), 1e9, mask=mask
            )
            if n > best_num_inliers:
                ret = (R, t[:, 0], mask.ravel() > 0)
                best_num_inliers = n

        return ret

    def relative_pose_error(self, T_0to1, R, t, ignore_gt_t_thr=0.0):
        T_0to1 = T_0to1.detach().cpu().numpy()
        # angle error between 2 vectors
        t_gt = T_0to1[:3, 3]
        n = np.linalg.norm(t) * np.linalg.norm(t_gt)
        t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
        t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
        # if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        #     t_err = 0
        if not np.isfinite(np.linalg.norm(t_gt)):  # pure rotation is challenging
            t_err = 0.0

        # t_err = np.linalg.norm(t - t_gt)

        # angle error between 2 rotation matrices
        R_gt = T_0to1[:3, :3]
        cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
        cos = np.clip(cos, -1.0, 1.0)  # handle numercial errors
        R_err = np.rad2deg(np.abs(np.arccos(cos)))

        return t_err, R_err

    def compute_all_auc(self):
        return compute_auc(self.error_list, self.pose_thresh)

    @torch.no_grad()
    def update_one(self, matched_keypoints1, matched_keypoints2, K0, K1, T_0to1):
        out_dict = {}

        ret = self.estimate_pose(
            matched_keypoints1,
            matched_keypoints2,
            K0,
            K1,
            thresh=self.ransac_thresh,
            conf=self.ransac_conf,
            ordering=self.ordering,
        )

        if ret is None:
            out_dict[self.metric_name + "_R_errs"] = np.inf
            out_dict[self.metric_name + "_t_errs"] = np.inf
            out_dict[self.metric_name + "_pose_errs"] = np.inf
            out_dict[self.metric_name + "_inliers"] = 0.0
            for i in range(len(self.pose_thresh)):
                out_dict[f"{self.metric_name}@{self.pose_thresh[i]}_ratio"] = 0.0
            self.error_list.append(np.inf)
        else:
            R, t, inliers = ret
            t_err, R_err = self.relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)
            if np.isfinite(t_err):
                pose_err = np.max([R_err, t_err])
            else:
                pose_err = R_err
            out_dict[self.metric_name + "_R_errs"] = R_err
            out_dict[self.metric_name + "_t_errs"] = t_err
            out_dict[self.metric_name + "_pose_errs"] = pose_err
            out_dict[self.metric_name + "_inliers"] = inliers.mean().item()
            # out_dict['inliers_list'] = inliers
            for i in range(len(self.pose_thresh)):
                correctness = (pose_err <= self.pose_thresh[i]).astype(np.float32)
                out_dict[f"{self.metric_name}@{self.pose_thresh[i]}_ratio"] = (
                    correctness
                )
            self.error_list.append(pose_err)

        return out_dict

    @torch.no_grad()
    def update_batch(self, matched_keypoints1, matched_keypoints2, K0, K1, T_0to1):
        out_dict = {}

        self.error_list = []

        assert (
            len(matched_keypoints1)
            == len(matched_keypoints2)
            == len(K0)
            == len(K1)
            == len(T_0to1)
        )

        for i in range(len(matched_keypoints1)):
            one_out_dict = self.update_one(
                matched_keypoints1[i],
                matched_keypoints2[i],
                K0[i],
                K1[i],
                T_0to1[i],
            )
            for k, v in one_out_dict.items():
                if k in out_dict.keys():
                    out_dict[k].append(v)
                else:
                    out_dict[k] = [v]

        # calculate auc
        auc = self.compute_all_auc()

        for k in out_dict.keys():
            v = np.array(out_dict[k])
            v = v[np.isfinite(v)]
            out_dict[k] = np.mean(v)

        for k in self.pose_thresh:
            out_dict[f"{self.metric_name}@{k}_auc"] = auc[f"{k}"]

        return out_dict
