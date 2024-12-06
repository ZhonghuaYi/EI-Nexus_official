import os
from os import PathLike
from typing import Dict, Any, Tuple

import cv2 as cv
import h5py
import numpy as np
import torch

from MVSEC import MVSEC
import matplotlib.pyplot as plt

import sys

sys.path.append("./")

from core.geometry.gt_generation import gt_matches_from_pose_depth
from core.geometry.wrappers import Camera, Pose


def estimate_pose(
    matched_keypoints1, matched_keypoints2, K0, K1, thresh, conf, ordering="yx"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    E, mask = cv.findEssentialMat(
        matched_keypoints1,
        matched_keypoints2,
        np.eye(3),
        threshold=ransac_thr,
        prob=conf,
        method=cv.RANSAC,
    )
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv.recoverPose(
            _E, matched_keypoints1, matched_keypoints2, np.eye(3), 1e9, mask=mask
        )
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n

    return ret


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def generate_pair_from_sequence(
    depth_and_image_data: Dict[str, Any], time_window: int, method: str, num_pairs: int
):

    depth_image_length = len(depth_and_image_data["depth"])
    print(depth_image_length)
    random_indices1 = np.random.randint(0, depth_image_length - 1, num_pairs)
    random_indices2 = []
    for i in random_indices1:
        if method == "uniform":
            index2 = np.random.randint(i, min(depth_image_length - 1, i + time_window))
        # elif method == 'gaussian':
        #     index2 = int(np.random.normal(i, time_window//3))
        #     index2 = max(0, min(depth_image_length-1, index2))

        random_indices2.append(index2)

    random_indices2 = np.array(random_indices2)

    indices = np.stack([random_indices1, random_indices2], axis=1)

    windows = np.abs(random_indices1 - random_indices2)

    # plt.subplot(1, 2, 1)
    # plt.hist(random_indices1, bins=depth_image_length, range=(0, depth_image_length))
    # plt.subplot(1, 2, 2)
    # plt.hist(windows, bins=time_window, range=(0, time_window))
    # plt.show()

    return indices


def load_val_sequence_data(mvsec_data: MVSEC, sequence_name: str):
    pose_timestamp = mvsec_data.get_pose_timestamp(sequence_name)
    ts_lower_bound, ts_higer_bound = np.min(pose_timestamp), np.max(pose_timestamp)
    depth_and_image_data = mvsec_data.get_paired_depth_and_image(
        sequence_name, "rect", "nearest"
    )

    # adapt the sequence to the pose timestamp
    idx0 = np.searchsorted(
        depth_and_image_data["depth_ts"], ts_lower_bound, side="right"
    )
    idx1 = np.searchsorted(
        depth_and_image_data["depth_ts"], ts_higer_bound, side="left"
    )
    depth_and_image_data["depth"] = depth_and_image_data["depth"][idx0:idx1]
    depth_and_image_data["image"] = depth_and_image_data["image"][idx0:idx1]
    depth_and_image_data["depth_ts"] = depth_and_image_data["depth_ts"][idx0:idx1]
    depth_and_image_data["image_ts"] = depth_and_image_data["image_ts"][idx0:idx1]

    # crop the sequence
    if sequence_name == "indoor_flying1":
        for key in depth_and_image_data.keys():
            depth_and_image_data[key] = depth_and_image_data[key][80:-80]
        sequence_K = mvsec_data.get_K("indoor_flying", "cam0")
        pose_interpolator = mvsec_data.get_pose_interpolator(sequence_name)
    elif sequence_name == "indoor_flying2":
        for key in depth_and_image_data.keys():
            depth_and_image_data[key] = depth_and_image_data[key][200:-100]
        sequence_K = mvsec_data.get_K("indoor_flying", "cam0")
        pose_interpolator = mvsec_data.get_pose_interpolator(sequence_name)
    elif sequence_name == "indoor_flying3":
        for key in depth_and_image_data.keys():
            depth_and_image_data[key] = depth_and_image_data[key][120:-40]
        sequence_K = mvsec_data.get_K("indoor_flying", "cam0")
        pose_interpolator = mvsec_data.get_pose_interpolator(sequence_name)
    elif sequence_name == "indoor_flying4":
        depth_and_image_data["depth"] = depth_and_image_data["depth"][110:-30]
        depth_and_image_data["image"] = depth_and_image_data["image"][110:-30]
        depth_and_image_data["depth_ts"] = depth_and_image_data["depth_ts"][110:-30]
        depth_and_image_data["image_ts"] = depth_and_image_data["image_ts"][110:-30]
        sequence_K = mvsec_data.get_K("indoor_flying", "cam0")
        pose_interpolator = mvsec_data.get_pose_interpolator(sequence_name)
    elif sequence_name == "outdoor_day1":
        depth_and_image_data["depth"] = depth_and_image_data["depth"][:-60]
        depth_and_image_data["image"] = depth_and_image_data["image"][:-60]
        depth_and_image_data["depth_ts"] = depth_and_image_data["depth_ts"][:-60]
        depth_and_image_data["image_ts"] = depth_and_image_data["image_ts"][:-60]
        sequence_K = mvsec_data.get_K("outdoor_day", "cam0")
        pose_interpolator = mvsec_data.get_pose_interpolator(sequence_name)
    elif sequence_name == "outdoor_day2":
        for key in depth_and_image_data.keys():
            depth_and_image_data[key] = depth_and_image_data[key][20:-40]
        sequence_K = mvsec_data.get_K("outdoor_day", "cam0")
        pose_interpolator = mvsec_data.get_pose_interpolator(sequence_name)
    else:
        raise ValueError("Sequence name not recognized")

    sequence_length = len(depth_and_image_data["depth"])
    print(f"Sequence length: {sequence_length}")

    return {
        **depth_and_image_data,
        "K": sequence_K,
        "pose_interpolator": pose_interpolator,
        "length": sequence_length,
    }


def check_indices(indices, dataset: MVSEC, sequence_name: str, image_size: Tuple):
    assert indices.shape[1] == 2

    H, W = image_size
    grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), dim=-1)
    kpts0_coor = grid.reshape(-1, 2).float() + 0.5
    # downsample the kpts num by 64
    # kpts0 = kpts0[::64]
    kpts1_coor = kpts0_coor.clone()

    new_indices = []

    R_err_list = []
    t_err_list = []
    inliers_list = []

    seq_data = load_val_sequence_data(dataset, sequence_name)

    for i in range(indices.shape[0]):
        index0, index1 = indices[i]
        print(index0, index1)

        if index0 == index1:
            continue

        # index0 = 72
        # index1 = 221

        depth0 = seq_data["depth"][index0]
        depth1 = seq_data["depth"][index1]
        image1 = seq_data["image"][index1]
        image0 = seq_data["image"][index0]
        image1_color = cv.cvtColor(image1, cv.COLOR_GRAY2BGR)
        image0_color = cv.cvtColor(image0, cv.COLOR_GRAY2BGR)

        image_color = cv.vconcat([image0_color, image1_color])

        K = seq_data["K"]
        pose_interpolator = seq_data["pose_interpolator"]
        pose0 = pose_interpolator.interpolate(seq_data["depth_ts"][index0])
        pose1 = pose_interpolator.interpolate(seq_data["depth_ts"][index1])

        T_0to1 = (pose1 @ np.linalg.inv(pose0)).astype(np.float32)
        T_1to0 = (pose0 @ np.linalg.inv(pose1)).astype(np.float32)

        kpts0 = kpts0_coor[None]
        kpts1 = kpts1_coor[None]
        camera0 = Camera.from_calibration_matrix(K.astype(np.float32))
        camera1 = Camera.from_calibration_matrix(K.astype(np.float32))
        depth0 = torch.from_numpy(depth0)
        depth1 = torch.from_numpy(depth1)
        if depth0.dim() == 2:
            depth0 = depth0[None]
        if depth1.dim() == 2:
            depth1 = depth1[None]

        gt = gt_matches_from_pose_depth(
            kp0=kpts0,
            kp1=kpts1,
            camera0=camera0,
            camera1=camera1,
            depth0=depth0,
            depth1=depth1,
            T_0to1=Pose.from_4x4mat(T_0to1),
            T_1to0=Pose.from_4x4mat(T_1to0),
        )

        matches0 = gt["matches0"].squeeze(0)
        matches1 = gt["matches1"].squeeze(0)

        matches0_num = (matches0 > -1).sum()
        matches1_num = (matches1 > -1).sum()
        assert matches0_num == matches1_num

        visible0 = gt["visible0"].float().sum()
        visible1 = gt["visible1"].float().sum()
        visible = min(visible0, visible1)

        valid_ratio = matches0_num / min(visible0, visible1)
        print(matches0_num, visible0, visible1, visible, valid_ratio)

        if 0.4 < valid_ratio < 0.8:

            kpts0 = kpts0.squeeze(0)
            kpts1 = kpts1.squeeze(0)
            matched_kpts0 = kpts0[matches0 > -1]
            matched_kpts1 = kpts1[matches0[matches0 > -1]]

            ret = estimate_pose(
                matched_kpts0.numpy(), matched_kpts1.numpy(), K, K, 1.0, 0.999
            )

            if ret is None:
                R_errs = np.inf
                t_errs = np.inf
                inliers = np.inf
            else:
                R, t, inliers = ret
                t_errs, R_errs = relative_pose_error(T_0to1, R, t)
                inliers = inliers.mean()

            new_indices.append([index0, index1])
            R_err_list.append(R_errs)
            t_err_list.append(t_errs)
            inliers_list.append(inliers)
            print(f"R_err: {R_errs}, t_err: {t_errs}, inliers: {inliers}")

            if R_errs < 1 and t_errs < 1 and inliers > 0.9:
                new_indices_array = np.array(new_indices)
                R_err_array = np.array(R_err_list)
                t_err_array = np.array(t_err_list)
                inliers_array = np.array(inliers_list)
                np.savetxt(
                    f"{sequence_name}_new_indices.txt", new_indices_array, fmt="%d"
                )
                np.savetxt(f"{sequence_name}_R_err.txt", R_err_array, fmt="%f")
                np.savetxt(f"{sequence_name}_t_err.txt", t_err_array, fmt="%f")
                np.savetxt(f"{sequence_name}_inliers.txt", inliers_array, fmt="%f")

            # for j in range(matched_kpts0.shape[0]):
            #     pt0 = matched_kpts0[j].numpy()
            #     pt1 = matched_kpts1[j].numpy()
            #     cv.circle(image_color, (int(pt0[1]), int(pt0[0])), 3, (0, 255, 0), -1)
            #     cv.circle(image_color, (int(pt1[1]), int(pt1[0] + H)), 3, (0, 255, 0), -1)
            #     cv.line(image_color, (int(pt0[1]), int(pt0[0])), (int(pt1[1]), int(pt1[0] + H)), (0, 255, 0), 1)

            # cv.imwrite(f'{sequence_name}_matches_{index0}_{index1}1.png', image_color)
            # depth0 = depth0.squeeze().numpy()
            # depth0[np.isnan(depth0)] = 0.
            # depth_color = (depth0.squeeze() / (depth0.max() - depth0.min()) * 255).astype(np.uint8)
            # cv.imwrite(f'{sequence_name}_depth_{index0}.png', depth_color)

            # cv.imwrite(f'{sequence_name}_image_{index0}.png', image0)

        # break

    R_err_list = np.array(R_err_list)
    t_err_list = np.array(t_err_list)
    inliers_list = np.array(inliers_list)
    # remove nan values
    R_err_list = R_err_list[np.isfinite(R_err_list)]
    t_err_list = t_err_list[np.isfinite(t_err_list)]
    inliers_list = inliers_list[np.isfinite(inliers_list)]
    print(
        f"\n\nR_err: {R_err_list.mean()}, t_err: {t_err_list.mean()}, inliers: {inliers_list.mean()}"
    )

    new_indices = np.array(new_indices)
    print(new_indices.shape)
    np.savetxt(f"{sequence_name}_new_indices.txt", new_indices, fmt="%d")
    np.savetxt(f"{sequence_name}_R_err.txt", R_err_list, fmt="%f")
    np.savetxt(f"{sequence_name}_t_err.txt", t_err_list, fmt="%f")
    np.savetxt(f"{sequence_name}_inliers.txt", inliers_list, fmt="%f")


def sample_final_indices(indices, num_samples):
    indices_len = indices.shape[0]
    if indices_len < num_samples:
        return indices
    else:
        return indices[np.random.choice(indices_len, num_samples, replace=False)]


if __name__ == "__main__":
    # # Load the MVSEC dataset
    # dataset = MVSEC('data/MVSEC')

    sequence_name = "indoor_flying3"

    # get sequence data
    # depth_and_image_data = load_val_sequence_data(dataset, sequence_name)

    # # generate random indices pair for relative pose estimation
    # indices = generate_pair_from_sequence(depth_and_image_data, 60, 'uniform', 500)

    # np.savetxt(f'{sequence_name}_indices.txt', indices, fmt='%d')

    # # load the indices
    # indices = np.loadtxt(f'{sequence_name}_indices.txt', dtype=int)

    # # check the indices
    # check_indices(indices, dataset, sequence_name, (260, 346))

    # sample the final indices
    indices = np.loadtxt(f"{sequence_name}_new_indices.txt", dtype=int)
    final_indices = sample_final_indices(indices, 200)
    np.savetxt(f"{sequence_name}_final_indices.txt", final_indices, fmt="%d")
