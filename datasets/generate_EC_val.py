import os
from os import PathLike
from typing import Dict, Any, Tuple

import cv2 as cv
import h5py
import numpy as np
import torch

from EC import EC, ECDataset


def load_val_sequence(ec, sequence_name):
    events = ec.get_events(sequence_name)  # dict
    frame_paths = ec.get_frame_paths(sequence_name)  # list
    frame_timestamps = ec.get_frame_timestamps(sequence_name)  # ndarray
    camera_matrix, _ = ec.get_calibration(sequence_name)
    pose_timestamps = ec.get_pose_timestamps(sequence_name)
    pose_interpolator = ec.get_pose_interpolator(sequence_name)

    # filter valid data with valid timestamps
    ts_lower_bound = max(events["t"][0], frame_timestamps[0], pose_timestamps[0])
    ts_upper_bound = min(events["t"][-1], frame_timestamps[-1], pose_timestamps[-1])
    valid_frame_indices = np.where(
        (frame_timestamps >= ts_lower_bound) & (frame_timestamps <= ts_upper_bound)
    )[0]
    valid_frame_timestamps = frame_timestamps[valid_frame_indices]
    valid_frame_paths = [frame_paths[i] for i in valid_frame_indices]

    # crop the sequence
    valid_frame_timestamps = valid_frame_timestamps[100:-100]
    valid_frame_paths = valid_frame_paths[100:-100]

    return valid_frame_paths


def generate_sequence_val(ec, sequence_name: str, index1_range, pair_num):
    frame_paths = load_val_sequence(ec, sequence_name)

    # random sample index0 in [pair_num]
    index0 = np.random.randint(0, len(frame_paths), size=(pair_num, 1))
    index1 = np.random.randint(index1_range[0], index1_range[1], size=(pair_num, 1))
    index1 = index1 + index0

    valid_index = np.logical_and(index1 > 0, index1 < len(frame_paths))

    index0 = index0[valid_index].reshape((-1, 1))
    index1 = index1[valid_index].reshape((-1, 1))

    print(f"index0: {index0.shape}, index1: {index1.shape}")

    index = np.concatenate((index0, index1), axis=1)

    return index


if __name__ == "__main__":
    root_dir = "../data/EC"
    ec = EC(root_dir)
    sequence_list = ECDataset.VAL_LIST
    for sequence_name in sequence_list:
        print(f"Generate val for {sequence_name}")
        index = generate_sequence_val(ec, sequence_name, [10, 60], 100)
        random_indices = np.random.randint(0, len(index), size=(50,))
        index = index[random_indices]
        save_path = f"{root_dir}/{sequence_name}_val.txt"
        print(f"Save to {save_path}")
        np.savetxt(save_path, index, fmt="%d")
