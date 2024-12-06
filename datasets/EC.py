from typing import List, Tuple, Dict

import torch
import numpy as np
import os

from glob import glob

import cv2 as cv
from torch.utils.data import DataLoader, Dataset, distributed
from scipy.spatial.transform import Rotation

from .Interpolator import PoseInterpolator
from .augment import EventPointsAugmentation, PairAugmentation, ImageArrayAugmentation
from .representations import (
    events_to_voxel_grid,
    events_to_voxel_grid_new,
    events_to_time_surface,
    events_to_event_stack,
    events_to_distance_map,
)
from .visualize import draw_events_accumulation_image


class EC(Dataset):
    RESOLUTION = (240, 180)
    SQUENCE_LIST = [
        "boxes_6dof",
        "boxes_rotation",
        "boxes_translation",
        "hdr_boxes",
        "poster_6dof",
        "poster_rotation",
        "poster_translation",
        "shapes_6dof",
        "shapes_rotation",
        "shapes_translation",
        "calibration",
    ]

    def __init__(self, data_path: str):
        super().__init__()

        self.data_path = os.path.join(data_path)
        self.supported_squence_list = EC.SQUENCE_LIST

    def _check_sequence_name(self, sequence_name: str):
        assert (
            sequence_name in self.supported_squence_list
        ), f"Sequence {sequence_name} not supported. Supported sequences: {self.supported_squence_list}"

    def get_frame_paths(self, sequence_name: str):
        self._check_sequence_name(sequence_name)
        sequence_dir = os.path.join(self.data_path, sequence_name)
        frame_paths = sorted(
            glob(os.path.join(sequence_dir, "images_corrected", "*.png"))
        )
        return frame_paths

    def get_frame_timestamps(self, sequence_name: str):
        self._check_sequence_name(sequence_name)
        sequence_dir = os.path.join(self.data_path, sequence_name)
        frame_timestamps = np.genfromtxt(
            os.path.join(sequence_dir, "images.txt"), usecols=[0]
        )
        return frame_timestamps

    def get_events(self, sequence_name: str):
        self._check_sequence_name(sequence_name)
        sequence_dir = os.path.join(self.data_path, sequence_name)
        # events = np.genfromtxt(os.path.join(sequence_dir, "events_corrected.txt"))
        events = np.load(os.path.join(sequence_dir, "events_corrected.npy"))
        t = events[:, 0]
        x = events[:, 1]
        y = events[:, 2]
        p = events[:, 3]
        return {"t": t, "x": x, "y": y, "p": p}

    def get_calibration(self, sequence_name: str):
        self._check_sequence_name(sequence_name)
        sequence_dir = os.path.join(self.data_path, sequence_name)
        calib_file = os.path.join(sequence_dir, "calib.txt")
        calib_data = np.genfromtxt(calib_file)
        camera_matrix = calib_data[:4]
        distortion_coeffs = calib_data[4:]
        camera_matrix = np.array(
            [
                [camera_matrix[0], 0, camera_matrix[2]],
                [0, camera_matrix[1], camera_matrix[3]],
                [0, 0, 1],
            ]
        )
        return camera_matrix, distortion_coeffs

    def get_pose(self, sequence_name: str):
        self._check_sequence_name(sequence_name)
        sequence_dir = os.path.join(self.data_path, sequence_name)
        pose_file = os.path.join(sequence_dir, "groundtruth.npy")
        pose_data = np.load(pose_file)
        pose_timestamps = pose_data[:, 0]
        p_xyz = pose_data[:, 1:4]
        q_xyzw = pose_data[:, 4:]
        b_R = Rotation.from_quat(q_xyzw).as_matrix()
        b_t = p_xyz
        return pose_timestamps, b_t, b_R

    def get_pose_timestamps(self, sequence_name: str):
        self._check_sequence_name(sequence_name)
        sequence_dir = os.path.join(self.data_path, sequence_name)
        pose_file = os.path.join(sequence_dir, "groundtruth.txt")
        pose_data = np.genfromtxt(pose_file, delimiter=" ")
        pose_timestamps = pose_data[:, 0]
        return pose_timestamps

    def get_pose_interpolator(self, sequence_name: str) -> PoseInterpolator:
        self._check_sequence_name(sequence_name)
        pose = self.get_pose(sequence_name)
        pose_timestamps, b_t, b_R = pose
        pose_interpolator = PoseInterpolator(
            pose_timestamps, b_t, b_R, quat_R=False, mode="linear"
        )
        return pose_interpolator


class ECDataset(Dataset):
    RESOLUTION = (240, 180)
    TRAIN_LIST = [
        "boxes_6dof",
        "hdr_boxes",
        "poster_6dof",
        "poster_rotation",
        "poster_translation",
        "calibration",
    ]
    VAL_LIST = [
        "boxes_rotation",
        "boxes_translation",
        "shapes_6dof",
        "shapes_rotation",
        "shapes_translation",
    ]

    def __init__(self, cfg, is_train: bool, use_aug: bool = True) -> None:
        super().__init__()
        self.ec = EC(cfg.data_path)
        self.representation_type = cfg.representation_type
        self.event_dt = cfg.event_dt
        self.select_matching_pair = cfg.select_matching_pair
        self.channel = cfg.channel
        self.is_train = is_train
        self.use_aug = use_aug

        # load data
        events_list = []
        frame_path_list = []
        frame_timestamp_list = []
        pose_interpolator_list = []
        sequence_length_list = []
        sequence_K_list = []
        frame_indices_list = []

        if is_train:
            self.support_list = ECDataset.TRAIN_LIST
            self.augment_event_points = EventPointsAugmentation(
                time_scale=cfg.train.event_point_aug.time_scale,
                slice_dt=cfg.train.event_point_aug.slice_dt,
                slice_mode=cfg.train.event_point_aug.slice_mode,
                flip_p=cfg.train.event_point_aug.flip_p,
                xy_std=cfg.train.event_point_aug.xy_std,
                ts_std=cfg.train.event_point_aug.ts_std,
                add_percent=cfg.train.event_point_aug.add_percent,
                del_percent=cfg.train.event_point_aug.del_percent,
            )
            self.augment_image = ImageArrayAugmentation(
                gamma_p=cfg.train.image_aug.gamma_p,
                gamma_limit=cfg.train.image_aug.gamma_limit,
                saturation_p=cfg.train.image_aug.saturation_p,
                val_shift_limit=cfg.train.image_aug.val_shift_limit,
                brightness_p=cfg.train.image_aug.brightness_p,
                brightness_limit=cfg.train.image_aug.brightness_limit,
                contrast_limit=cfg.train.image_aug.contrast_limit,
                noise_p=cfg.train.image_aug.noise_p,
            )
            self.augment_pair = PairAugmentation(
                crop_size=cfg.train.pair_aug.crop_size,
                flip_p_h=cfg.train.pair_aug.flip_p_h,
                flip_p_w=cfg.train.pair_aug.flip_p_w,
                rotate_angle=cfg.train.pair_aug.rotate_angle,
            )
        else:
            self.support_list = ECDataset.VAL_LIST

        for sequence_name in self.support_list:
            events = self.ec.get_events(sequence_name)  # dict
            frame_paths = self.ec.get_frame_paths(sequence_name)  # list
            frame_timestamps = self.ec.get_frame_timestamps(sequence_name)  # ndarray
            camera_matrix, _ = self.ec.get_calibration(sequence_name)
            pose_timestamps = self.ec.get_pose_timestamps(sequence_name)
            pose_interpolator = self.ec.get_pose_interpolator(sequence_name)

            # filter valid data with valid timestamps
            ts_lower_bound = max(
                events["t"][0], frame_timestamps[0], pose_timestamps[0]
            )
            ts_upper_bound = min(
                events["t"][-1], frame_timestamps[-1], pose_timestamps[-1]
            )
            valid_frame_indices = np.where(
                (frame_timestamps >= ts_lower_bound)
                & (frame_timestamps <= ts_upper_bound)
            )[0]
            valid_frame_timestamps = frame_timestamps[valid_frame_indices]
            valid_frame_paths = [frame_paths[i] for i in valid_frame_indices]

            # crop the sequence
            valid_frame_timestamps = valid_frame_timestamps[100:-100]
            valid_frame_paths = valid_frame_paths[100:-100]
            valid_frame_indices = valid_frame_indices[100:-100]

            events_list.append(events)
            frame_path_list.append(valid_frame_paths)
            frame_timestamp_list.append(valid_frame_timestamps)
            frame_indices_list.append(valid_frame_indices)
            pose_interpolator_list.append(pose_interpolator)
            sequence_length_list.append(len(valid_frame_paths))
            sequence_K_list.append(camera_matrix)

        self.events_list = events_list
        self.frame_path_list = frame_path_list
        self.frame_timestamp_list = frame_timestamp_list
        self.frame_indices_list = frame_indices_list
        self.pose_interpolator_list = pose_interpolator_list
        self.sequence_length_list = np.array(sequence_length_list)
        self.sequence_K_list = sequence_K_list

        # load representation tool
        if cfg.representation_type == "TimeSurface":
            self.representation = events_to_time_surface
        elif cfg.representation_type == "EventStack":
            self.representation = events_to_event_stack
        elif cfg.representation_type == "EventDistanceMap":
            self.representation = events_to_distance_map
        elif cfg.representation_type == "VoxelGrid":
            self.representation = events_to_voxel_grid
        else:
            raise ValueError(
                f"Unsupported representation type '{cfg.representation_type}'."
            )

    def __len__(self):
        return self.sequence_length_list.sum()

    def get_events_at_timestamp(self, events: Dict, timestamp: float, events_dt: float):
        # left side
        index0 = np.searchsorted(events["t"], timestamp - events_dt, side="left")
        index1 = np.searchsorted(events["t"], timestamp, side="right")
        return {
            "t": events["t"][index0:index1],
            "x": events["x"][index0:index1],
            "y": events["y"][index0:index1],
            "p": events["p"][index0:index1],
        }

    def get_image_events_pose(
        self,
        frame_path_list,
        frame_timestamp_list,
        frame_indices_list,
        sequence_events,
        pose_interpolator,
        events_dt,
        view_index,
    ):
        frame_path = frame_path_list[view_index]
        frame = cv.imread(frame_path, 0)
        frame = np.expand_dims(frame, axis=-1)
        frame_timestamp = frame_timestamp_list[view_index]
        frame_index = frame_indices_list[view_index]
        events = self.get_events_at_timestamp(
            sequence_events, frame_timestamp, events_dt
        )
        pose = pose_interpolator.interpolate(frame_timestamp)
        return {
            "image": frame,
            "image_ts": torch.tensor(frame_timestamp, dtype=torch.float32).reshape(1),
            "image_index": torch.tensor(frame_index, dtype=torch.int32),
            "events_raw": events,
            "pose": pose.astype(np.float32),
        }

    def get_relative_pose(self, pose0, pose1):
        return pose1 @ np.linalg.inv(pose0)

    def augment_and_convert(self, data, augment_pair: bool):
        if self.is_train and self.use_aug:
            data["events_raw"] = self.augment_event_points(data["events_raw"])
            data["image"] = self.augment_image(data["image"])

        data["image"] = torch.from_numpy(data["image"].transpose(2, 0, 1)).float()
        events_image = draw_events_accumulation_image(
            data["events_raw"], self.RESOLUTION
        )
        data["events_image"] = torch.from_numpy(events_image).unsqueeze(0).float()
        data["events_rep"] = self.representation(
            data["events_raw"], (self.channel, self.RESOLUTION[1], self.RESOLUTION[0])
        )

        if self.is_train and self.use_aug and augment_pair:
            data["events_rep"], data["image"], data["events_image"] = self.augment_pair(
                data["events_rep"], data["image"], data["events_image"]
            )

        return data

    def __getitem__(self, index):
        # find the sequence index
        sequence_index = np.searchsorted(
            self.sequence_length_list.cumsum(), index, side="right"
        )
        if sequence_index > 0:
            index = index - self.sequence_length_list[:sequence_index].sum()

        # load camera intrinsics
        K = self.sequence_K_list[sequence_index]

        # load sequence data
        sequence_frame_paths = self.frame_path_list[sequence_index]
        sequence_frame_timestamps = self.frame_timestamp_list[sequence_index]
        sequence_frame_indices = self.frame_indices_list[sequence_index]
        sequence_events = self.events_list[sequence_index]
        pose_interpolator = self.pose_interpolator_list[sequence_index]

        # load image, events and pose at the specific index
        data0 = self.get_image_events_pose(
            sequence_frame_paths,
            sequence_frame_timestamps,
            sequence_frame_indices,
            sequence_events,
            pose_interpolator,
            self.event_dt,
            index,
        )
        data0["K"] = torch.from_numpy(K).float().contiguous()

        # load depth, image, events and pose at a random index of the same sequence
        if self.select_matching_pair:
            sequence_length = self.sequence_length_list[sequence_index]
            index1_upper_bound = (
                sequence_length if index + 60 > sequence_length else index + 60
            )
            index1 = np.random.randint(index + 1, index1_upper_bound)
            data1 = self.get_image_events_pose(
                sequence_frame_paths,
                sequence_frame_timestamps,
                sequence_frame_indices,
                sequence_events,
                pose_interpolator,
                self.event_dt,
                index1,
            )

            # augment the matching pair
            data0 = self.augment_and_convert(data0, augment_pair=False)
            data1 = self.augment_and_convert(data1, augment_pair=False)
            data1["K"] = data0["K"]

            if self.is_train:
                del data0["events_raw"]
                del data1["events_raw"]

            T_0to1 = self.get_relative_pose(data0["pose"], data1["pose"])
            T_0to1 = torch.from_numpy(T_0to1.astype("float32")).contiguous()
            T_1to0 = self.get_relative_pose(data1["pose"], data0["pose"])
            T_1to0 = torch.from_numpy(T_1to0.astype("float32")).contiguous()

            return data0, data1, T_0to1, T_1to0

        else:
            data0 = self.augment_and_convert(data0, augment_pair=True)

            if self.is_train:
                del data0["events_raw"]

            return (
                data0,
                data0,
                torch.eye(4, dtype=torch.float32),
                torch.eye(4, dtype=torch.float32),
            )


class ECDataset_VAL(ECDataset):
    def __init__(self, cfg):
        super().__init__(cfg, is_train=False, use_aug=False)

        val_indices_path_list = [
            os.path.join(cfg.data_path, f"new_{sequence_name}_val.txt")
            for sequence_name in ECDataset.VAL_LIST
        ]

        assert len(val_indices_path_list) == len(self.support_list)

        indices_list = []
        indices_length_list = []
        for path in val_indices_path_list:
            indices = np.loadtxt(path)
            indices_list.append(indices)
            indices_length_list.append(indices.shape[0])

        self.indices_list = indices_list
        self.indices_length_list = np.array(indices_length_list)

    def __len__(self) -> int:
        return np.array(self.indices_length_list).sum()

    def __getitem__(self, index):
        # find the sequence index
        sequence_index = np.searchsorted(
            self.indices_length_list.cumsum(), index, side="right"
        )
        if sequence_index > 0:
            index = index - self.indices_length_list[:sequence_index].sum()

        # load two indices for two views
        indices = self.indices_list[sequence_index]
        view0_index, view1_index = indices[index]
        view0_index = int(view0_index)
        view1_index = int(view1_index)

        # load camera K
        K = self.sequence_K_list[sequence_index]

        # load sequence data
        sequence_frame_paths = self.frame_path_list[sequence_index]
        sequence_frame_timestamps = self.frame_timestamp_list[sequence_index]
        sequence_frame_indices = self.frame_indices_list[sequence_index]
        sequence_events = self.events_list[sequence_index]
        pose_interpolator = self.pose_interpolator_list[sequence_index]

        # load data for view0
        data0 = self.get_image_events_pose(
            sequence_frame_paths,
            sequence_frame_timestamps,
            sequence_frame_indices,
            sequence_events,
            pose_interpolator,
            self.event_dt,
            view0_index,
        )
        data0["K"] = torch.from_numpy(K).float().contiguous()

        # load data for view1
        data1 = self.get_image_events_pose(
            sequence_frame_paths,
            sequence_frame_timestamps,
            sequence_frame_indices,
            sequence_events,
            pose_interpolator,
            self.event_dt,
            view1_index,
        )
        data1["K"] = torch.from_numpy(K).float().contiguous()

        # convert data
        data0 = self.augment_and_convert(data0, augment_pair=False)
        data1 = self.augment_and_convert(data1, augment_pair=False)

        # calculate relative pose
        T_0to1 = self.get_relative_pose(data0["pose"], data1["pose"])
        T_0to1 = torch.from_numpy(T_0to1.astype("float32")).contiguous()
        T_1to0 = self.get_relative_pose(data1["pose"], data0["pose"])
        T_1to0 = torch.from_numpy(T_1to0.astype("float32")).contiguous()

        return data0, data1, T_0to1, T_1to0


def fetch_ec_dataloader(cfg, split: str, logger, rank=-1, world_size=1):
    if split == "train":
        is_train = True

        data = ECDataset(cfg, is_train=is_train, use_aug=True)

        logger.log_info(f"Split '{split}' dataset length: {len(data)}")
        train_sampler = (
            None if rank == -1 else distributed.DistributedSampler(data, shuffle=True)
        )
        data_loader = DataLoader(
            dataset=data,
            batch_size=cfg.train.batch_size // world_size,
            shuffle=cfg.train.shuffle and train_sampler is None,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            drop_last=cfg.train.drop_last,
            sampler=train_sampler,
        )
    else:
        is_train = False
        data = ECDataset_VAL(cfg)
        logger.log_info(f"Split '{split}' dataset length: {len(data)}")
        data_loader = DataLoader(
            dataset=data,
            batch_size=cfg.val.batch_size,
            shuffle=cfg.val.shuffle,
            num_workers=cfg.val.num_workers,
            pin_memory=cfg.val.pin_memory,
            drop_last=cfg.val.drop_last,
        )

    return data_loader


if __name__ == "__main__":
    ec = EC("../data/EC")
    # frames = ec.get_frame_paths("boxes_6dof")
    for s in EC.SQUENCE_LIST:
        # events = ec.get_events(s)
        pose = ec.get_pose(s)
