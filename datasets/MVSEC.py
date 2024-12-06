import os
from os import PathLike
from typing import Dict, Any, List

import cv2 as cv
import h5py
import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, distributed

from .Interpolator import PoseInterpolator, T_to_Rt
from .augment import EventPointsAugmentation, PairAugmentation, ImageArrayAugmentation
from .representations import (
    events_to_voxel_grid,
    events_to_voxel_grid_new,
    events_to_time_surface,
    events_to_event_stack,
    events_to_distance_map,
)
from .visualize import draw_events_accumulation_image


class MVSEC:
    """
    Original MVSEC dataset.
    """

    RESOLUTION = (346, 260)
    SQUENCE_DICT = {
        "indoor_flying": [
            "indoor_flying1",
            "indoor_flying2",
            "indoor_flying3",
            "indoor_flying4",
        ],
        "outdoor_day": ["outdoor_day1", "outdoor_day2"],
    }

    def __init__(self, data_path: PathLike) -> None:
        super().__init__()

        self.data_path = os.path.join(data_path)
        self.supported_sequence_dict = MVSEC.SQUENCE_DICT
        self.supported_scene_list = list(self.supported_sequence_dict.keys())
        self.supported_sequence_list = []
        for scene in self.supported_scene_list:
            self.supported_sequence_list.extend(self.supported_sequence_dict[scene])

    def load_data(self, sequence_name: str) -> h5py.File:
        """
        load the "*_data.hdf5" of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.

        Returns:
            HDF5 file object.

        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        scene_name = None
        for scene in self.supported_scene_list:
            if sequence_name in self.supported_sequence_dict[scene]:
                scene_name = scene
                break

        data_file = os.path.join(
            self.data_path, scene_name, sequence_name + "_data.hdf5"
        )

        return h5py.File(data_file, "r")

    def load_gt(self, sequence_name: str) -> h5py.File:
        """
        load the "*_gt.hdf5" of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.

        Returns:
            HDF5 file object.

        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        scene_name = None
        for scene in self.supported_scene_list:
            if sequence_name in self.supported_sequence_dict[scene]:
                scene_name = scene
                break

        data_file = os.path.join(self.data_path, scene_name, sequence_name + "_gt.hdf5")

        return h5py.File(data_file, "r")

    def load_rectified_data(self, sequence_name: str) -> h5py.File:
        """
        load the "*_rectified.h5" of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.

        Returns:
            HDF5 file object.

        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        scene_name = None
        for scene in self.supported_scene_list:
            if sequence_name in self.supported_sequence_dict[scene]:
                scene_name = scene
                break

        data_file = os.path.join(
            self.data_path, scene_name, sequence_name + "_rectified.h5"
        )

        if not os.path.exists(data_file):
            raise ValueError(f"Sequence '{sequence_name}' not rectified.")

        return h5py.File(data_file, "r")

    def load_calibration_map(self, scene_name: str) -> dict:
        """
        load the calibration map of the specified scene.

        Args:
            scene_name (str): the name of scene.

        Returns:
            dict: a dict {'left_x_map', 'left_y_map', 'right_x_map', 'right_y_map'} of calibration map.
        """

        calib_dir = os.path.join(self.data_path, scene_name + "_calib")
        left_x_map = np.loadtxt(
            os.path.join(calib_dir, scene_name + "_left_x_map.txt"), dtype=np.float32
        )
        left_y_map = np.loadtxt(
            os.path.join(calib_dir, scene_name + "_left_y_map.txt"), dtype=np.float32
        )

        right_x_map = np.loadtxt(
            os.path.join(calib_dir, scene_name + "_right_x_map.txt"), dtype=np.float32
        )
        right_y_map = np.loadtxt(
            os.path.join(calib_dir, scene_name + "_right_y_map.txt"), dtype=np.float32
        )

        return {
            "left_x_map": left_x_map,
            "left_y_map": left_y_map,
            "right_x_map": right_x_map,
            "right_y_map": right_y_map,
        }

    def load_calibration_yaml(self, scene_name: str) -> dict:
        """
        load the calibration yaml of the specified scene.

        Args:
            scene_name (str): the name of scene.

        Returns:
            dict: a dict of calibration yaml.
        """

        file_path = os.path.join(
            self.data_path, scene_name + "_calib", f"camchain-imucam-{scene_name}.yaml"
        )
        with open(file_path, "r") as f:
            calib_data = yaml.load(f, Loader=yaml.SafeLoader)

        return calib_data

    def get_K(self, scene_name: str, camera: str) -> np.ndarray:
        """
        get the camera intrinsic matrix of the specified scene.

        Args:
            scene_name (str): the name of scene.
            camera (str): the name of camera.

        Returns:
            a numpy array of camera intrinsic matrix [3, 3].
        """

        calib_data = self.load_calibration_yaml(scene_name)
        K = np.eye(3)
        K[[0, 1, 0, 1], [0, 1, 2, 2]] = calib_data[camera]["intrinsics"]
        return K

    def get_events(
        self, sequence_name: str, event_type: str = "rect"
    ) -> Dict[str, np.ndarray]:
        """
        get events of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            event_type (str): the type of events, 'raw' or 'rect'.

        Returns:
            a dict {'x', 'y', 't', 'p'} of events.
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        if event_type == "rect":
            data_file = self.load_rectified_data(sequence_name)
            return {
                "x": np.array(data_file["davis"]["left"]["events_rect"][:, 0]),
                "y": np.array(data_file["davis"]["left"]["events_rect"][:, 1]),
                "t": np.array(data_file["davis"]["left"]["events_rect"][:, 2]),
                "p": np.array(data_file["davis"]["left"]["events_rect"][:, 3]),
            }
        elif event_type == "raw":
            data_file = self.load_data(sequence_name)
            return {
                "x": np.array(data_file["davis"]["left"]["events"][:, 0]),
                "y": np.array(data_file["davis"]["left"]["events"][:, 1]),
                "t": np.array(data_file["davis"]["left"]["events"][:, 2]),
                "p": np.array(data_file["davis"]["left"]["events"][:, 3]),
            }
        else:
            raise ValueError(f"Unsupported event type '{type}'.")

    def get_image(self, sequence_name: str, image_type: str = "rect") -> np.ndarray:
        """
        get image of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            image_type (str): the type of image, 'raw' or 'rect'.

        Returns:
            a numpy array of grayscale image [N, H, W].
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        if image_type == "rect":
            data_file = self.load_rectified_data(sequence_name)
            return np.array(data_file["davis"]["left"]["image_rect"])
        elif image_type == "raw":
            data_file = self.load_data(sequence_name)
            return np.array(data_file["davis"]["left"]["image_raw"])

    def get_image_timestamp(self, sequence_name: str) -> np.ndarray:
        """
        get timestamp of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.

        Returns:
            a numpy array of timestamps [N].
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        data_file = self.load_data(sequence_name)
        return np.array(data_file["davis"]["left"]["image_raw_ts"])

    def get_depth_image(
        self, sequence_name: str, depth_type: str = "rect"
    ) -> np.ndarray:
        """
        get depth image of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            depth_type (str): the type of depth image, 'raw' or 'rect'.

        Returns:
            a numpy array of depth image [N, H, W].
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        gt_file = self.load_gt(sequence_name)
        if depth_type == "raw":
            return np.array(gt_file["davis"]["left"]["depth_image_raw"])
        elif depth_type == "rect":
            return np.array(gt_file["davis"]["left"]["depth_image_rect"])
        else:
            raise ValueError(f"Unsupported depth image type '{type}'.")

    def get_depth_image_timestamp(
        self, sequence_name: str, depth_type: str = "rect"
    ) -> np.ndarray:
        """
        get depth image timestamp of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            depth_type (str): the type of depth image, 'raw' or 'rect'.

        Returns:
            a numpy array of depth image timestamps [N].
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        gt_file = self.load_gt(sequence_name)
        if depth_type == "raw":
            return np.array(gt_file["davis"]["left"]["depth_image_raw_ts"])
        elif depth_type == "rect":
            return np.array(gt_file["davis"]["left"]["depth_image_rect_ts"])
        else:
            raise ValueError(f"Unsupported depth image type '{type}'.")

    def get_paired_depth_and_image(
        self, sequence_name: str, depth_type: str = "rect", pair_type="nearest"
    ) -> dict:
        """
        get paired depth image and image of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            depth_type (str): the type of depth image, 'raw' or 'rect'.

        Returns:
            a dict containing:
                - depth (np.ndarray): a numpy array of depth image [N, H, W].
                - image (np.ndarray): a numpy array of grayscale image [N, H, W].
                - depth_ts (np.ndarray): a numpy array of depth image timestamps [N].
                - image_ts (np.ndarray): a numpy array of timestamps [N].
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        depth_image = self.get_depth_image(sequence_name, depth_type)
        depth_timestamp = self.get_depth_image_timestamp(sequence_name, depth_type)
        image = self.get_image(sequence_name, image_type=depth_type)
        image_timestamp = self.get_image_timestamp(sequence_name)

        image_index = np.arange(len(image_timestamp))
        if pair_type == "nearest":
            nearest_index = np.abs(
                np.subtract.outer(image_timestamp, depth_timestamp)
            ).argmin(axis=0)
            image_index = nearest_index
            indexed_image = image[nearest_index]
            indexed_image_timestamp = image_timestamp[nearest_index]

        return {
            "depth": depth_image,
            "image": indexed_image,
            "image_index": image_index,
            "depth_ts": depth_timestamp,
            "image_ts": indexed_image_timestamp,
        }

    def get_pose(self, sequence_name: str, pose_type: str = "pose") -> np.ndarray:
        """
        get pose of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            pose_type (str): the type of pose, 'pose' or 'odometry'.

        Returns:
            a numpy array of pose [N, 4, 4].
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        gt_file = self.load_gt(sequence_name)
        if pose_type == "pose":
            return np.array(gt_file["davis"]["left"]["pose"])
        elif pose_type == "odometry":
            return np.array(gt_file["davis"]["left"]["odometry"])
        else:
            raise ValueError(f"Unsupported pose type '{type}'.")

    def get_pose_timestamp(
        self, sequence_name: str, pose_type: str = "pose"
    ) -> np.ndarray:
        """
        get pose timestamp of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            pose_type (str): the type of pose, 'pose' or 'odometry'.S

        Returns:
            a numpy array of pose timestamps [N].
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        gt_file = self.load_gt(sequence_name)
        if pose_type == "pose":
            return np.array(gt_file["davis"]["left"]["pose_ts"])
        elif pose_type == "odometry":
            return np.array(gt_file["davis"]["left"]["odometry_ts"])
        else:
            raise ValueError(f"Unsupported pose type '{type}'.")

    def get_pose_interpolator(self, sequence_name: str, pose_type: str = "pose"):
        """
        get pose interpolator of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            pose_type (str): the type of pose, 'pose' or 'odometry'.

        Returns:
            a PoseInterpolator.
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        pose = self.get_pose(sequence_name, pose_type)
        b_R, b_t = T_to_Rt(pose, batch=True)
        pose_timestamp = self.get_pose_timestamp(sequence_name, pose_type)
        return PoseInterpolator(pose_timestamp, b_t, b_R, quat_R=False)

    def get_blended_image(self, sequence_name: str):
        """
        get the visualization of all events from the left DAVIS that are 25ms from each left depth map superimposed on the depth map.

        Args:
            sequence_name (str): the name of sequence.

        Returns:
            a numpy array of blended image [N, H, W].
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        gt_file = self.load_gt(sequence_name)
        return {
            "blended_image": np.array(gt_file["davis"]["left"]["blended_image_rect"])
        }

    def get_events_iterator(
        self, sequence_name: str, dt: float, event_type: str = "rect"
    ):
        """
        get events iterator of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            dt (float): the time interval of each iteration.
            event_type (str): the type of events, 'raw' or 'rect'.

        Returns:
            a iterator.
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        events = self.get_events(sequence_name, event_type)
        frame_ts_arr = self.get_depth_image_timestamp(
            sequence_name, depth_type=event_type
        )
        dt_elapsed = 0

        for t1 in np.arange(frame_ts_arr[0], frame_ts_arr[-1], dt):
            t0 = t1 - dt
            idx0 = np.searchsorted(events["t"], t0, side="left")
            idx1 = np.searchsorted(events["t"], t1, side="right")

            yield dt_elapsed, {
                "x": events["x"][idx0:idx1],
                "y": events["y"][idx0:idx1],
                "p": events["p"][idx0:idx1],
                "t": events["t"][idx0:idx1],
            }

            dt_elapsed += dt

    def get_depth_and_image_iterator(
        self, sequence_name: str, depth_type: str = "rect"
    ):
        """
        get depth and image iterator of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            depth_type (str): the type of depth image, 'raw' or 'rect'.

        Returns:
            a iterator.
        """

        assert sequence_name in self.supported_sequence_list, (
            f"The specified sequence '{sequence_name}' is not " f"supported."
        )

        pair = self.get_paired_depth_and_image(sequence_name, depth_type)
        depth_image = pair["depth"]
        depth_timestamp = pair["depth_ts"]
        image = pair["image"]
        image_index = pair["image_index"]
        image_timestamp = pair["image_ts"]

        for i in range(len(depth_timestamp)):
            yield depth_timestamp[i], {
                "depth": depth_image[i],
                "image": image[i],
                "image_index": image_index[i],
                "depth_ts": depth_timestamp[i],
                "image_ts": image_timestamp[i],
            }


class MVSECDataset(Dataset):
    """
    MVSEC dataset for pytorch training and validation.
    Only use camera0, which is the left DAVIS camera.
    """

    RESOLUTION = (346, 260)
    TRAIN_SEQUENCE_DICT = {
        "indoor_flying": [
            "indoor_flying1",
            "indoor_flying2",
            "indoor_flying3",
        ],
        "outdoor_day": [
            "outdoor_day2",
        ],
    }
    VAL_SEQUENCE_DICT = {
        'indoor_flying': ['indoor_flying4'],
        "outdoor_day": [
            "outdoor_day1",
        ]
    }

    def __init__(self, cfg, is_train: bool, use_aug: bool = True) -> None:
        super().__init__()
        self.mvsec = MVSEC(cfg.data_path)
        self.representation_type = cfg.representation_type
        self.event_dt = cfg.event_dt
        self.select_matching_pair = cfg.select_matching_pair
        self.channel = cfg.channel
        self.is_train = is_train
        self.use_aug = use_aug

        # load data
        events_list = []
        depth_and_image_list = []
        pose_interpolator_list = []
        sequence_length_list = []
        sequence_K_list = []

        if is_train:
            self.supported_sequence_dict = MVSECDataset.TRAIN_SEQUENCE_DICT
            self.supported_scene_list = list(self.supported_sequence_dict.keys())
            self.supported_sequence_list = []
            for scene in self.supported_scene_list:
                self.supported_sequence_list.extend(self.supported_sequence_dict[scene])
                for sequence_name in self.supported_sequence_dict[scene]:
                    pose_timestamp = self.mvsec.get_pose_timestamp(sequence_name)
                    ts_lower_bound, ts_higer_bound = np.min(pose_timestamp), np.max(
                        pose_timestamp
                    )
                    events_list.append(self.mvsec.get_events(sequence_name))
                    depth_and_image_data = self.mvsec.get_paired_depth_and_image(
                        sequence_name, "rect", "nearest"
                    )

                    # adapt the sequence to the pose timestamp
                    idx0 = np.searchsorted(
                        depth_and_image_data["depth_ts"], ts_lower_bound, side="right"
                    )
                    idx1 = np.searchsorted(
                        depth_and_image_data["depth_ts"], ts_higer_bound, side="left"
                    )
                    for key in depth_and_image_data.keys():
                        depth_and_image_data[key] = depth_and_image_data[key][idx0:idx1]

                    # crop the specific sequence
                    if sequence_name == "indoor_flying1":
                        for key in depth_and_image_data.keys():
                            depth_and_image_data[key] = depth_and_image_data[key][80:-80]
                    elif sequence_name == "indoor_flying2":
                        for key in depth_and_image_data.keys():
                            depth_and_image_data[key] = depth_and_image_data[key][200:-100]
                    elif sequence_name == "indoor_flying3":
                        for key in depth_and_image_data.keys():
                            depth_and_image_data[key] = depth_and_image_data[key][120:-40]
                    elif sequence_name == "outdoor_day2":
                        for key in depth_and_image_data.keys():
                            depth_and_image_data[key] = depth_and_image_data[key][20:-40]

                    depth_and_image_list.append(depth_and_image_data)
                    sequence_length_list.append(len(depth_and_image_data["depth_ts"]))
                    sequence_K_list.append(self.mvsec.get_K(scene, "cam0"))
                    pose_interpolator_list.append(
                        self.mvsec.get_pose_interpolator(sequence_name)
                    )

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
            self.supported_sequence_dict = MVSECDataset.VAL_SEQUENCE_DICT
            self.supported_scene_list = list(self.supported_sequence_dict.keys())
            self.supported_sequence_list = []
            for scene in self.supported_scene_list:
                self.supported_sequence_list.extend(self.supported_sequence_dict[scene])
                for sequence_name in self.supported_sequence_dict[scene]:
                    pose_timestamp = self.mvsec.get_pose_timestamp(sequence_name)
                    ts_lower_bound, ts_higer_bound = np.min(pose_timestamp), np.max(
                        pose_timestamp
                    )
                    events_list.append(self.mvsec.get_events(sequence_name))
                    depth_and_image_data = self.mvsec.get_paired_depth_and_image(
                        sequence_name, "rect", "nearest"
                    )

                    # adapt the sequence to the pose timestamp
                    idx0 = np.searchsorted(
                        depth_and_image_data["depth_ts"], ts_lower_bound, side="right"
                    )
                    idx1 = np.searchsorted(
                        depth_and_image_data["depth_ts"], ts_higer_bound, side="left"
                    )
                    for key in depth_and_image_data.keys():
                        depth_and_image_data[key] = depth_and_image_data[key][idx0:idx1]

                    # crop the specific sequence
                    if sequence_name == "outdoor_day1":
                        for key in depth_and_image_data.keys():
                            depth_and_image_data[key] = depth_and_image_data[key][20:-40]
                    elif sequence_name == "indoor_flying4":
                        for key in depth_and_image_data.keys():
                            depth_and_image_data[key] = depth_and_image_data[key][20:-40]

                    depth_and_image_list.append(depth_and_image_data)
                    sequence_length_list.append(len(depth_and_image_data["depth_ts"]))
                    sequence_K_list.append(self.mvsec.get_K(scene, "cam0"))
                    pose_interpolator_list.append(
                        self.mvsec.get_pose_interpolator(sequence_name)
                    )

        self.events_list = events_list
        self.depth_and_image_list = depth_and_image_list
        self.pose_interpolator_list = pose_interpolator_list
        self.sequence_K_list = sequence_K_list
        self.sequence_length_list = np.array(sequence_length_list)

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

    def __len__(self) -> int:
        return self.sequence_length_list.sum()

    def get_events_at_timestamp(self, event_data: dict, timestamp: float, events_dt):
        """
        Get events at a specific timestamp

        Args:
            event_data: dict with keys 'x', 'y', 't', 'p'
            timestamp: timestamp at which to get events
            events_dt: time window around timestamp to get events

        Returns:
            a dict containing:
                - x: (N,) ndarray of x coordinates
                - y: (N,) ndarray of y coordinates
                - t: (N,) ndarray of timestamps
                - p: (N,) ndarray of polarities
        """
        # left side
        index0 = np.searchsorted(
            event_data["t"], timestamp - events_dt, side="left"
        )
        index1 = np.searchsorted(
            event_data["t"], timestamp, side="right"
        )
        # two sides
        # index0 = np.searchsorted(
        #     event_data["t"], timestamp - events_dt / 2, side="left"
        # )
        # index1 = np.searchsorted(
        #     event_data["t"], timestamp + events_dt / 2, side="right"
        # )
        return {
            "x": event_data["x"][index0:index1],
            "y": event_data["y"][index0:index1],
            "t": event_data["t"][index0:index1],
            "p": event_data["p"][index0:index1],
        }

    def get_depth_image_events_pose(
        self,
        sequence_depth_and_image,
        sequence_events,
        pose_interpolator,
        events_dt,
        depth_index,
    ) -> Dict[str, Any]:
        """
        get depth, image, events and pose at the specified index of the specified sequence.

        Args:
            sequence_depth_and_image (dict): the depth and image data of the specified sequence.
            sequence_events (dict): the events data of the specified sequence.
            pose_interpolator (PoseInterpolator): the pose interpolator of the specified sequence.
            events_dt (float): the time interval of each iteration.
            depth_index (int): the index of depth image.

        Returns:
            a dict containing:
                - depth (torch.Tensor): a tensor of depth image [H, W].
                - depth_ts (torch.Tensor): a tensor of depth image timestamp [1].
                - depth_mask (torch.Tensor): a tensor of depth mask [H, W].
                - image (np.ndarray): a grayscale image [H, W].
                - events (Dict): a dict of events.
                - pose (np.ndarray): a numpy array of pose [4, 4].
        """
        depth_ts = sequence_depth_and_image["depth_ts"][depth_index]
        depth = sequence_depth_and_image["depth"][depth_index]
        depth_mask = np.logical_not(np.isnan(depth))
        image = sequence_depth_and_image["image"][depth_index]
        image_index = sequence_depth_and_image["image_index"][depth_index]
        image_ts = sequence_depth_and_image["image_ts"][depth_index]
        image = np.expand_dims(image, axis=-1)

        pose = pose_interpolator.interpolate(depth_ts)
        # events = self.get_events_at_timestamp(sequence_events, depth_ts, events_dt)
        events = self.get_events_at_timestamp(sequence_events, image_ts, events_dt)

        return {
            "depth": torch.from_numpy(depth.astype("float32")).contiguous(),
            "depth_ts": torch.tensor(depth_ts.astype("float32"))
            .reshape(
                1,
            )
            .contiguous(),
            "depth_mask": torch.from_numpy(depth_mask).contiguous(),
            "image": image,
            "image_index": torch.tensor(image_index.astype("int32")),
            "image_ts": torch.tensor(image_ts.astype("float32"))
            .reshape(
                1,
            )
            .contiguous(),
            "events_raw": events,
            "pose": pose.astype("float32"),
        }

    def get_relative_pose(self, pose0: np.ndarray, pose1: np.ndarray) -> np.ndarray:
        """
        get the relative pose from pose0 to pose1.

        Args:
            pose0 (np.ndarray): the first pose.
            pose1 (np.ndarray): the second pose.

        Returns:
            a numpy array of relative pose [4, 4].
        """

        return pose1 @ np.linalg.inv(pose0)

    def augment_and_convert(
        self, data: Dict[str, Any], augment_pair: bool
    ) -> Dict[str, Any]:
        if self.is_train and self.use_aug:
            # augment the events in points form
            data["events_raw"] = self.augment_event_points(
                data["events_raw"]
            )  # form in points Dict[str, np.ndarray]
            # augment the image in numpy.ndarray form
            data["image"] = self.augment_image(
                data["image"]
            )  # form in numpy.ndarray (H, W, C)

        # transform to tensor
        data["image"] = (
            torch.from_numpy(data["image"]).permute(2, 0, 1).float()
        )  # form in torch.Tensor (C, H, W)
        # calculate the events image
        events_image = draw_events_accumulation_image(
            data["events_raw"], MVSECDataset.RESOLUTION
        )
        data["events_image"] = (
            torch.from_numpy(events_image).unsqueeze(0).float()
        )  # form in torch.Tensor (C, H, W)
        # convert events into representation
        data["events_rep"] = self.representation(
            data["events_raw"],
            (self.channel, MVSECDataset.RESOLUTION[1], MVSECDataset.RESOLUTION[0]),
        )

        if self.is_train and self.use_aug and augment_pair:
            # augment the pair
            data["events_rep"], data["image"], data["events_image"] = self.augment_pair(
                data["events_rep"], data["image"], data["events_image"]
            )

        return data

    def __getitem__(self, index: int) -> Any:
        # find the sequence index
        sequence_index = np.searchsorted(
            self.sequence_length_list.cumsum(), index, side="right"
        )
        if sequence_index > 0:
            index = index - self.sequence_length_list[:sequence_index].sum()

        # load camera K
        K = self.sequence_K_list[sequence_index]

        # load depth, image, events and pose of the sequence
        depth_and_image_data = self.depth_and_image_list[sequence_index]
        sequence_event_data = self.events_list[sequence_index]
        pose_interpolator = self.pose_interpolator_list[sequence_index]

        # load depth, image, events and pose at the specified index
        data0 = self.get_depth_image_events_pose(
            depth_and_image_data,
            sequence_event_data,
            pose_interpolator,
            self.event_dt,
            index,
        )
        data0["K"] = torch.from_numpy(K.astype("float32")).contiguous()

        # load depth, image, events and pose at a random index of the same sequence
        if self.select_matching_pair:
            sequence_length = self.sequence_length_list[sequence_index]

            index1_right = (
                sequence_length if index + 60 > sequence_length else index + 60
            )

            index1 = np.random.randint(index, index1_right)
            data1 = self.get_depth_image_events_pose(
                depth_and_image_data,
                sequence_event_data,
                pose_interpolator,
                self.event_dt,
                index1,
            )

            # augment the matching pair
            data0 = self.augment_and_convert(data0, augment_pair=False)
            data1 = self.augment_and_convert(data1, augment_pair=False)
            data1['K'] = data0['K']

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

            return data0, data0, torch.eye(4, dtype=torch.float32), torch.eye(4, dtype=torch.float32)
        
        
class MVSECDataset_RPE_TRAIN(MVSECDataset):
    def __init__(self, cfg, train_indices_path_list: List[PathLike]) -> None:
        super().__init__(cfg, is_train=True, use_aug=True)
        
        assert len(train_indices_path_list) == len(self.supported_sequence_list)

        indices_list = []
        indices_length_list = []
        for path in train_indices_path_list:
            indices = np.loadtxt(path)
            indices_list.append(indices)
            indices_length_list.append(indices.shape[0])
        
        self.indices_list = indices_list
        self.indices_length_list = np.array(indices_length_list)
        
    def __len__(self) -> int:
        return np.array(self.indices_length_list).sum()
    
    def __getitem__(self, index: int) -> Any:
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

        # load depth, image, events and pose of the sequence
        depth_and_image_data = self.depth_and_image_list[sequence_index]
        sequence_event_data = self.events_list[sequence_index]
        pose_interpolator = self.pose_interpolator_list[sequence_index]
        
        # load data for the view0
        data0 = self.get_depth_image_events_pose(
            depth_and_image_data,
            sequence_event_data,
            pose_interpolator,
            self.event_dt,
            view0_index,
        )
        data0["K"] = torch.from_numpy(K.astype("float32")).contiguous()
        
        # load data for the view1
        data1 = self.get_depth_image_events_pose(
            depth_and_image_data,
            sequence_event_data,
            pose_interpolator,
            self.event_dt,
            view1_index,
        )
        data1["K"] = torch.from_numpy(K.astype("float32")).contiguous()
        
        # convert data  
        data0 = self.augment_and_convert(data0, augment_pair=False)
        data1 = self.augment_and_convert(data1, augment_pair=False)
        
        if self.is_train:
            del data0["events_raw"]
            del data1["events_raw"]

        # calculate relative pose
        T_0to1 = self.get_relative_pose(data0["pose"], data1["pose"])
        T_0to1 = torch.from_numpy(T_0to1.astype("float32")).contiguous()
        T_1to0 = self.get_relative_pose(data1["pose"], data0["pose"])
        T_1to0 = torch.from_numpy(T_1to0.astype("float32")).contiguous()
        
        return data0, data1, T_0to1, T_1to0
        
        
class MVSECDataset_RPE_VAL(MVSECDataset):
    def __init__(self, cfg, val_indices_path_list: List[PathLike]) -> None:
        super().__init__(cfg, is_train=False, use_aug=False)
        
        assert len(val_indices_path_list) == len(self.supported_sequence_list)

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
    
    def __getitem__(self, index: int) -> Any:
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

        # load depth, image, events and pose of the sequence
        depth_and_image_data = self.depth_and_image_list[sequence_index]
        sequence_event_data = self.events_list[sequence_index]
        pose_interpolator = self.pose_interpolator_list[sequence_index]
        
        # load data for the view0
        data0 = self.get_depth_image_events_pose(
            depth_and_image_data,
            sequence_event_data,
            pose_interpolator,
            self.event_dt,
            view0_index,
        )
        data0["K"] = torch.from_numpy(K.astype("float32")).contiguous()
        
        # load data for the view1
        data1 = self.get_depth_image_events_pose(
            depth_and_image_data,
            sequence_event_data,
            pose_interpolator,
            self.event_dt,
            view1_index,
        )
        data1["K"] = torch.from_numpy(K.astype("float32")).contiguous()
        
        # convert data  
        data0 = self.augment_and_convert(data0, augment_pair=False)
        data1 = self.augment_and_convert(data1, augment_pair=False)

        # calculate relative pose
        T_0to1 = self.get_relative_pose(data0["pose"], data1["pose"])
        T_0to1 = torch.from_numpy(T_0to1.astype("float32")).contiguous()
        T_1to0 = self.get_relative_pose(data1["pose"], data0["pose"])
        T_1to0 = torch.from_numpy(T_1to0.astype("float32")).contiguous()
        
        return data0, data1, T_0to1, T_1to0


class MVSECVisualizer(MVSEC):
    def __init__(self, data_path: PathLike) -> None:
        super().__init__(data_path)

    def depth_to_color(self, depth):
        """
        Convert depth map to color map
        """
        depth_min = depth[~np.isnan(depth)].min()
        depth_max = depth[~np.isnan(depth)].max()
        depth = (depth - depth_min) / (depth_max - depth_min) * 255
        depth[np.isnan(depth)] = 0
        depth = depth.astype(np.uint8)

        return cv.applyColorMap(depth, cv.COLORMAP_JET)

    def visualize_events(
        self,
        sequence_name: str,
        event_type: str = "rect",
        dt: float = 0.1,
        save_path: PathLike = None,
        show_in_window: bool = True,
    ) -> None:
        """
        visualize events of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            event_type (str): the type of events, 'raw' or 'rect'.
            dt (float): the time interval of each iteration.
            save_path (PathLike): the path to save the visualization.
            show_in_window (bool): whether to show the visualization in a window.
        """

        assert (
            sequence_name in self.supported_sequence_list
        ), f"The specified sequence '{sequence_name}' is not supported."

        from .visualize import draw_events_accumulation_image

        events_iterator = self.get_events_iterator(sequence_name, dt, event_type)

        if save_path is not None:
            save_dir = os.path.join(save_path, sequence_name, f"events_{dt}")
            os.makedirs(save_dir, exist_ok=True)

        for event in events_iterator:
            dt_elapsed, event_data = event

            event_image = draw_events_accumulation_image(event_data, MVSEC.RESOLUTION)

            if save_path is not None:
                event_save_path = os.path.join(save_dir, f"{dt_elapsed:06f}.png")
                cv.imwrite(event_save_path, event_image)

            if show_in_window:
                cv.imshow("event", event_image)
                if cv.waitKey(0) == ord("q"):
                    break
                elif cv.waitKey(0) == ord("s"):
                    continue

    def visualize_depth_and_image(
        self,
        sequence_name: str,
        depth_type: str = "rect",
        save_path: PathLike = None,
        show_in_window: bool = True,
    ) -> None:
        """
        visualize depth image and image of the specified sequence.

        Args:
            sequence_name (str): the name of sequence.
            depth_type (str): the type of depth image, 'raw' or 'rect'.
            save_path (PathLike): the path to save the visualization.
            show_in_window (bool): whether to show the visualization in a window.
        """

        assert (
            sequence_name in self.supported_sequence_list
        ), f"The specified sequence '{sequence_name}' is not supported."

        data = self.get_paired_depth_and_image(sequence_name, depth_type)
        depth_list = data["depth"]
        image_list = data["image"]

        if save_path is not None:
            depth_save_dir = os.path.join(save_path, sequence_name, "depth")
            image_save_dir = os.path.join(save_path, sequence_name, "image")
            os.makedirs(depth_save_dir, exist_ok=True)
            os.makedirs(image_save_dir, exist_ok=True)

        for i in range(len(depth_list)):
            depth = depth_list[i]
            image = image_list[i]
            depth_color = self.depth_to_color(depth)

            if save_path is not None:
                depth_save_path = os.path.join(depth_save_dir, f"{i:06d}.png")
                image_save_path = os.path.join(image_save_dir, f"{i:06d}.png")
                cv.imwrite(depth_save_path, depth_color)
                cv.imwrite(image_save_path, image)

            if show_in_window:
                cv.imshow("depth", depth_color)
                cv.imshow("image", image)

                if cv.waitKey(0) == ord("s"):
                    continue
                elif cv.waitKey(0) == ord("q"):
                    break


def fetch_mvsec_dataloader(cfg: DictConfig, split: str, logger, rank=-1, world_size=1):
    if split == "train":
        is_train = True

        if cfg.train_on_rpe_data:
            indices_list = [
            "./indoor_flying1_final_indices.txt",
            "./indoor_flying2_final_indices.txt",
            "./indoor_flying3_final_indices.txt",
            "./outdoor_day2_final_indices.txt",
        ]
            data = MVSECDataset_RPE_TRAIN(cfg, indices_list)
        else:
            data = MVSECDataset(cfg, is_train=is_train)

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
        indices_list = [
            "./indoor_flying4_final_indices.txt",
            "./outdoor_day1_final_indices.txt",
        ]
        data = MVSECDataset_RPE_VAL(cfg, indices_list)

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
    # import cv2 as cv
    mvsec = MVSECVisualizer("data/MVSEC")
    # print(len(mvsec))
    # mvsec.visualize_depth_and_image('indoor_flying1', 'rect', 'data/MVSEC/indoor_flying/visualization', True)
    mvsec.visualize_events(
        "indoor_flying4", "rect", 1.0, "data/MVSEC/indoor_flying/visualization", True
    )

    # val_loader = DataLoader(MVSECDataset("data/MVSEC", 'voxel_grid', 0.05, 'val'), batch_size=2, shuffle=True)

    # for i, (data0, data1) in enumerate(val_loader):

    #     # print(data0['depth'].shape)
    #     # print(data0['depth_ts'].shape)
    #     # print(data0['image'].shape)
    #     # print(data0['events'].shape)
    #     # print(data0['pose'].shape)
    #     # print('------')

    #     # print(data1['depth'].shape)
    #     # print(data1['depth_ts'].shape)
    #     # print(data1['image'].shape)
    #     # print(data1['events'].shape)
    #     # print(data1['pose'].shape)

    #     depth = data0['depth'][0]
    #     depth_mask = data0['depth_mask'][0]

    #     print(depth)
    #     a = depth * depth_mask
    #     print(a)

    #     break
