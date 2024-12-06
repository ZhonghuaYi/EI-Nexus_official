from typing import Tuple
import numpy as np
import os
import cv2 as cv
import h5py

from MVSEC import MVSEC
from visualize import draw_events_accumulation_image


class MVSECRectifier:
    def __init__(self, data_path: str) -> None:
        self.mvsec = MVSEC(data_path)
        self.data_path = data_path

    def recitfy_images(self, images: np.ndarray, x_map, y_map) -> np.ndarray:
        retified_images = np.zeros_like(images)
        for i in range(images.shape[0]):
            retified_images[i] = cv.remap(images[i], x_map, y_map, cv.INTER_LINEAR)

        return retified_images

    def rectify_events(
        self, events: np.ndarray, x_map, y_map, resolution
    ) -> np.ndarray:
        W, H = resolution

        retified_events = events.copy()
        origin_x = np.round(events[:, 0]).astype(np.int32)
        origin_y = np.round(events[:, 1]).astype(np.int32)

        rectified_x = x_map[origin_y, origin_x]
        rectified_y = y_map[origin_y, origin_x]
        retified_events[:, 0] = rectified_x
        retified_events[:, 1] = rectified_y

        mask = (
            (retified_events[:, 0] >= 0)
            & (retified_events[:, 0] < W - 1)
            & (retified_events[:, 1] >= 0)
            & (retified_events[:, 1] < H - 1)
        )
        retified_events = retified_events[mask]

        return retified_events

    def rectify_sequence(
        self, sequence_name
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        rectify the specified sequence.

        Args:
            sequence_name (str): the name of sequence.

        Returns:
            rectified_left_images (np.ndarray): the rectified left images.
            rectified_right_images (np.ndarray): the rectified right images.
            rectified_left_events (np.ndarray): the rectified left events.
            rectified_right_events (np.ndarray): the rectified right events.
        """

        assert (
            sequence_name in self.mvsec.supported_sequence_list
        ), f"The specified sequence '{sequence_name}' is not supported."

        scene_name = None
        for scene in self.mvsec.supported_scene_list:
            if sequence_name in self.mvsec.supported_sequence_dict[scene]:
                scene_name = scene
                break

        if scene_name is None:
            raise ValueError(f"Sequence '{sequence_name}' not found.")

        # load calibration map
        calibration_map = self.mvsec.load_calibration_map(scene_name)
        left_x_map = calibration_map["left_x_map"]
        left_y_map = calibration_map["left_y_map"]
        right_x_map = calibration_map["right_x_map"]
        right_y_map = calibration_map["right_y_map"]

        # load calibration parameters
        calib_data = self.mvsec.load_calibration_yaml(scene_name)
        left_K = np.eye(3)
        left_K[[0, 1, 0, 1], [0, 1, 2, 2]] = calib_data["cam0"]["intrinsics"]
        right_K = np.eye(3)
        right_K[[0, 1, 0, 1], [0, 1, 2, 2]] = calib_data["cam1"]["intrinsics"]

        # init rectify map
        image_left_x_map, image_left_y_map = cv.fisheye.initUndistortRectifyMap(
            left_K,
            np.array(calib_data["cam0"]["distortion_coeffs"]),
            np.array(calib_data["cam0"]["rectification_matrix"]),
            np.array(calib_data["cam0"]["projection_matrix"]),
            calib_data["cam0"]["resolution"],
            cv.CV_32FC1,
        )
        image_right_x_map, image_right_y_map = cv.fisheye.initUndistortRectifyMap(
            right_K,
            np.array(calib_data["cam1"]["distortion_coeffs"]),
            np.array(calib_data["cam1"]["rectification_matrix"]),
            np.array(calib_data["cam1"]["projection_matrix"]),
            calib_data["cam1"]["resolution"],
            cv.CV_32FC1,
        )

        # load data
        data_file = self.mvsec.load_data(sequence_name)

        # load images and events
        left_raw_image = np.array(data_file["davis"]["left"]["image_raw"])
        # right_raw_image = np.array(data_file['davis']['right']['image_raw'])
        left_events = np.array(data_file["davis"]["left"]["events"])
        # right_events = np.array(data_file['davis']['right']['events'])

        # rectify images
        rectified_left_images = self.recitfy_images(
            left_raw_image, image_left_x_map, image_left_y_map
        )
        # rectified_right_images = self.recitfy_images(right_raw_image, image_right_x_map, image_right_y_map)

        # rectify events
        rectified_left_events = self.rectify_events(
            left_events, left_x_map, left_y_map, calib_data["cam0"]["resolution"]
        )
        # rectified_right_events = self.rectify_events(right_events, right_x_map, right_y_map, calib_data['cam1']['resolution'])

        # return rectified_left_images, rectified_right_images, rectified_left_events, rectified_right_events
        return rectified_left_images, None, rectified_left_events, None

    def save_rectified_sequence(self, sequence_name):
        """
        save the rectified sequence.

        Args:
            sequence_name (str): the name of sequence.
        """

        assert (
            sequence_name in self.mvsec.supported_sequence_list
        ), f"The specified sequence '{sequence_name}' is not supported."

        scene_name = None
        for scene in self.mvsec.supported_scene_list:
            if sequence_name in self.mvsec.supported_sequence_dict[scene]:
                scene_name = scene
                break

        save_file = os.path.join(
            self.data_path, scene_name, f"{sequence_name}_rectified.h5"
        )
        with h5py.File(save_file, "w") as h5f:
            left_images, right_images, left_events, right_events = (
                self.rectify_sequence(sequence_name)
            )
            # data_dict = {
            #     'davis':{
            #         'left':{
            #             'image_rect': left_images,
            #             'events_rect': left_events
            #         },
            #         'right':{
            #             'image_rect': right_images,
            #             'events_rect': right_events
            #         }
            #     }
            # }
            davis = h5f.create_group("davis")
            left = davis.create_group("left")
            # right = davis.create_group('right')
            left.create_dataset("image_rect", data=left_images)
            left.create_dataset("events_rect", data=left_events)
            # right.create_dataset('image_rect', data=right_images)
            # right.create_dataset('events_rect', data=right_events)

        print(f"Sequence '{sequence_name}' rectified and saved to '{save_file}'.")


if __name__ == "__main__":
    rectifier = MVSECRectifier("data/MVSEC")
    rectify_sequence = ["outdoor_day2"]
    for seq in rectify_sequence:
        rectifier.save_rectified_sequence(seq)
