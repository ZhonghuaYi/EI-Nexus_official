from typing import Dict, Tuple, List, Any

import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv


def draw_events(events: Dict[str, np.ndarray], ax=None, **kwargs):
    """
    Draw events in 3D space.

    Args:
        events: Dictionary with keys "x", "y", "t", "p" and values numpy arrays of the same length.
        ax: Matplotlib 3D axis.
        **kwargs: Keyword arguments passed to ax.scatter3D.
    """
    if ax is None:
        ax = plt.axes(projection="3d")

    ax.scatter3D(events["x"], events["t"], events["y"], c=events["p"], **kwargs)


def draw_events_accumulation_image(events: Any, image_shape):
    """
    Draw events as an accumulation image.

    Args:
        events: Dictionary with keys "x", "y", "t", "p" and values numpy arrays of the same length, or numpy array of shape [N, 4].
        image_shape: Shape of the image.
    """
    image = np.zeros((image_shape[1], image_shape[0]))

    if isinstance(events, dict):
        for i in range(events["t"].shape[0]):
            # image[int(events["y"][i]), int(events["x"][i])] += events["p"][i]
            image[int(events["y"][i]), int(events["x"][i])] += 1
        # image[events["y"].astype(np.int32), events["x"].astype(np.int32)] = 255
    elif isinstance(events, np.ndarray):
        for i in range(events.shape[0]):
            image[int(events[i, 1]), int(events[i, 0])] += 2 * events[i, 3] - 1
            # image[int(events[i, 1]), int(events[i, 0])] += 1
    else:
        raise ValueError("events must be a dictionary or numpy array.")

    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image[image > 255] = 255
    # plt.imshow(image)
    image = image.astype(np.uint8)
    return image


def draw_events_color_image(events: Any, image_shape):
    """
    Draw events as an accumulation image.

    Args:
        events: Dictionary with keys "x", "y", "t", "p" and values numpy arrays of the same length, or numpy array of shape [N, 4].
        image_shape: Shape of the image.
    """
    image = np.zeros((image_shape[1], image_shape[0], 3))

    if isinstance(events, dict):
        for i in range(events["t"].shape[0]):
            if events["p"][i] == 1:
                image[int(events["y"][i]), int(events["x"][i]), 0] = 0
                image[int(events["y"][i]), int(events["x"][i]), 2] = 128
            else:
                image[int(events["y"][i]), int(events["x"][i]), 0] = 255
                image[int(events["y"][i]), int(events["x"][i]), 2] = 0
    elif isinstance(events, np.ndarray):
        for i in range(events.shape[0]):
            if events[i, 3] == 1:
                image[int(events[i, 1]), int(events[i, 0]), 2] = 255
            else:
                image[int(events[i, 1]), int(events[i, 0]), 0] = 255
    else:
        raise ValueError("events must be a dictionary or numpy array.")

    image = image.astype(np.uint8)
    return image


def visualize_paired_depth_and_image(depth: np.ndarray, image: np.ndarray):
    """
    Visualize paired depth and image.

    Args:
        depth: Depth image in [N, H, W].
        image: Image in [N, H, W].
    """
    for i in range(len(depth)):
        depth_image = depth[i]
        image_image = image[i]

        depth_image = cv.normalize(depth_image, None, 0, 255, cv.NORM_MINMAX)
        depth_image = depth_image.astype(np.uint8)
        image_image = image_image.astype(np.uint8)

        cv.imshow("depth", depth_image)
        cv.imshow("image", image_image)
        if cv.waitKey(0) == ord("q"):
            break


if __name__ == "__main__":
    pass
