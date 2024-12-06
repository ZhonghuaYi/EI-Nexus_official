from typing import List, Dict, Tuple

import torch
import cv2 as cv
import numpy as np


def time_normalization(events: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Normalize the time of events to [0, 1].

    Args:
        events (dict): Dictionary with keys "x", "y", "t", "p" and values numpy arrays of the same length.

    Returns:
        events (dict): Dictionary with keys "x", "y", "t", "p" and values numpy arrays of the same length.
    """

    events["t"] = events["t"] - events["t"][0]
    events["t"] = events["t"] / (events["t"][-1] + 1e-8)

    return events


@torch.no_grad()
def events_to_time_surface(events: Dict, input_size: Tuple) -> torch.Tensor:
    bins, H, W = input_size
    n_bins = bins // 2

    events = time_normalization(events)

    time_surface = np.zeros(input_size, dtype=np.float32)
    t = events["t"]
    dt_bin = 1.0 / n_bins
    x0 = events["x"].astype(np.int32)
    y0 = events["y"].astype(np.int32)
    p0 = events["p"].astype(np.int32)
    t0 = events["t"]

    # iterate over bins
    for i_bin in range(n_bins):
        t0_bin = i_bin * dt_bin
        t1_bin = t0_bin + dt_bin

        # mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
        # x_bin, y_bin, p_bin, t_bin = x[mask_t], y[mask_t], p[mask_t], time[mask_t]
        idx0 = np.searchsorted(t, t0_bin, side="left")
        idx1 = np.searchsorted(t, t1_bin, side="right")
        x_bin = x0[idx0:idx1]
        y_bin = y0[idx0:idx1]
        p_bin = p0[idx0:idx1]
        t_bin = t0[idx0:idx1]

        # n_events = len(x_bin)
        # for i in range(n_events):
        #     if 0 <= x_bin[i] < W and 0 <= y_bin[i] < H:
        #         time_surface[2 * i_bin + p_bin[i], y_bin[i], x_bin[i]] = t_bin[i]

        time_surface[2 * i_bin + p_bin, y_bin, x_bin] = t_bin

    time_surface = torch.from_numpy(time_surface).float().contiguous()

    return time_surface


@torch.no_grad()
def events_to_voxel_grid(
    events: Dict, input_size: Tuple, normalize: bool = True
) -> torch.Tensor:
    bins, H, W = input_size

    events = time_normalization(events)
    events["x"] = torch.from_numpy(events["x"].astype("float32"))
    events["y"] = torch.from_numpy(events["y"].astype("float32"))
    events["p"] = torch.from_numpy(events["p"].astype("float32"))
    events["t"] = torch.from_numpy(events["t"].astype("float32"))

    voxel_grid = torch.zeros((input_size), dtype=torch.float, requires_grad=False)

    t_norm = events["t"]
    t_norm = (bins - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])

    x0 = events["x"].int()
    y0 = events["y"].int()
    t0 = t_norm.int()

    # value = 2 * events['p'] - 1  # p in [0, 1]
    value = events["p"]  # p in [-1, 1]
    value[value < 1] = -1

    for xlim in [x0, x0 + 1]:
        for ylim in [y0, y0 + 1]:
            for tlim in [t0, t0 + 1]:

                mask = (
                    (xlim < W)
                    & (xlim >= 0)
                    & (ylim < H)
                    & (ylim >= 0)
                    & (tlim >= 0)
                    & (tlim < bins)
                )
                interp_weights = (
                    value
                    * (1 - (xlim - events["x"]).abs())
                    * (1 - (ylim - events["y"]).abs())
                    * (1 - (tlim - t_norm).abs())
                )

                index = H * W * tlim.long() + W * ylim.long() + xlim.long()

                voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

    if normalize:
        mask = torch.nonzero(voxel_grid, as_tuple=True)
        if mask[0].size()[0] > 0:
            mean = voxel_grid[mask].mean()
            std = voxel_grid[mask].std()
            if std > 0:
                voxel_grid[mask] = (voxel_grid[mask] - mean) / std
            else:
                voxel_grid[mask] = voxel_grid[mask] - mean

    return voxel_grid


@torch.no_grad()
def events_to_voxel_grid_new(
    events: Dict, input_size: Tuple, normalize: bool = True
) -> torch.Tensor:

    def events_to_image_torch(xs, ys, ps, device=None, sensor_size=(260, 346)):
        """
        Method to turn event tensor to image. Allows for bilinear interpolation.
            :param xs: tensor of x coords of events
            :param ys: tensor of y coords of events
            :param ps: tensor of event polarities/weights
            :param device: the device on which the image is. If none, set to events device
            :param sensor_size: the size of the image sensor/output image
        """
        if device is None:
            device = xs.device

        img_size = list(sensor_size)

        img = torch.zeros(img_size).to(device)
        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        img.index_put_((ys, xs), ps, accumulate=True)
        return img

    num_bins, H, W = input_size

    xs = torch.from_numpy(events["x"].astype("float32"))
    ys = torch.from_numpy(events["y"].astype("float32"))
    ts = torch.from_numpy(events["t"].astype("float32"))
    ps = torch.from_numpy(events["p"].astype("float32"))

    bins = []
    dt = ts[-1] - ts[0]
    if dt.item() < 1e-9:
        t_norm = torch.linspace(0, num_bins - 1, steps=len(ts))
    else:
        t_norm = (ts - ts[0]) / dt * (num_bins - 1)
    zeros = torch.zeros(t_norm.size())
    for bi in range(num_bins):
        bilinear_weights = torch.max(zeros, 1.0 - torch.abs(t_norm - bi))
        weights = ps * bilinear_weights
        vb = events_to_image_torch(xs, ys, weights, sensor_size=(H, W))
        bins.append(vb)
    bins = torch.stack(bins)
    return bins


@torch.no_grad()
def events_to_event_stack(events: Dict, input_size: Tuple) -> torch.Tensor:
    bins, H, W = input_size

    events = time_normalization(events)

    t = events["t"]
    dt_bin = 1.0 / bins
    x0 = events["x"].astype(np.int32)
    y0 = events["y"].astype(np.int32)
    p0 = 2 * events["p"].astype(np.int32) - 1
    t0 = events["t"]

    event_stack = np.zeros(input_size, dtype=np.float32)

    # iterate over bins
    for i_bin in range(bins):
        t0_bin = i_bin * dt_bin
        t1_bin = t0_bin + dt_bin

        # mask_t = np.logical_and(time > t0_bin, time <= t1_bin)
        # x_bin, y_bin, p_bin, t_bin = x[mask_t], y[mask_t], p[mask_t], time[mask_t]
        idx0 = np.searchsorted(t, t0_bin, side="left")
        idx1 = np.searchsorted(t, t1_bin, side="right")
        x_bin = x0[idx0:idx1]
        y_bin = y0[idx0:idx1]
        p_bin = p0[idx0:idx1]

        n_events = len(x_bin)
        for i in range(n_events):
            if 0 <= x_bin[i] < W and 0 <= y_bin[i] < H:
                event_stack[i_bin, y_bin[i], x_bin[i]] += p_bin[i]

    event_stack = torch.from_numpy(event_stack).float().contiguous()

    return event_stack


@torch.no_grad()
def events_to_distance_map(events: Dict, input_size: Tuple) -> torch.Tensor:
    bins, H, W = input_size

    events = time_normalization(events)

    event_distance_map = np.zeros(input_size, dtype=np.float32)
    channel_t = 1 / bins

    for i in range(bins):

        index0 = np.searchsorted(events["t"], i * channel_t, side="left")
        index1 = np.searchsorted(events["t"], (i + 1) * channel_t, side="right")

        event_map = np.zeros((H, W), dtype=np.uint8)

        event_slice = {
            "p": events["p"][index0:index1],
            "t": events["t"][index0:index1],
            "x": events["x"][index0:index1],
            "y": events["y"][index0:index1],
        }

        event_map[
            event_slice["y"].astype(np.int32), event_slice["x"].astype(np.int32)
        ] = 1

        event_distance_map_slice = cv.distanceTransform(1 - event_map, cv.DIST_L2, 3)

        event_distance_map[i, :, :] = event_distance_map_slice

    event_distance_map = torch.from_numpy(event_distance_map).float().contiguous()

    return event_distance_map
