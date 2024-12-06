from typing import Any, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import albumentations as A


def random_time_scale(
    events: Dict[str, np.ndarray], scale: Tuple[float, float] = (0.8, 1.2)
) -> Dict[str, np.ndarray]:
    """
    Randomly scale the time of events.
    """
    if scale is not None:
        t_start = events["t"][..., 0]
        events["t"] = (events["t"] - t_start) * np.random.uniform(
            scale[0], scale[1]
        ) + t_start
    return events


def slice_events(
    events: Dict[str, np.ndarray], slice_dt: Tuple[float, float], mode: str
) -> Dict[str, np.ndarray]:
    """
    Slice the events into a time crop with the given time interval dt(milliseconds).
    """
    if slice_dt is not None:
        dt = np.random.uniform(slice_dt[0], slice_dt[1])
        t = events["t"]
        t_length = t[..., -1] - t[..., 0]  # microseconds
        if t_length > dt * 1000.0:
            if mode == "start":
                t_start = t[..., 0]
                t_end = t_start + dt / 1000.0
                t_mask = (t >= t_start) & (t < t_end)
                events = {k: v[t_mask] for k, v in events.items()}
            elif mode == "end":
                t_end = t[..., -1]
                t_start = t_end - dt / 1000.0
                t_mask = (t >= t_start) & (t < t_end)
                events = {k: v[t_mask] for k, v in events.items()}
            elif mode == "middle":
                t_middle = (t[..., 0] + t[..., -1]) / 2
                t_start = t_middle - dt / 2000.0
                t_end = t_middle + dt / 2000.0
                t_mask = (t >= t_start) & (t < t_end)
                events = {k: v[t_mask] for k, v in events.items()}
            elif mode == "random":
                t_start = t[..., 0]
                t_end = t[..., -1] - dt / 1000.0
                t_random = np.random.uniform(t_start, t_end)
                t_mask = (t >= t_random) & (t < t_random + dt / 1000.0)
                events = {k: v[t_mask] for k, v in events.items()}
            else:
                raise ValueError(f"Unknown mode: {mode}")

    return events


def random_time_flip(
    events: Dict[str, np.ndarray], p: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Randomly flip the time of events.
    """
    if np.random.rand() < p:
        events["t"] = events["t"][::-1]
        events["t"] = events["t"][0] - events["t"]
        events["p"] = -events["p"].astype(
            np.float32
        )  # Inversion in time means inversion in polarity

    return events


def add_correlated_events(
    event: Dict[str, np.ndarray],
    xy_std: float = 1.5,
    ts_std: float = 0.1,
    percent: Tuple[float, float] = (0.001, 0.01),
) -> Dict[str, np.ndarray]:
    """
    Add correlated events to the event.

    Args:
        event (dict): Dictionary with keys "x", "y", "t", "p" and values numpy arrays of the same length.
        xy_std (float): Standard deviation of the gaussian distribution for the x and y coordinates.
        ts_std (float): Standard deviation of the gaussian distribution for the timestamp.

    Returns:
        event (dict): Dictionary with keys "x", "y", "t", "p" and values numpy arrays of the same length.
    """

    N = event["x"].shape[0]
    if N < 1000:
        return event
    to_add = np.random.randint(int(N * percent[0]), int(N * percent[1]))
    event_new = {
        "x": event["x"] + np.random.normal(0, xy_std, size=event["x"].shape),
        "y": event["y"] + np.random.normal(0, xy_std, size=event["x"].shape),
        "t": event["t"] + np.random.normal(0, ts_std / 1000.0, size=event["x"].shape),
        "p": event["p"],
    }
    idx = np.random.choice(np.arange(N), size=to_add, replace=False)
    for key in event_new:
        event_new[key] = event_new[key][idx]
    event_new["x"] = np.clip(event_new["x"], 0, event["x"].max())
    event_new["y"] = np.clip(event_new["y"], 0, event["y"].max())
    for key in event:
        event[key] = np.concatenate((event[key], event_new[key]))
    idx_sorted = np.argsort(event["t"], axis=0)[::-1]
    for key in event:
        event[key] = event[key][idx_sorted]
    return event


def random_delete_events(
    event: Dict[str, np.ndarray], percent: Tuple[float, float] = (0.001, 0.01)
) -> Dict[str, np.ndarray]:
    """
    Randomly delete events.
    """
    N = event["x"].shape[0]
    if N > 1000:
        to_delete = np.random.randint(int(N * percent[0]), int(N * percent[1]))
        idx = np.random.choice(np.arange(N), size=N - to_delete, replace=False)
        for key in event:
            event[key] = event[key][idx]
    return event


def random_crop_pair(
    x: torch.Tensor, y: torch.Tensor, crop_size: Tuple[int, int], mask=None
):

    assert x.shape[-2:] == y.shape[-2:], f"Shape mismatch: {x.shape} and {y.shape}"
    assert (
        x.shape[-2] >= crop_size[0] and x.shape[-1] >= crop_size[1]
    ), f"Shape mismatch: {x.shape} and {crop_size}"

    H, W = x.shape[-2:]
    h, w = crop_size
    x0 = np.random.randint(0, H - h)
    y0 = np.random.randint(0, W - w)
    x = x[..., x0 : x0 + h, y0 : y0 + w]
    y = y[..., x0 : x0 + h, y0 : y0 + w]
    if mask is not None:
        mask = mask[..., x0 : x0 + h, y0 : y0 + w]
        return x, y, mask

    return x, y


def random_flip_pair(
    x: torch.Tensor, y: torch.Tensor, p_h: float = 0.5, p_w: float = 0.5, mask=None
):
    if np.random.rand() < p_h:
        x = torch.flip(x, dims=(-2,))
        y = torch.flip(y, dims=(-2,))
        if mask is not None:
            mask = torch.flip(mask, dims=(-2,))
    if np.random.rand() < p_w:
        x = torch.flip(x, dims=(-1,))
        y = torch.flip(y, dims=(-1,))
        if mask is not None:
            mask = torch.flip(mask, dims=(-1,))

    if mask is not None:
        return x, y, mask
    else:
        return x, y


def random_rotate_pair(
    x: torch.Tensor, y: torch.Tensor, angle: float = 10.0, mask=None
):
    angle = np.random.uniform(-angle, angle)
    x = TF.rotate(x, angle)
    y = TF.rotate(y, angle)
    if mask is not None:
        mask = TF.rotate(mask, angle)
        return x, y, mask
    return x, y


class EventPointsAugmentation:

    def __init__(
        self,
        time_scale: Tuple[float, float] = (0.5, 2.0),
        slice_dt: Tuple[float, float] = (30, 100),
        slice_mode: str = "random",
        flip_p: float = 0.5,
        xy_std: float = 1.5,
        ts_std: float = 0.1,
        add_percent: Tuple[float, float] = (0.01, 0.001),
        del_percent: Tuple[float, float] = (0.01, 0.001),
    ) -> None:
        self.time_scale = time_scale
        self.slice_dt = slice_dt
        self.slice_mode = slice_mode
        self.flip_p = flip_p
        self.xy_std = xy_std
        self.ts_std = ts_std
        self.add_percent = add_percent
        self.del_percent = del_percent

    def __call__(self, events) -> Any:
        # events = random_time_scale(events, scale=self.time_scale)
        # events = slice_events(events, slice_dt=self.slice_dt, mode=self.slice_mode)
        # events = random_time_flip(events, p=self.flip_p)
        # events = add_correlated_events(events, self.xy_std, self.ts_std, self.add_percent)
        # events = random_delete_events(events, self.del_percent)
        return events


class PairAugmentation:
    def __init__(
        self,
        crop_size: Tuple[int, int],
        flip_p_h: float = 0.5,
        flip_p_w: float = 0.5,
        rotate_angle: float = 10.0,
    ) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.flip_p_h = flip_p_h
        self.flip_p_w = flip_p_w
        self.rotate_angle = rotate_angle

    def __call__(self, x: torch.Tensor, y: torch.Tensor, mask=None) -> Any:
        x, y, mask = random_flip_pair(
            x, y, p_h=self.flip_p_h, p_w=self.flip_p_w, mask=mask
        )
        x, y, mask = random_rotate_pair(x, y, angle=self.rotate_angle, mask=mask)
        x, y, mask = random_crop_pair(x, y, crop_size=self.crop_size, mask=mask)

        return x, y, mask


class ImageArrayAugmentation:
    def __init__(
        self,
        gamma_p: float = 0.1,
        gamma_limit: Tuple[int, int] = (15, 65),
        saturation_p: float = 0.1,
        val_shift_limit: Tuple[int, int] = (-100, -40),
        brightness_p: float = 0.5,
        brightness_limit: Tuple[float, float] = (-0.3, -0.0),
        contrast_limit: Tuple[float, float] = (-0.5, 0.3),
        noise_p: float = 0.5,
    ) -> None:
        self.gamma_p = gamma_p
        self.gamma_limit = gamma_limit
        self.saturation_p = saturation_p
        self.val_shift_limit = val_shift_limit
        self.brightness_p = brightness_p
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.noise_p = noise_p

        self.transform = A.Compose(
            [
                A.RandomGamma(p=self.gamma_p, gamma_limit=self.gamma_limit),
                A.RandomBrightnessContrast(
                    p=self.brightness_p,
                    brightness_limit=self.brightness_limit,
                    contrast_limit=self.contrast_limit,
                ),
                A.HueSaturationValue(
                    p=self.saturation_p, val_shift_limit=self.val_shift_limit
                ),
                A.GaussNoise(p=self.noise_p),
            ]
        )

    def __call__(self, img: np.ndarray) -> Any:
        # img = self.transform(image=img)['image']
        return img


if __name__ == "__main__":
    pass
