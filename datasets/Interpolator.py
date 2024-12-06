import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


def T_to_Rt(T, batch=False):
    if batch:
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        return R, t
    else:
        R = T[:3, :3]
        t = T[:3, 3]
        return R, t


def Rt_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


class PoseInterpolator:
    def __init__(
        self,
        timestamp: np.ndarray,
        t: np.ndarray,
        R: np.ndarray,
        quat_R=True,
        mode="linear",
    ):
        """
        :param timestamp: [N,] numpy array with timestamps
        :param t: [N, 3] numpy array with translation
        :param R: [N, 3, 3] matricies or [N, 4] quaternions with rotation
        :param quat_R: True if R is in quaternion format
        :param mode: Interpolation mode
        """
        self.timestamp = timestamp

        self.x_interp = interp1d(timestamp, t[:, 0], kind=mode, bounds_error=True)
        self.y_interp = interp1d(timestamp, t[:, 1], kind=mode, bounds_error=True)
        self.z_interp = interp1d(timestamp, t[:, 2], kind=mode, bounds_error=True)

        if quat_R:
            self.rot_interp = Slerp(timestamp, Rotation.from_quat(R))
        else:
            self.rot_interp = Slerp(timestamp, Rotation.from_matrix(R))

    def interpolate(self, t: [float, np.ndarray]):
        """
        Interpolate 6-dof pose from the initial pose data
        :param t: Query time at which to interpolate the pose
        :return: 4x4 Transformation matrix T_j_W
        """
        if t < np.min(self.timestamp) or t > np.max(self.timestamp):
            print(
                f"Query time is {t}, but time range in pose data is [{np.min(self.timestamp)}, {np.max(self.timestamp)}]"
            )
        T_W_j = np.eye(4)
        T_W_j[0, 3] = self.x_interp(t)
        T_W_j[1, 3] = self.y_interp(t)
        T_W_j[2, 3] = self.z_interp(t)
        T_W_j[:3, :3] = self.rot_interp(t).as_matrix()
        return np.linalg.inv(T_W_j)

    def interpolate_colmap(self, t: float):
        """
        Interpolate 6-dof pose from the initial pose data
        :param t: Query time at which to interpolate the pose
        :return:
        """
        if t < np.min(self.timestamp) or t > np.max(self.timestamp):
            print(
                f"Query time is {t}, but time range in pose data is [{np.min(self.timestamp)}, {np.max(self.timestamp)}]"
            )
        T_W_j = np.eye(4)
        T_W_j[0, 3] = self.x_interp(t)
        T_W_j[1, 3] = self.y_interp(t)
        T_W_j[2, 3] = self.z_interp(t)
        T_W_j[:3, :3] = self.rot_interp(t).as_matrix()
        T_j_W = np.linalg.inv(T_W_j)
        quat = Rotation.from_matrix(T_j_W[:3, :3]).as_quat()
        return np.asarray(
            [T_j_W[0, 3], T_j_W[1, 3], T_j_W[2, 3], quat[0], quat[1], quat[2], quat[3]],
            dtype=np.float32,
        )
