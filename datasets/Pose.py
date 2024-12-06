import os

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


class PoseInterpolator:
    def __init__(self, pose_data: np.ndarray, mode="linear"):
        """
        :param pose_data: Nx7 numpy array with [t, x, y, z, qx, qy, qz, qw] as the row format
        """
        self.pose_data = pose_data
        self.x_interp = interp1d(
            pose_data[:, 0], pose_data[:, 1], kind=mode, bounds_error=True
        )
        self.y_interp = interp1d(
            pose_data[:, 0], pose_data[:, 2], kind=mode, bounds_error=True
        )
        self.z_interp = interp1d(
            pose_data[:, 0], pose_data[:, 3], kind=mode, bounds_error=True
        )
        self.rot_interp = Slerp(pose_data[:, 0], Rotation.from_quat(pose_data[:, 4:]))
        #
        # self.qx_interp = interp1d(pose_data[:, 0], pose_data[:, 4], kind='linear')
        # self.qy_interp = interp1d(pose_data[:, 0], pose_data[:, 5], kind='linear')
        # self.qz_interp = interp1d(pose_data[:, 0], pose_data[:, 6], kind='linear')
        # self.qw_interp = interp1d(pose_data[:, 0], pose_data[:, 7], kind='linear')

    def interpolate(self, t):
        """
        Interpolate 6-dof pose from the initial pose data
        :param t: Query time at which to interpolate the pose
        :return: 4x4 Transformation matrix T_j_W
        """
        if t < np.min(self.pose_data[:, 0]) or t > np.max(self.pose_data[:, 0]):
            print(
                f"Query time is {t}, but time range in pose data is [{np.min(self.pose_data[:, 0])}, {np.max(self.pose_data[:, 0])}]"
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
        if t < np.min(self.pose_data[:, 0]) or t > np.max(self.pose_data[:, 0]):
            print(
                f"Query time is {t}, but time range in pose data is [{np.min(self.pose_data[:, 0])}, {np.max(self.pose_data[:, 0])}]"
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


class TrackTriangulator:
    def __init__(
        self,
        track_data: np.ndarray,
        pose_interpolator: PoseInterpolator,
        t_init: float,
        camera_matrix: np.ndarray,
        depths=None,
    ):
        self.camera_matrix = camera_matrix
        self.camera_matrix_inv = np.linalg.inv(camera_matrix)
        self.T_init_W = pose_interpolator.interpolate(t_init)
        self.T_W_init = np.linalg.inv(self.T_init_W)

        if isinstance(depths, type(None)):
            # Triangulate the points
            self.eigenvalues = []
            self.corners_3D_homo = []
            for idx_track in np.unique(track_data[:, 0]):
                track_data_curr = track_data[track_data[:, 0] == idx_track, 1:]
                n_obs = track_data_curr.shape[0]
                if n_obs < 10:
                    print(f"Warning: not very many observations for triangulation")

                # Construct A
                A = []
                for idx_obs in range(n_obs):
                    corner = track_data_curr[idx_obs, 1:]

                    t = track_data_curr[idx_obs, 0]
                    T_j_W = pose_interpolator.interpolate(t)
                    T_j_init = T_j_W @ np.linalg.inv(self.T_init_W)

                    P = self.camera_matrix @ T_j_init[:3, :]
                    A.append(corner[0] * P[2, :] - P[0, :])
                    A.append(corner[1] * P[2, :] - P[1, :])

                A = np.array(A)
                _, s, vh = np.linalg.svd(A)
                X = vh[-1, :].reshape((-1))
                X /= X[-1]
                self.corners_3D_homo.append(X.reshape((1, 4)))
                self.eigenvalues.append(s[-1])
            self.corners_3D_homo = np.concatenate(self.corners_3D_homo, axis=0)
        else:
            # Back-project using the depths
            self.corners_3D_homo = []
            for idx_track in np.unique(track_data[:, 0]):
                corner_coords = track_data[track_data[:, 0] == idx_track, 1:].reshape(
                    (-1,)
                )
                corner_depth = float(depths[depths[:, 0] == idx_track, 1])
                assert (
                    len(corner_coords) == 2
                ), "Backprojection using depths only supports corner set as input"
                xy_homo = np.array([corner_coords[0], corner_coords[1], 1]).reshape(
                    (3, 1)
                )
                ray_backproj = self.camera_matrix_inv @ xy_homo
                xyz = ray_backproj * corner_depth
                X = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1]).reshape(
                    (1, 4)
                )
                # X = self.T_W_init @ X.T
                self.corners_3D_homo.append(X.reshape((1, 4)))
            self.corners_3D_homo = np.concatenate(self.corners_3D_homo, axis=0)
        self.n_corners = self.corners_3D_homo.shape[0]

    def get_corners(self, T_j_init: np.ndarray):
        """
        Determine the 2D position of the features from the initial extraction step
        :param T_j_init
        :return:
        """
        corners_3D = (T_j_init @ self.corners_3D_homo.T).T
        corners_3D = corners_3D[:, :3]
        corners_2D_proj = (self.camera_matrix @ corners_3D.T).T
        corners_2D_proj = corners_2D_proj / corners_2D_proj[:, 2].reshape((-1, 1))
        corners_2D_proj = corners_2D_proj[:, :2]
        return corners_2D_proj

    def get_depths(self, T_j_init: np.ndarray):
        """
        Determine the 2D position of the features from the initial extraction step
        :param T_j_init
        :return:
        """
        corners_3D = (T_j_init @ self.corners_3D_homo.T).T
        corners_3D = corners_3D[:, :3]
        return corners_3D[:, 2]


if __name__ == "__main__":
    pass
