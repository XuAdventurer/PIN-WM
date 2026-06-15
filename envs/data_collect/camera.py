import math

import cv2
import numpy as np
import pybullet as p


class Pybullet_Camera:
    def __init__(self, cam_pos, cam_tar, cam_up_vector, near, far, size, fov):
        self.width, self.height = size
        self.near = near
        self.far = far
        self.fov = fov

        self.cam_pos = cam_pos
        self.cam_tar = cam_tar
        self.cam_up_vector = cam_up_vector
        self.aspect = self.width / self.height
        self.update_matrix()

    def update_matrix(self):
        self.view_matrix = p.computeViewMatrix(self.cam_pos, self.cam_tar, self.cam_up_vector)
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov,
            self.aspect,
            self.near,
            self.far,
        )
        view_matrix = np.array(self.view_matrix).reshape((4, 4), order="F")
        projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order="F")
        self.tran_pix_world = np.linalg.inv(projection_matrix @ view_matrix)

    def get_c2w(self):
        w2c = np.array(self.view_matrix).reshape((4, 4), order="F")
        c2w = np.linalg.inv(w2c)
        return c2w[:3, :4]

    def set_cam_pos(self, cam_pos):
        self.cam_pos = cam_pos
        self.update_matrix()

    def shot(self, cam_id, frame_id, frame_dt):
        _, _, rgba, _, _ = p.getCameraImage(
            self.width,
            self.height,
            self.view_matrix,
            self.projection_matrix,
        )
        image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        image_name = f"r_{cam_id}_{frame_id}.png"
        view_matrix = np.array(
            [
                [self.view_matrix[0], self.view_matrix[1], self.view_matrix[2], self.view_matrix[3]],
                [self.view_matrix[4], self.view_matrix[5], self.view_matrix[6], self.view_matrix[7]],
                [self.view_matrix[8], self.view_matrix[9], self.view_matrix[10], self.view_matrix[11]],
                [self.view_matrix[12], self.view_matrix[13], self.view_matrix[14], self.view_matrix[15]],
            ]
        ).T
        c2w = np.linalg.inv(view_matrix)[:3, :4]
        focal = 0.5 * self.width / np.tan(0.5 * math.radians(self.fov))
        intrinsic = np.array(
            [
                [focal, 0, self.width * 0.5],
                [0, focal, self.height * 0.5],
                [0, 0, 1],
            ]
        )
        image_data = {
            "image_name": image_name,
            "time": frame_dt * frame_id,
            "c2w": c2w.tolist(),
            "intrinsic": intrinsic.tolist(),
        }
        return image, image_data

    def render_rgb(self, shadow=0):
        _, _, rgba, depth, _ = p.getCameraImage(
            self.width,
            self.height,
            self.view_matrix,
            self.projection_matrix,
            shadow=shadow,
            flags=p.ER_NO_SEGMENTATION_MASK,
        )
        image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        return image, depth
