import math
import os

import numpy as np
import pybullet as p

from envs.base import Base_Pybullet_Env
from envs.data_collect.camera import Pybullet_Camera
from utils.data_utils import save_data,save_video_data


class Pybullet_Data_Collect_Env(Base_Pybullet_Env):
    def __init__(self, data_args, sim_args):
        super().__init__(sim_args)
        self.n_frames = data_args.n_frames
        self.obj_id = None
        self.obj_init_position = data_args.obj_position
        self.obj_init_rotation = data_args.obj_rotation
        self.obj_path = data_args.obj_path
        self.ee_init_position = data_args.ee_position
        self.ee_init_rotation = data_args.ee_rotation
        self.ee_path = data_args.ee_path
        self.ee_id = None
        self.plane_path = "./envs/asset/plane/plane.urdf"
        self.plane_id = None
        self.tex_path = getattr(data_args, "tex_path", None)
        self.data_path = data_args.data_path
        os.makedirs(self.data_path, exist_ok=True)

        self.camera_surround = Pybullet_Camera(
            (0.0, 0.0, 0.0),
            (0, 0.0, 0),
            (0, 0, 1),
            0.01,
            100,
            (data_args.W, data_args.H),
            45,
        )
        self.camera_out_hand = Pybullet_Camera(
            (0.3, -0.3, 0.4),
            (0, -0.1, 0),
            (0, 0, 1),
            0.01,
            100,
            (data_args.W, data_args.H),
            45,
        )

    def reset(self):
        self.time = 0.0
        p.resetSimulation()
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.dtime)
        self.obj_id = None
        self.plane_id = p.loadURDF(self.plane_path, [0.0, 0.0, -5.0])
        self.ee_id = p.loadURDF(self.ee_path, self.ee_init_position, self.ee_init_rotation)

    def cal_surround_ee_positon_batch(self, surround_shoot_num):
        surround_ee_positon_batch = []
        surround_shoot_num = int(surround_shoot_num)
        d_theta = math.radians(360.0 / surround_shoot_num)
        current_theta = math.radians(90)
        radius = 0.3
        height = 0.4
        for _ in range(surround_shoot_num):
            surround_ee_positon_batch.append(
                [radius * math.cos(current_theta), radius * math.sin(current_theta), height]
            )
            current_theta += d_theta
        return surround_ee_positon_batch

    def camera_surround_shoot(self, frame_id, surround_ee_positon_batch):
        camera_images = []
        camera_image_datas = []
        for index, ee_position in enumerate(surround_ee_positon_batch):
            self.camera_surround.set_cam_pos(ee_position)
            image, image_data = self.camera_surround.shot(index, frame_id, self.frame_dt)
            camera_images.append(image)
            camera_image_datas.append(image_data)
        return camera_images, camera_image_datas

    def step_video_motion(self, ee_goal_position):
        pos, _ = p.getBasePositionAndOrientation(self.ee_id)
        vel = np.array([ee_goal_position[0] - pos[0], ee_goal_position[1] - pos[1], 0.0]) * 5
        p.resetBaseVelocity(self.ee_id, vel, [0, 0, 0])
        for _ in range(self.n_substeps):
            self.step_simulation()

    def video_push_shoot(self, ee_goal_position):
        video_images = []
        video_image_datas = []
        pose_datas = []
        for frame_id in range(0, self.n_frames):
            if frame_id > 0:
                self.step_video_motion(ee_goal_position)
            image, image_data = self.camera_out_hand.shot(0, frame_id, self.frame_dt)
            video_images.append(image)
            video_image_datas.append(image_data)
            obj_position, obj_rotation = p.getBasePositionAndOrientation(self.obj_id)
            pose_datas.append(list(obj_position) + list(obj_rotation))
        return video_images, video_image_datas, pose_datas

    def collect_video_data(self, ee_goal_position):
        self.reset()
        if self.ee_id is not None:
            self.hide(self.ee_id)
        if self.plane_id is not None:
            self.hide(self.plane_id)
        background_image, background_image_data = self.camera_out_hand.shot(0, -1, self.frame_dt)
        self.obj_id = self.add_body(
            self.obj_init_position,
            self.obj_init_rotation,
            self.obj_path,
            self.tex_path,
        )
        video_images, video_image_datas, pose_datas = self.video_push_shoot(ee_goal_position)
        collected_images = video_images + [background_image]
        collected_image_datas = video_image_datas + [background_image_data]
        save_video_data(
            collected_images,
            collected_image_datas,
            np.array(self.ee_init_position),
            np.array(ee_goal_position),
            pose_datas,
            self.data_path,
        )

    def collect_2dgs_data_for_body(
        self,
        body_path,
        body_position,
        body_rotation,
        tex_path,
        surround_shoot_num,
    ):
        self.reset()
        if self.ee_id is not None:
            self.hide(self.ee_id)
        if self.plane_id is not None:
            self.hide(self.plane_id)
        surround_ee_positon_batch = self.cal_surround_ee_positon_batch(surround_shoot_num)
        background_images, background_image_datas = self.camera_surround_shoot(
            -1,
            surround_ee_positon_batch,
        )
        self.obj_id = self.add_body(body_position, body_rotation, body_path, tex_path)
        object_images, object_image_datas = self.camera_surround_shoot(
            0,
            surround_ee_positon_batch,
        )
        collected_images = background_images + object_images
        collected_image_datas = background_image_datas + object_image_datas
        save_data(
            collected_images,
            collected_image_datas,
            np.array([0.0, 0.0]),
            [],
            self.data_path,
            "2dgs_data",
            "2dgs_data.json",
        )
