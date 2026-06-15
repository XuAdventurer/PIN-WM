import time
import random

import gym
import numpy as np
import pybullet as p

from utils.quaternion_utils import quaternion_standardize_np


class Base_Env(gym.Env):
    def __init__(self, sim_args):
        self.vis = sim_args.vis
        self.dtime = sim_args.dtime
        self.frame_dt = sim_args.frame_dt
        self.n_substeps = round(self.frame_dt / self.dtime)
        self.f_max = sim_args.f_max
        self.time = 0.0
        self.sync = sim_args.sync

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]


class Base_Pybullet_Env(Base_Env):
    def __init__(self, sim_args):
        super().__init__(sim_args)
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraYaw=45.0,
            cameraPitch=-50,
            cameraDistance=0.5,
            cameraTargetPosition=[0.0, 0.0, 0.0],
        )
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.dtime)

    def step_simulation(self):
        p.stepSimulation()
        if self.vis and self.sync:
            time.sleep(self.dtime)
        self.time += self.dtime

    def close(self):
        p.disconnect(self.physicsClient)

    def hide(self, body_id):
        joint_num = p.getNumJoints(body_id)
        for link_index in range(-1, joint_num):
            p.changeVisualShape(body_id, link_index, rgbaColor=[1, 1, 1, 0])

    def show(self, body_id, color):
        joint_num = p.getNumJoints(body_id)
        for link_index in range(-1, joint_num):
            p.changeVisualShape(body_id, link_index, rgbaColor=color)

    def change_body_pose(self, body_id, body_position, body_rotation):
        p.resetBasePositionAndOrientation(body_id, body_position, body_rotation)

    def add_body(self, body_position, body_rotation, body_path, tex_path=None):
        body_id = p.loadURDF(body_path, body_position, body_rotation)
        if tex_path is not None:
            joint_num = p.getNumJoints(body_id)
            tex_uid = p.loadTexture(tex_path)
            for link_index in range(-1, joint_num):
                p.changeVisualShape(body_id, link_index, textureUniqueId=tex_uid)
        return body_id

    def remove_body(self, body_id):
        p.removeBody(body_id)

    def get_body_pose(self, body_id):
        body_position, body_rotation = p.getBasePositionAndOrientation(body_id)
        return (
            np.array(body_position, dtype=np.float32),
            quaternion_standardize_np(np.array(body_rotation, dtype=np.float32)),
        )

    def change_body_vel(self, body_id, linear_velocity=None, angular_velocity=None):
        if linear_velocity is None:
            linear_velocity, _ = p.getBaseVelocity(body_id)
        if angular_velocity is None:
            _, angular_velocity = p.getBaseVelocity(body_id)
        p.resetBaseVelocity(body_id, linear_velocity, angular_velocity)
