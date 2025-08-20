#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from torch import nn
import os
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
import json
import trimesh
import copy
import open3d as o3d

from diff_rendering.gaussian_splatting_2d.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from diff_rendering.gaussian_splatting_2d.utils.system_utils import mkdir_p
from diff_rendering.gaussian_splatting_2d.utils.sh_utils import RGB2SH,C0,C1,C2,C3
from diff_rendering.gaussian_splatting_2d.utils.graphics_utils import BasicPointCloud
from diff_rendering.gaussian_splatting_2d.utils.general_utils import strip_symmetric, build_scaling_rotation

from utils.quaternion_utils import rotate_point_by_quaternion,quaternion_multiply
from diff_rendering.gaussian_splatting_2d.scene.gaussian_model import GaussianModel
from diff_rendering.gaussian_splatting_2d.utils.sh_utils import SH2RGB

class MeshGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, mesh:trimesh.base.Trimesh):
        super().__init__(sh_degree)
        self.mesh = mesh
        

    def create_pcd_from_mesh(self,num_pts):
        xyz,face_index = self.mesh.sample(num_pts,return_index=True)
        normals = self.mesh.face_normals[face_index]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=normals)
        return pcd
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        # import pdb;pdb.set_trace()
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        # rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        rots = self.normal2quaternion(pcd.normals).float().cuda()

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud,requires_grad = True)
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous(),requires_grad = True)
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous(),requires_grad = True)
        self._scaling = nn.Parameter(scales,requires_grad = True)
        self._rotation = nn.Parameter(rots,requires_grad = True)
        self._opacity = nn.Parameter(opacities,requires_grad = True)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def normal2quaternion(self,normals):
        normals = torch.tensor(normals)
        normals = torch.nn.functional.normalize(normals)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=normals.device,dtype=normals.dtype).expand_as(normals)
        rotation_axis = torch.nn.functional.normalize(torch.cross(z_axis, normals))
        cos_theta = torch.clamp(torch.sum(z_axis * normals, dim=1), -1.0, 1.0)
        theta = torch.acos(cos_theta)
        # Handle the special case where the normal is already aligned with z-axis
        aligned_mask = (rotation_axis.norm(dim=1) < 1e-6)
        rotation_axis[aligned_mask] = torch.tensor([1.0, 0.0, 0.0], device=normals.device, dtype=normals.dtype)
        
        # Normalize the rotation axes
        rotation_axis = rotation_axis / rotation_axis.norm(dim=1, keepdim=True)
        
        # Compute quaternion components
        w = torch.cos(theta / 2)
        xyz = rotation_axis * torch.sin(theta / 2).unsqueeze(-1)
        
        # Concatenate quaternion components [w, x, y, z]
        quaternions = torch.cat([w.unsqueeze(-1), xyz], dim=1)
        return quaternions  

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            requires_grad = group['params'][0].requires_grad
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0),requires_grad=requires_grad)
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0),requires_grad=requires_grad)
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            requires_grad = group["params"][0].requires_grad
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(requires_grad)),requires_grad=requires_grad)
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(requires_grad),requires_grad=requires_grad)
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                requires_grad = group["params"][0].requires_grad
                if requires_grad:
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(requires_grad),requires_grad=requires_grad)
                if requires_grad:
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors