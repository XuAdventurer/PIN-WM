import os
from argparse import  Namespace
import torch
from functools import partial
import numpy as np
import trimesh
import random
import json
import copy


from diff_rendering.gaussian_splatting_2d.gaussian_renderer import GaussianModel
from diff_rendering.gaussian_splatting_2d.train_2dgs import train_static_2dgs
from diff_rendering.gaussian_splatting_2d.utils.system_utils import searchForMaxIteration
import diff_rendering.gaussian_splatting_2d.gaussian_renderer as Gaussian_Renderer

from diff_simulation.body.body_mesh import Body_Mesh
from diff_simulation.force.constant_force import Constant_Force
from diff_simulation.simulator import Simulator

from utils.cfg_utils import get_gaussian_args

import gym


class Builder():
    def __init__(self,all_args):
        self.data_args = Namespace(**all_args['data_args'])
        self.render_args = Namespace(**all_args['render_args'])
        self.sim_args = Namespace(**all_args['sim_args'])
        self.sys_args = Namespace(**all_args['sys_args'])
        self.gaussian_args = get_gaussian_args(self.data_args,self.render_args,self.sys_args)
        self.set_seed(self.sys_args.seed)
    
    def set_seed(self,seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def load_gaussian(self,gaussian_path):
        gaussians = GaussianModel(self.gaussian_args.dataset.sh_degree)
        gaussians.load_ply(gaussian_path)   
        return gaussians     

    def build_static_2dgs(self):
        loaded_iter = searchForMaxIteration(os.path.join(self.gaussian_args.dataset.model_path, "point_cloud"))
        gaussian_path = os.path.join(self.gaussian_args.dataset.model_path,
                                                "point_cloud",
                                                "iteration_" + str(loaded_iter),
                                                "point_cloud.ply")
        gaussians = self.load_gaussian(gaussian_path)
        return gaussians
    
    def build_renderer(self):
        bg_color = [1.0, 1.0, 1.0] if self.gaussian_args.dataset.white_background else [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color,device="cuda")
        Gaussian_Renderer.render = partial(Gaussian_Renderer.render, pipe=self.gaussian_args.pipe, bg_color=background)
        return Gaussian_Renderer

    def build_optimizer(self,simulator:Simulator):
        l = []
        all_physical_materials = simulator.get_all_physical_materials()
        for physical_materials in all_physical_materials:
            # if physical_materials.requires_grad:
                for key,value in physical_materials.all.items():
                    if value.requires_grad is not True:
                        continue

                    if key == "inertia":
                        lr = 2e-3
                    else:
                        lr = 2e-1
                    l_item = {'params':value,"name":key,'lr':lr}
                    l.append(l_item)
        optimizer = torch.optim.Adam(l)
        return optimizer
