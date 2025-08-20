from abc import ABCMeta, abstractmethod
import torch
import copy

from diff_simulation.force.constant_force import Constant_Force
from diff_simulation.physical_material import Physical_Materials
from utils.quaternion_utils import multiply,rotate_point_by_quaternion,quaternion_to_rotmat
from utils.quaternion_utils import wxyz2xyzw

import pybullet as p

class Body(metaclass=ABCMeta):
    def __init__(self,collision_geom,visual_geom,physical_materials:Physical_Materials,
                 urdf,world_position,world_rotation,device) :
        self.id = None
        self.collision_geom = collision_geom # for ode collision detection
        if urdf is not None:
            self.urdf = urdf
            self.pybullet_geom_id = p.loadURDF(urdf,world_position.detach().cpu().numpy(),wxyz2xyzw(world_rotation).detach().cpu().numpy())
        else:
            self.pybullet_geom_id = None
        self.visual_geom = visual_geom
        self.device = device
        self.physical_materials = physical_materials
        self.mass = self.physical_materials.get_material("mass")
        self.friction_coefficient = self.physical_materials.get_material("friction_coefficient")
        self.restitution = self.physical_materials.get_material("restitution")
        self.inertia_body = self.physical_materials.get_material("inertia") * self.mass 

        
        self.world_position = world_position
        # self.position = torch.mean(vertices_xyz_world, dim = 0)
        self.world_rotation = world_rotation
        # self.init_world_position = copy.deepcop·y(self.world_position)
        # self.init_world_rotation = copy.deepcopy(self.world_rotation)
        self.linear_momentum = torch.zeros((3),device=self.device)
        self.angular_momentum = torch.zeros((3),device=self.device)
        self.linear_velocity = torch.zeros((3),device=self.device)
        self.angular_velocity = torch.zeros((3),device=self.device)

        self.forces = []
        self.apply_positions = []
        self.add_gravity()

        # for reset
        self.init_world_position = self.world_position.clone()
        self.init_world_rotation = self.world_rotation.clone()
        

    def get_physical_materials(self):
        return self.physical_materials
    
    def set_physical_materials(self,physical_materials):
        self.physical_materials = physical_materials
        self.mass = self.physical_materials.get_material("mass")
        # self.inertia_body = torch.diag(torch.cat((self.physical_materials.get_material("inertia_x_unit").unsqueeze(0),
        #                                 self.physical_materials.get_material("inertia_y_unit").unsqueeze(0),
        #                                 self.physical_materials.get_material("inertia_z_unit").unsqueeze(0)),dim=0)) * self.mass 
        self.inertia_body = self.physical_materials.get_material("inertia") * self.mass 
        # self.inertia_body_inv = 1.0 / self.inertia_body
        self.friction_coefficient = self.physical_materials.get_material("friction_coefficient")
        self.restitution = self.physical_materials.get_material("restitution")

    def set_id(self,id):
        self.id = id
        self.physical_materials.body_id = self.id
        
    def reset(self):
        self.mass = self.physical_materials.get_material("mass")
        self.friction_coefficient = self.physical_materials.get_material("friction_coefficient")
        self.restitution = self.physical_materials.get_material("restitution")        
        self.inertia_body = self.physical_materials.get_material("inertia") * self.mass 

        self.world_position = self.init_world_position.clone()
        self.world_rotation = self.init_world_rotation.clone()
        self.linear_momentum = torch.zeros((3),device=self.device)
        self.angular_momentum = torch.zeros((3),device=self.device)
        self.linear_velocity = torch.zeros((3),device=self.device)
        self.angular_velocity = torch.zeros((3),device=self.device)
        self.forces = []
        self.apply_positions = []
        self.add_gravity()

    def add_gravity(self):
        gravity = Constant_Force(
            magnitude = 9.8 * self.mass,
            direction = torch.tensor(([0.0, 0.0, -1.0]),device=self.device),
        )
        self.add_external_force(gravity)

    def get_pose_clone(self):
        return self.world_position.clone(),self.world_rotation.clone()

    def get_pose_cpu(self):
        return self.world_position.detach().cpu(),self.world_rotation.detach().cpu()

    def get_pose_np(self):
        return self.world_position.detach().cpu().numpy(),self.world_rotation.detach().cpu().numpy()
    
    def get_pose(self):
        return self.world_position,self.world_rotation
    
    def get_body_vel_np(self):
        return self.linear_velocity.detach().cpu().numpy(),self.angular_velocity.detach().cpu().numpy()
    
    def change_pose(self,world_position=None,world_rotation=None):
        if world_position is not None:
            self.world_position = world_position
        if world_rotation is not None:
            self.world_rotation = world_rotation
        self.update_collision_geom()

    def change_vel(self,linear_velocity=None,angular_velocity=None):
        if linear_velocity is not None:
            self.linear_velocity = linear_velocity
        if angular_velocity is not None:
            self.angular_velocity = angular_velocity

    def compute_inertia_world_inv(self):
        rotmat = quaternion_to_rotmat(self.world_rotation)
        return torch.matmul(torch.matmul(rotmat, self.inertia_body_inv), rotmat.transpose(0, 1))
    
    def compute_inertia_world(self):
        rotmat = quaternion_to_rotmat(self.world_rotation)
        # inertia_body = torch.diag(self.inertia_body)
        return torch.matmul(torch.matmul(rotmat, self.inertia_body), rotmat.transpose(0, 1))
    
    def compute_angular_velocity_from_angular_momentum(inertia_world_inv, angular_momentum,):    
        return torch.matmul(inertia_world_inv, angular_momentum.view(-1, 1)).squeeze(-1)
    
    def compute_linear_velocity_from_linear_momentum(linear_momentum, mass,):
        return linear_momentum / mass
    
    def compute_center_of_mass(vertices, masses):
        return (masses.view(-1, 1) * vertices).sum(0) / masses.sum()


    def compute_state_derivatives(self, cur_time):
        dposition = self.linear_velocity
        angular_velocity_quat = torch.zeros((4),device = self.device)
        angular_velocity_quat[1:] = self.angular_velocity
        drotation = 0.5 * multiply(angular_velocity_quat, self.world_rotation)
        dlinear_momentum, dangular_momentum = self.apply_external_forces(cur_time)
        return dposition, drotation, dlinear_momentum, dangular_momentum
    
    def add_external_force(self, force, apply_pos=None):
        self.forces.append(force)
        self.apply_positions.append(apply_pos)

    def clear_force(self):
        self.forces = []
        self.apply_positions =[]

    def apply_external_forces(self, cur_time):
        total_force = torch.zeros((3),device=self.device)
        total_torque = torch.zeros((3),device=self.device)
        for force, apply_pos in zip(self.forces, self.apply_positions):
            force_vector = force.apply(cur_time) 
            total_force += force_vector
            if apply_pos is not None:
                torque_vector = torch.cross(apply_pos-self.world_position, force_vector)
                total_torque += torque_vector
        return torch.cat([total_torque, total_force])
    
    def apply_impulse(self,contact_pos,j):
        self.linear_momentum = self.linear_momentum + j
        self.angular_momentum = self.angular_momentum + torch.cross(contact_pos - self.world_position, j)
        self.linear_velocity = self.linear_momentum / self.mass
        self.angular_velocity = torch.matmul(self.compute_inertia_world_inv(), self.angular_momentum)    
    
    # inertia-mass
    def get_M_world(self):
        M = torch.zeros((6, 6),device=self.device)
        M[:3,:3] = self.compute_inertia_world()
        M[3:,3:] = torch.eye((3),device=self.device) * self.mass
        return M
    
    # [angular_velocity,linear_velocity]
    def get_vel_vec(self):
        v = torch.cat((self.angular_velocity,self.linear_velocity))
        return v
    
    def get_collision_geom_world(self):
        pass
    
    def get_collision_geom_local(self):
        pass
    
    def get_visual_geom_world(self):
        pass
    
    def get_visual_geom_local(self):
        pass

    def update_collision_geom(self):
        pass
