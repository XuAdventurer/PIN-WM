import torch
from typing import List
import trimesh
import numpy as np

from diff_simulation.solver.lcp_solver import LCP_Solver
from diff_simulation.body.base import Body
from diff_simulation.constraints.base import Constraint
from diff_simulation.physical_material import Physical_Materials

import pybullet as p


class Simulator(object):
    """A (differentiable) physics simulation world, with rigid bodies. """

    def __init__(self,dtime,device,vis):
        self.device = device
        self.dtime = dtime
        self.cur_time = 0.0
        self.solver = LCP_Solver(self)
        self.velocity_dim = 6
        self.fric_dirs = 8
        
        self.bodies: List[Body] = []
        self.joints: List[Constraint] = []
        self.body_id_counter = 0
        self.joint_id_counter = 0

        # self.space = ode.HashSpace()
        self.physicsClient = p.connect(p.GUI if vis else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraYaw=45.0,cameraPitch=-30,cameraDistance=0.5,cameraTargetPosition=[0.0,0.0,0.0])

    def add_body(self,body:Body):
        body_id = self.body_id_counter
        self.body_id_counter += 1
        body.set_id(body_id) 
        self.bodies.append(body)
        return body_id

    def add_joint(self,joint):
        joint_id = self.joint_id_counter
        self.joint_id_counter += 1
        joint.set_id(joint_id)
        self.joints.append(joint)
        return joint_id
    
    def get_body(self,body_id):
        for body in self.bodies:
            if body.id == body_id:
                return body
        return None
    
    def get_body_from_pybullet_id(self,pybullet_id):
        for body in self.bodies:
            if body.pybullet_geom_id == pybullet_id:
                return body
        return None

    def get_body_list_index(self,body_id):
        for index,body in enumerate(self.bodies):
            if body.id == body_id:
                return index
        return None
    
    def get_joint(self,joint_id):
        for joint in self.joints:
            if joint.id == joint_id:
                return joint
        return None        
    
    def get_joint_list_index(self,joint_id):
        for index,joint in enumerate(self.joints):
            if joint.id == joint_id:
                return index
        return None
    

    def collision_detection_pybullet(self):
        for body in self.bodies:
            body.update_collision_geom()
        p.performCollisionDetection()
        pybullet_contact_infos = p.getContactPoints()
        contact_infos = []
        vis_contact_points =[]
        for pybullet_contact_info in pybullet_contact_infos:
            vis_contact_points.append(pybullet_contact_info[5]+pybullet_contact_info[7])
            body_a = self.get_body_from_pybullet_id(pybullet_contact_info[1])
            body_b = self.get_body_from_pybullet_id(pybullet_contact_info[2])
            normal = torch.tensor(pybullet_contact_info[7],device=self.device)
            p_a = torch.tensor(pybullet_contact_info[5],device=self.device) - body_a.world_position
            p_b = torch.tensor(pybullet_contact_info[6],device=self.device) - body_b.world_position
            penetration = torch.tensor(pybullet_contact_info[8],device=self.device)

            contact_infos.append(((normal, p_a[:3], p_b[:3], penetration),
                                    body_a.id, body_b.id))
        return contact_infos,vis_contact_points
    
    def collision_detection_plane(self):
        contact_infos = []
        plane_world_position = self.bodies[0].world_position
        plane_id = 0 
        normal = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        penetration = torch.tensor([0.0], device=self.device, dtype=torch.float32)
        for body in self.bodies[1:]:
            vertices_world = body.get_vertices_world()
            index = vertices_world[...,-1] <(plane_world_position[2] + 5.0 + 0.0001).detach().cpu().numpy()
            contact_points = torch.tensor(vertices_world[index],dtype=torch.float32,device=self.device)

            if contact_points.any():
                p_a_batch = contact_points - plane_world_position
                p_b_batch = contact_points - body.world_position    
                contact_infos += [((normal,p_a,p_b,penetration),plane_id,body.id) for p_a,p_b in zip(p_a_batch,p_b_batch)]

        return contact_infos



            

    def get_sum_constraint_dim(self):
        sum_constraint_dim = 0
        for joint in self.joints:
            sum_constraint_dim += joint.constraint_dim
        return sum_constraint_dim
    
    def apply_external_forces(self, cur_time):
        return torch.cat([b.apply_external_forces(cur_time) for b in self.bodies])
    
    def get_vel_vec(self):
        return torch.cat([b.get_vel_vec() for b in self.bodies])

    def integrate_transform(self,new_velocitys):
        from utils.quaternion_utils import multiply,normalize
        for index,body in enumerate(self.bodies):
            new_velocity = new_velocitys[index * self.velocity_dim : index * self.velocity_dim + self.velocity_dim]
            body.angular_velocity = new_velocity[:3]
            body.linear_velocity = new_velocity[3:]
            body.world_position = body.world_position + body.linear_velocity * self.dtime
            body.world_rotation = body.world_rotation + multiply(torch.cat([torch.tensor(([0]),device = self.device),
                                                                            0.5*self.dtime*body.angular_velocity],dim=0),body.world_rotation)
            body.world_rotation = normalize(body.world_rotation)

    def vis_scene(self,vis_contact_points):
        scene = trimesh.Scene()
        vis_contact_points = np.array(vis_contact_points)
        contact_point_positions = vis_contact_points[...,:3]
        contact_point_normals = vis_contact_points[...,3:]
        contact_points = trimesh.PointCloud(contact_point_positions)
        scene.add_geometry(contact_points)
        lines = []
        for contact_point,contact_point_normal in zip(contact_point_positions,contact_point_normals):
            start_point = contact_point
            end_point = start_point + contact_point_normal * 0.003 
            lines.append([start_point, end_point])
        normal_lines = trimesh.load_path(lines)
        scene.add_geometry(normal_lines)
        for body in self.bodies[1:] :
            vis_geom = body.get_visual_geom_world()
            scene.add_geometry(vis_geom)
        scene.show()

    def get_body_pose(self,body_id):
        body = self.get_body(body_id)
        return body.get_pose()

    def get_body_pose_clone(self,body_id):
        body = self.get_body(body_id)
        return body.get_pose_clone()
    
    def get_body_pose_cpu(self,body_id):
        body = self.get_body(body_id)
        return body.get_pose_cpu()
    
    def get_body_pose_np(self,body_id):
        body = self.get_body(body_id)
        return body.get_pose_np()
    
    def get_body_vel_np(self,body_id):
        body = self.get_body(body_id)
        return body.get_body_vel_np()
        

    def change_body_pose(self,body_id,world_position=None,world_rotation=None):
        body = self.get_body(body_id)
        body.change_pose(world_position,world_rotation)

    def change_body_vel(self,body_id,linear_velocity=None,angular_velocity=None):
        body = self.get_body(body_id)
        body.change_vel(linear_velocity,angular_velocity)
        
    def create_mesh_body(self,mesh,physical_materials,urdf,requires_grad,world_position=None,world_rotation=None):
        from diff_simulation.body.body_mesh import Body_Mesh
        body = Body_Mesh(mesh,physical_materials,urdf,requires_grad,self.device,world_position,world_rotation)
        body_id = self.add_body(body)
        return body_id


    def create_joint(self,body_id,joint_type):
        from diff_simulation.constraints.base import Joint_Type
        if joint_type == Joint_Type.FIX_CONSTRAINT:
            from diff_simulation.constraints.fix_constraint import Fix_Constraint
            constraint = Fix_Constraint(body_id)
        elif joint_type == Joint_Type.NO_ROT_CONSTRATNT:
            from diff_simulation.constraints.rot_constraint import Rot_Constraint
            constraint = Rot_Constraint(body_id)
        elif joint_type == Joint_Type.NO_TRANS_Z_CONSTRATNT:
            from diff_simulation.constraints.trans_constraint import TransZ_Constraint
            constraint = TransZ_Constraint(body_id)
        joint_id = self.add_joint(constraint)
        return joint_id
            
    def get_body_physical_materials(self,body_id):
        body = self.get_body(body_id)
        return body.get_physical_materials()
    
    def get_all_physical_materials(self):
        all_physical_materials = []
        for body in self.bodies:
            all_physical_materials.append(body.get_physical_materials())
        return all_physical_materials

    def load_all_physical_materials(self,json_path):
        import json
        with open(json_path, 'r') as json_file:
            all_physical_materials_original_json = json.load(json_file)["activate"]
        for physical_materials_original in all_physical_materials_original_json:
            body = self.get_body(physical_materials_original["body_id"])
            physical_materials = body.physical_materials
            for key,value in physical_materials_original.items():
                if key in physical_materials.all:
                    physical_materials.set_material(key,value)
            body.set_physical_materials(physical_materials)

    def set_physical_materials(self,body_id,physical_materials):
        body = self.get_body(body_id)
        return body.set_physical_materials(physical_materials)

    def step(self):
        contact_infos1,vis_contact_points = self.collision_detection_pybullet()
        contact_infos2 = self.collision_detection_plane()
        contact_infos = contact_infos1 + contact_infos2

        x = self.solver.solve_constraint(contact_infos)
        new_velocitys = x[:self.velocity_dim * len(self.bodies)].squeeze(0)
        self.integrate_transform(new_velocitys)
        self.cur_time = self.cur_time + self.dtime

    def reset(self):
        self.cur_time = 0.0
        for body in self.bodies:
            body.reset()

    def close(self):
        p.disconnect(self.physicsClient)