from .base import Body
import torch
import trimesh
import numpy as np
import copy

from diff_simulation.physical_material import Physical_Materials

import pybullet as p
class Body_Mesh(Body):


    def __init__(self,mesh:trimesh.base.Trimesh,physical_materials:Physical_Materials,urdf,
                 requires_grad,device,world_position=None,world_rotation=None):
        from utils.mesh_utils import cal_MassProperties,get_ang_inertia,inertia_diagonalize
        volume,_,inertia_body_unit = cal_MassProperties(mesh,device)

        if world_position is None:
            world_position = torch.nn.Parameter(torch.tensor([0.0,0.0,0.0],device=device),requires_grad)
        else:
            world_position = torch.nn.Parameter(world_position,requires_grad)

        if world_rotation is None:
            world_rotation = torch.nn.Parameter(torch.tensor([1.0,0.0,0.0,0.0],device=device),requires_grad)
        else:
            world_rotation = torch.nn.Parameter(world_rotation,requires_grad)
            
        trimesh_mesh_local = mesh
        
        super().__init__(None,trimesh_mesh_local,physical_materials,urdf,
                         world_position,world_rotation,device)

    def sample_face(self,mesh):
        # use for kaolin Mesh
        face_vertices = mesh.face_vertices.cuda() #(face_num,3,3)
        sample_num = 100
        sample_points = self.random_point_on_triangle(face_vertices,sample_num) #(face_num,sample_num,3)
        return sample_points


    def get_visual_geom_world(self):
        mesh_copy = copy.deepcopy(self.visual_geom)
        from utils.mesh_utils import cal_transform_matrix
        from utils.quaternion_utils import wxyz2xyzw
        obj_position,obj_rotation = self.get_pose_cpu()
        rotation = wxyz2xyzw(obj_rotation).numpy()
        position = obj_position.numpy()
        transform_matrix = cal_transform_matrix([1,1,1],
                                                rotation,
                                                position)
        mesh_copy.apply_transform(transform_matrix)
        return mesh_copy

    def get_vertices_world(self):
        vertices_loacl = self.visual_geom.vertices
        from utils.mesh_utils import cal_transform_matrix
        from utils.quaternion_utils import wxyz2xyzw
        obj_position,obj_rotation = self.get_pose_cpu()
        rotation = wxyz2xyzw(obj_rotation).numpy()
        position = obj_position.numpy()
        transform_matrix = cal_transform_matrix([1,1,1],
                                                rotation,
                                                position)
        # mesh_copy.apply_transform(transform_matrix)
        vertices_h = np.hstack((vertices_loacl, np.ones((vertices_loacl.shape[0], 1))))  # 齐次坐标
        vertices_world = (transform_matrix @ vertices_h.T).T[:, :3]  # 更新顶点位置
        return vertices_world
    
    def get_visual_geom_local(self):
        return copy.deepcopy(self.visual_geom)

    def update_collision_geom(self):
        from utils.quaternion_utils import wxyz2xyzw
        if self.pybullet_geom_id is not None:
            p.resetBasePositionAndOrientation(self.pybullet_geom_id,
                                            self.world_position.detach().cpu().numpy(),wxyz2xyzw(self.world_rotation).detach().cpu().numpy())
        