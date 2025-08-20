import mmcv
import torch
import numpy as np
import random
import os,sys
import json
from tqdm import tqdm
import trimesh
import time


cur_work_path = os.getcwd()
sys.path.append(cur_work_path)

from builder import Builder
from utils.sys_utils import prepare_output_and_logger
from utils.data_utils import load_pybullet_dynamic_dataset
from utils.metric_utils import mean_psnr
from utils.quaternion_utils import wxyz2xyzw,rotmat_to_quaternion

from diff_rendering.gaussian_splatting_2d.utils.camera_utils import cameraList_from_camInfos
from diff_rendering.gaussian_splatting_2d.utils.render_utils import save_img_u8
from diff_rendering.gaussian_splatting_2d.scene.gaussian_model import GaussianModel
from diff_rendering.gaussian_splatting_2d.utils.loss_utils import rendering_loss_batch,pose_loss_batch

from diff_simulation.simulator import Simulator

from diff_simulation.physical_material import Physical_Materials
from diff_simulation.constraints.base import Joint_Type

sys_args = dict(
    seed = 0,
    output_path = './output/sim_push_t'
)

data_args = dict(
    white_background = True,
    n_frames = 32,
    test_num = 0,
    train_num = 1,
    H = 800,
    W = 800,
    data_path = 'dataset/sim_push_t2'
)

render_args = dict(
    sh_degree = 3,

)

sim_args = dict(
    dtime = 1.0 / 240,
    frame_dt = 1.0 / 24,
    train_iteration = 100,
    save_iteration = 10,
    opt_interval_num = 8
)

policy_args = dict()


def train_physical_materials(output_path,builder,obj_id,ee_id,dataset,simulator:Simulator,
                             gaussians:GaussianModel,gaussian_renderer,random_init_index,
                             optimizer,logger=None):
    n_substeps = round(builder.sim_args.frame_dt / builder.sim_args.dtime)
    f_max = builder.data_args.n_frames
    dynamic_train_camera = cameraList_from_camInfos(dataset["cam_infos"], 1.0, builder.gaussian_args.dataset)
    gt_images = []
    for cam in dynamic_train_camera:
        gt_images.append(cam.original_image.cuda())
    gt_poses = torch.tensor(dataset["pose_datas"],device=simulator.device).cuda()
    goal_pos = torch.tensor(dataset["ee_goal_position"],device=simulator.device)
    
    train_iteration = builder.sim_args.train_iteration
    opt_interval_num = builder.sim_args.opt_interval_num
    opt_interval = int(train_iteration / opt_interval_num)
    assert f_max % opt_interval_num == 0
    frame_interval = int(f_max / opt_interval_num)
    start_idx = 0
    max_idx = f_max - frame_interval
    progress_bar = tqdm(range(0, train_iteration), desc="Dynamic Train Progress")
    for iter in range(0,train_iteration + 1):
        simulator.reset()
        images = []
        poses = []
        for f in range(f_max):
            if f > 0:
                ee_world_position,_ = simulator.get_body_pose_clone(ee_id)
                ee_linear_velocity = (goal_pos - ee_world_position[:2]) * 5
                ee_linear_velocity = torch.cat((ee_linear_velocity,torch.tensor([0],device=simulator.device)))
                simulator.change_body_vel(ee_id,linear_velocity=ee_linear_velocity,angular_velocity=torch.zeros((3),device=simulator.device))
                for i in range(n_substeps):
                    simulator.step()

            obj_position,obj_rotation = simulator.get_body_pose(obj_id)
            gaussians.reset_position_rotation(obj_position,obj_rotation)
            render_pkg = gaussian_renderer.render(dynamic_train_camera[0], gaussians)


            image = render_pkg['render'] 
            images.append(image)
            poses.append(torch.cat((obj_position.clone(),wxyz2xyzw(obj_rotation.clone())),dim=-1))

        if iter % opt_interval == 0 and iter > 0:
            start_idx += frame_interval
            if start_idx > max_idx :
                start_idx = max_idx

        loss = rendering_loss_batch(images[:12],gt_images[:12])

        loss.backward(retain_graph=True)
        
        psnr = mean_psnr(images,gt_images)

        all_physical_materials = simulator.get_all_physical_materials()
        if logger is not None:
            logger.record("metric/loss"+ str(random_init_index),loss)
            logger.record("metric/psnr"+ str(random_init_index),psnr)
            logger.dump(iter)


        info_dict = {"loss": loss.item(),"psnr": psnr.item()}
        progress_bar.set_postfix(info_dict)
        progress_bar.update(1)

        if (iter % builder.sim_args.save_iteration) == 0:
            save_path = os.path.join(output_path, "iteration_{}".format(iter))
            os.makedirs(save_path, exist_ok = True)
            image_path = os.path.join(save_path, "images")
            os.makedirs(image_path, exist_ok = True)
            for index,image in enumerate(images):
                save_img_u8(image.permute(1,2,0).cpu().detach().numpy(), os.path.join(image_path, 'sim_{}.png'.format(index)))
            physical_materials_original_json_list = []
            physical_materials_activate_json_list = []
            for physical_materials in all_physical_materials:
                physical_materials_original_json_list.append(physical_materials.get_original_json_dict())
                physical_materials_activate_json_list.append(physical_materials.get_activate_json_dict())
            with open(os.path.join(save_path,'physical_materials_iter.json'), 'w') as json_file:
                json.dump({
                           "activate":physical_materials_activate_json_list}, fp=json_file ,indent=4)

        optimizer.step()
        optimizer.zero_grad()
    progress_bar.close()

    return loss

def create_push_t_scene(builder,simulator:Simulator):

    plane_mesh_path = "./envs/asset/plane/plane_collision.obj"
    obj_mesh_path = "./envs/asset/cube_t/cube_t.obj"
    ee_mesh_path = "./envs/asset/ee/ee.obj"
    plane_urdf_path = "./envs/asset/plane/plane.urdf"
    obj_urdf_path = "./envs/asset/cube_t/cube_t_mesh.urdf"
    ee_urdf_path = "./envs/asset/ee/ee.urdf"

    plane_mesh = trimesh.load(plane_mesh_path)
    plane_mesh.apply_scale([30,30,10])
    plane_physical_material = Physical_Materials(requires_grad = True,device = simulator.device)
    plane_physical_material.no_optimize("mass")
    plane_physical_material.no_optimize('inertia')
    plane_id = simulator.create_mesh_body(plane_mesh,plane_physical_material,requires_grad=True,
                                            urdf = plane_urdf_path,
                                            world_position=torch.tensor([0.0,0.0,-5.0],device=simulator.device),
                                            world_rotation=torch.tensor([1.0,0.0,0.0,0.0],device=simulator.device))

    simulator.create_joint(plane_id,Joint_Type.FIX_CONSTRAINT)
    obj_mesh = trimesh.load(obj_mesh_path)
    obj_physical_material = Physical_Materials(requires_grad = True,device=simulator.device)
    obj_physical_material.set_material("inertia",[
                [
                    0.01,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.01,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    0.01
                ]
            ])   
    obj_id = simulator.create_mesh_body(obj_mesh,obj_physical_material,requires_grad=True,
                                        urdf = obj_urdf_path,
                                        world_position=torch.tensor([0.0,0.0,0.025],device=simulator.device),
                                        world_rotation=torch.tensor([1.0,0.0,0.0,0.0],device=simulator.device)) 

    ee_mesh = trimesh.load(ee_mesh_path)
    ee_physical_material = Physical_Materials(requires_grad = True,device = simulator.device)
    ee_physical_material.set_material("mass",1000.0)
    ee_physical_material.no_optimize("inertia")
    ee_physical_material.no_optimize("mass")
    ee_id = simulator.create_mesh_body(ee_mesh,ee_physical_material,requires_grad=True,
                                        urdf = ee_urdf_path,
                                        world_position=torch.tensor([0.05 ,0.05 ,0.05],device=simulator.device),
                                        world_rotation=torch.tensor([1.0,0.0,0.0,0.0],device=simulator.device))

    simulator.create_joint(ee_id,Joint_Type.NO_TRANS_Z_CONSTRATNT)
    simulator.create_joint(ee_id,Joint_Type.NO_ROT_CONSTRATNT)

    return obj_id,ee_id



def test_dynamic_train_multbody_push_t():
    all_args = {
        'sys_args':sys_args,
        'data_args':data_args,
        'render_args':render_args,
        'sim_args':sim_args,
        'policy_args':policy_args
    }
    all_args,loggers = prepare_output_and_logger(all_args,need_logger=True)
    builder = Builder(all_args)
    
    dynamic_train_datasets,_ = load_pybullet_dynamic_dataset(builder.data_args)

    train_path = os.path.join(builder.sys_args.output_path, "dynamic","train",str(int(time.time())))
    os.makedirs(train_path, exist_ok = True)    

    random_init_num = 1
    for random_init_index in range(random_init_num):
        train_sub_path = os.path.join(train_path, "random_init" + str(random_init_index))
        os.makedirs(train_sub_path, exist_ok = True)
        loss_mean = 0.0
        for dataset_index,dynamic_train_dataset in enumerate(dynamic_train_datasets):
            output_path = os.path.join(train_sub_path, "dataset" + str(dataset_index))
            os.makedirs(output_path, exist_ok = True)
            print("Save folder:",train_path)
            sim_device = "cuda"
            vis = True
            simulator = Simulator(builder.sim_args.dtime,device=sim_device,vis=vis)
            obj_id,ee_id = create_push_t_scene(builder,simulator)
            obj_position,obj_rotation = simulator.get_body_pose_clone(obj_id)
            gaussians = builder.build_static_2dgs()
            gaussians.translate2localframe(obj_position,obj_rotation)
            gaussian_renderer = builder.build_renderer()
            optimizer = builder.build_optimizer(simulator)
            optimizer.param_groups[4]['lr'] = 4e-2
            optimizer.param_groups[5]['lr'] = 1e-4
            loss = train_physical_materials(output_path,builder,obj_id,ee_id,dynamic_train_dataset,
                                    simulator,gaussians,gaussian_renderer,random_init_index,
                                    optimizer,loggers["dynamic"][dataset_index])
            simulator.close()
            loss_mean += loss / len(dynamic_train_datasets)

        print("loss_mean",loss_mean)



if __name__=='__main__':
    test_dynamic_train_multbody_push_t()





