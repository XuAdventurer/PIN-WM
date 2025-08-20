import torch
import numpy as np
import imageio
from tqdm import tqdm
import os,json,sys
import cv2
from PIL import Image
from typing import NamedTuple


from diff_rendering.gaussian_splatting_2d.scene.dataset_readers import SceneInfo,BasicPointCloud,SH2RGB,storePly,fetchPly,CameraInfo,fov2focal,focal2fov,getNerfppNorm
from diff_rendering.gaussian_splatting_2d.utils.camera_utils import cameraList_from_camInfos

from diff_simulation.force.constant_force import Constant_Force
from scipy.spatial.transform import Rotation as RR

def readDataFromJson(path, json_file, white_background,extension):
    with open(os.path.join(path, json_file)) as json_file:
        contents = json.load(json_file)

    ee_init_position = np.array(contents["ee_init_position"],dtype=np.float32)
    ee_goal_position = np.array(contents["ee_goal_position"],dtype=np.float32)

    cam_infos = []
    image_datas = contents["image_datas"]
    for entry in image_datas:
        image_path = entry["file_path"]
        cam_id, frame_id = [int(i) for i in image_path.split("/")[-1].rstrip(extension).lstrip("r_").split("_")]
        if frame_id < 0:
            continue
        
        c2w = entry["c2w"]
        c2w.append([0.0,0.0,0.0,1.0])
        c2w = np.array(c2w)
        # temp = np.copy(c2w[1, :])
        # c2w[1, :]=c2w[2, :]
        # c2w[2, :]=temp
        
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]      
        

        # image_path = os.path.join(path, image_path.replace('r', 'm'))
        image_path = os.path.join(path, 'm'.join(image_path.rsplit('r', 1)))
        image_name = os.path.basename(image_path).split(".")[0]
        
        image = Image.open(image_path)
        
        im_data = np.array(image.convert("RGBA"))
        
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        arr = np.concatenate((arr, norm_data[:,:,3][..., np.newaxis]), axis=-1)
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")

        intrinsic = entry["intrinsic"]
        fovx = np.arctan(0.5*image.size[0]/intrinsic[0][0])*2
        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx            
        
        cam_infos.append(CameraInfo(uid=cam_id,frameid=frame_id, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
        
    if "pose_datas" in contents:
        pose_datas = np.array(contents["pose_datas"])
    else:
        pose_datas = None

    return cam_infos,ee_init_position,ee_goal_position,pose_datas


def load_pybullet_dynamic_dataset(data_args):
    dynamic_datasets = []
    for index in range(data_args.train_num+data_args.test_num):
        if index < data_args.train_num:
            json_name = "dynamic_train{}.json".format(index)
        else:
            json_name = "dynamic_test{}.json".format(index-data_args.train_num)
        dynamic_cam_infos,ee_init_position,ee_goal_position,pose_datas = readDataFromJson(data_args.data_path, json_name,
                                                                  data_args.white_background, extension=".png")
        dynamic_datasets.append(
            {
                "cam_infos":dynamic_cam_infos,
                'ee_init_position':ee_init_position,
                "ee_goal_position":ee_goal_position,
                "pose_datas":pose_datas
            }
        )
    return dynamic_datasets[:data_args.train_num],dynamic_datasets[data_args.train_num:data_args.test_num]


def get_all_frames_train_camera(cam_id,all_train_cam_infos,dataset):
    all_frames_train_camera_infos = []
    for cam in all_train_cam_infos:
        if cam.uid == cam_id:
            all_frames_train_camera_infos.append(cam)
    all_frames_train_camera_infos = sorted(all_frames_train_camera_infos,key=lambda x: x.frameid)
    all_frames_train_camera = cameraList_from_camInfos(all_frames_train_camera_infos, 1.0,dataset)
    return all_frames_train_camera

def save_images(images,image_paths):
    for image,image_path in zip(images,image_paths):
        cv2.imwrite(image_path, image)
