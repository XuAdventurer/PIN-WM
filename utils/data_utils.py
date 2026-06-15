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

from scipy.spatial.transform import Rotation as RR

def readDataFromJson(path, json_file, white_background,extension):
    with open(os.path.join(path, json_file)) as json_file:
        contents = json.load(json_file)

    ee_init_position = np.array(contents["ee_init_position"],dtype=np.float32) if "ee_init_position" in contents else None
    ee_goal_position = np.array(contents["ee_goal_position"],dtype=np.float32) if "ee_goal_position" in contents else None

    cam_infos = []
    image_datas = contents["image_datas"]
    for entry in image_datas:
        image_path = entry["file_path"]
        image_filename = os.path.basename(image_path)
        image_stem = image_filename[:-len(extension)] if image_filename.endswith(extension) else image_filename
        cam_id, frame_id = [int(i) for i in image_stem.lstrip("r_").split("_")]
        if frame_id < 0:
            continue
        
        c2w = np.array(entry["c2w"] + [[0.0,0.0,0.0,1.0]])
        # temp = np.copy(c2w[1, :])
        # c2w[1, :]=c2w[2, :]
        # c2w[2, :]=temp
        
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]      
        
        image_path = os.path.join(path, image_path)
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


def load_video_dataset(data_args):
    cam_infos,ee_init_position,ee_goal_position,pose_datas = readDataFromJson(data_args.data_path, "video_data.json",
                                                       data_args.white_background, extension=".png")
    return {
        "cam_infos":cam_infos,
        'ee_init_position':ee_init_position,
        "ee_goal_position":ee_goal_position,
        "pose_datas":pose_datas
    }


def load_2dgs_dataset(data_args):
    json_name = getattr(data_args, "data_json", "2dgs_data.json")
    cam_infos,_,_,_ = readDataFromJson(data_args.data_path, json_name,
                                       data_args.white_background, extension=".png")
    return {
        "cam_infos":cam_infos
    }


def get_2dgs_train_cam_infos(all_train_cam_infos):
    train_cam_infos = []
    for cam in all_train_cam_infos:
        if cam.frameid == 0:
            train_cam_infos.append(cam)
    return train_cam_infos


def get_gaussian_scene_info(train_cam_infos,data_args,render_args):
    ply_path = os.path.join(data_args.data_path, "points3d.ply")
    num_pts = 10_000
    print(f"Generating random point cloud ({num_pts})...")

    xyz_scale = getattr(render_args, "xyz_scale", 0.09)
    obj_position = np.array(getattr(data_args, "obj_position", [0.0, 0.0, 0.025]))
    if getattr(render_args, "sphere_pcd_init", False):
        theta = np.random.uniform(0, np.pi, num_pts)
        phi = np.random.uniform(0, 2 * np.pi, num_pts)
        xyz = np.column_stack((
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        )) * xyz_scale + obj_position
    else:
        xyz = (np.random.random((num_pts, 3)) * 2.6 - 1.3) * xyz_scale + obj_position

    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pass

    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=train_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


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


def save_data(images,image_datas,ee_goal_position,pose_datas,data_path,data_folder_name,json_name):
    data_folder_path = os.path.join(data_path, data_folder_name)
    os.makedirs(data_folder_path, exist_ok=True)
    image_paths = []
    for image_data in image_datas:
        image_name = image_data.pop('image_name')
        image_data['file_path'] = os.path.join(data_folder_name, image_name)
        image_paths.append(os.path.join(data_folder_path, image_name))
    save_images(images, image_paths)
    json_data = json.dumps(
        {
            "ee_goal_position": ee_goal_position.tolist(),
            "image_datas": image_datas,
            "pose_datas": pose_datas,
        },
        indent=4,
    )
    with open(os.path.join(data_path, json_name), 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)


def save_video_data(images,image_datas,ee_init_position,ee_goal_position,pose_datas,data_path):
    data_folder_name = "video_data"
    data_folder_path = os.path.join(data_path, data_folder_name)
    os.makedirs(data_folder_path, exist_ok=True)
    image_paths = []
    for image_data in image_datas:
        image_name = image_data.pop('image_name')
        image_data['file_path'] = os.path.join(data_folder_name, image_name)
        image_paths.append(os.path.join(data_folder_path, image_name))
    save_images(images, image_paths)
    json_data = json.dumps(
        {
            "ee_init_position": ee_init_position.tolist(),
            "ee_goal_position": ee_goal_position.tolist(),
            "image_datas": image_datas,
            "pose_datas": pose_datas,
        },
        indent=4,
    )
    with open(os.path.join(data_path, "video_data.json"), 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)
