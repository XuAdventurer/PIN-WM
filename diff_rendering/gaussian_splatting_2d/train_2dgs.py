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

import os
import torch
from random import randint
from diff_rendering.gaussian_splatting_2d.utils.loss_utils import l1_loss, ssim
from diff_rendering.gaussian_splatting_2d.gaussian_renderer import render, network_gui
import sys
from diff_rendering.gaussian_splatting_2d.scene import Scene, GaussianModel,MeshGaussianModel

from diff_rendering.gaussian_splatting_2d.utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import open3d as o3d
import numpy as np
import trimesh

from diff_rendering.gaussian_splatting_2d.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from diff_rendering.gaussian_splatting_2d.arguments import ModelParams, PipelineParams, OptimizationParams
from diff_rendering.gaussian_splatting_2d.utils.mesh_utils import GaussianExtractor,post_process_mesh
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,scene_info,mesh=None,logger=None):
    first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    if mesh is not None:
        gaussians = MeshGaussianModel(dataset.sh_degree,mesh)
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians,scene_info=scene_info)
    # import pdb;pdb.set_trace()
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # dataset.white_background = False
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="2DGS Training Progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        # import pdb;pdb.set_trace()
        image, opacity,viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["rend_alpha"],render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # from diff_rendering.gaussian_splatting_2d.utils.render_utils import save_img_u8
        # save_img_u8(image1.permute(1,2,0).cpu().detach().numpy(),  'test.png')
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        # print(Ll1)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        opacity_loss = opt.lambda_opacity * gaussians.get_opacity.mean()

        # scale loss
        # scale_loss=-0.0001*gaussians._scaling.mean()

        # import pdb;pdb.set_trace()

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        # import pdb;pdb.set_trace()
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss + opacity_loss

        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if logger is not None:
                logger.record('train_loss_patches/dist_loss', ema_dist_for_log, )
                logger.record('train_loss_patches/normal_loss', ema_normal_for_log, )

                training_report(logger, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                logger.dump(step = iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # render images
    image_output_dir = os.path.join(dataset.model_path, 'image', "ours_{}".format(opt.iterations))
    os.makedirs(image_output_dir, exist_ok=True)
    TrainCameras = sorted(scene.getTrainCameras(), key=lambda x: x.colmap_id) #scene训练的时候把顺序打乱了，这边要重新排序回去
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)   
    gaussExtractor.reconstruction(TrainCameras)
    gaussExtractor.export_image(image_output_dir)

    return gaussians

def prepare_output_and_logger(args):    

    # Set up output folder
    print("static 2dgs folder: {}".format(args.model_path))
    # os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(logger, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if logger:
        logger.record("train_loss_patches/reg_loss",Ll1.item(), )
        logger.record("train_loss_patches/total_loss",loss.item(), )
        logger.record('iter_time', elapsed, )
        logger.record('total_points', scene.gaussians.get_xyz.shape[0], )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if logger and (idx < 5):
                        from diff_rendering.gaussian_splatting_2d.utils.general_utils import colormap
                        from utils.sys_utils import Image
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        logger.record(config['name'] + "_view_{}/depth".format(viewpoint.image_name), Image(depth,dataformats="CHW"), )
                        logger.record(config['name'] + "_view_{}/render".format(viewpoint.image_name), Image(image,dataformats="CHW"), )

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            logger.record(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), Image(rend_normal,dataformats="CHW"), )
                            logger.record(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), Image(surf_normal,dataformats="CHW"), )
                            logger.record(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), Image(rend_alpha,dataformats="CHW"), )

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            logger.record(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), Image(rend_dist,dataformats="CHW"), )
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            logger.record(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), Image(gt_image,dataformats="CHW"), )

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if logger:
                    logger.record(config['name'] + '/loss_viewpoint - l1_loss', l1_test,  )
                    logger.record(config['name'] + '/loss_viewpoint - psnr', psnr_test,  )

        torch.cuda.empty_cache()


    

def train_static_2dgs(gaussian_args,scene_info,mesh=None,logger=None):
    
    gaussians = training(gaussian_args.dataset, gaussian_args.opt, gaussian_args.pipe, gaussian_args.testing_iterations, 
                       gaussian_args.saving_iterations, gaussian_args.checkpoint_iterations, gaussian_args.checkpoint,scene_info,mesh,logger)

    # All done
    print("\nTraining complete.")

    return gaussians