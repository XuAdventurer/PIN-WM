import argparse
import os
import sys

import trimesh


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from builder import Builder
from utils.data_utils import load_2dgs_dataset
from utils.sys_utils import prepare_output_and_logger


sys_args = dict(
    seed=0,
    output_path="./output/sim_push_t",
    log_dir_name="2dgs",
    model_path=None,
)

data_args = dict(
    white_background=True,
    H=800,
    W=800,
    data_path="./dataset/sim_push_t",
    data_json="2dgs_data.json",
    obj_path="./envs/asset/cube_t/cube_t.obj",
    obj_position=[0.0, 0.0, 0.025],
    obj_rotation=[0.0, 0.0, 0.0, 1.0],
)

render_args = dict(
    sh_degree=3,
    lambda_opacity=0,
    iterations=30000,
    sphere_pcd_init=True,
    xyz_scale=0.09,
)

sim_args = dict(
    epsilon=0.004,
)

policy_args = dict()


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2DGS from collected 2DGS data.")
    parser.add_argument("--data-path", default=data_args["data_path"])
    parser.add_argument("--data-json", default=data_args["data_json"])
    parser.add_argument("--output-path", default=sys_args["output_path"])
    parser.add_argument("--iterations", type=int, default=render_args["iterations"])
    parser.add_argument("--obj-path", default=data_args["obj_path"])
    parser.add_argument("--height", type=int, default=data_args["H"])
    parser.add_argument("--width", type=int, default=data_args["W"])
    return parser.parse_args()


def train_2dgs(args=None):
    if args is None:
        args = parse_args()

    current_sys_args = dict(sys_args)
    current_sys_args["output_path"] = args.output_path

    current_data_args = dict(data_args)
    current_data_args.update(
        data_path=args.data_path,
        data_json=args.data_json,
        obj_path=args.obj_path,
        H=args.height,
        W=args.width,
    )

    current_render_args = dict(render_args)
    current_render_args["iterations"] = args.iterations

    all_args = {
        "sys_args": current_sys_args,
        "data_args": current_data_args,
        "render_args": current_render_args,
        "sim_args": sim_args,
        "policy_args": policy_args,
    }
    all_args, loggers = prepare_output_and_logger(all_args, need_logger=True)
    all_args["sys_args"]["model_path"] = all_args["sys_args"]["log_path"]

    builder = Builder(all_args)
    dataset = load_2dgs_dataset(builder.data_args)
    train_cam_infos = dataset["cam_infos"]

    mesh = trimesh.load(builder.data_args.obj_path)
    mesh.apply_translation(builder.data_args.obj_position)

    builder.build_static_2dgs(train_cam_infos, mesh, loggers["static"])
    print("2DGS output folder:", builder.gaussian_args.dataset.model_path)


if __name__ == "__main__":
    train_2dgs()
