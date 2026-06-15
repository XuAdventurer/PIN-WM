import argparse
import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from builder import Builder


sys_args = dict(
    seed=3,
    output_path="./output/sim_push_t",
)

data_args = dict(
    surround_shoot_num=30,
    n_frames=32,
    H=800,
    W=800,
    data_path="./dataset/sim_push_t",
    obj_path="./envs/asset/cube_t/cube_t_target.urdf",
    obj_position=[0.0, 0.0, 0.025],
    obj_rotation=[0.0, 0.0, 0.0, 1.0],
    ee_path="./envs/asset/ee/ee.urdf",
    ee_position=[0.05, 0.05, 0.05],
    ee_rotation=[0.0, 0.0, 0.0, 1.0],
    ee_goal_position=[-0.006, 0.04],
)

render_args = dict()

sim_args = dict(
    dtime=1.0 / 240,
    frame_dt=1.0 / 24,
    f_max=4,
    sync=False,
    vis=True,
)

policy_args = dict()


def parse_args():
    parser = argparse.ArgumentParser(description="Collect video data with PyBullet.")
    parser.add_argument("--data-path", default=data_args["data_path"])
    parser.add_argument("--goal-x", type=float, default=data_args["ee_goal_position"][0])
    parser.add_argument("--goal-y", type=float, default=data_args["ee_goal_position"][1])
    parser.add_argument("--frames", type=int, default=data_args["n_frames"])
    parser.add_argument("--height", type=int, default=data_args["H"])
    parser.add_argument("--width", type=int, default=data_args["W"])
    vis_group = parser.add_mutually_exclusive_group()
    vis_group.add_argument("--vis", action="store_true", dest="vis", default=sim_args["vis"])
    vis_group.add_argument("--no-vis", action="store_false", dest="vis")
    return parser.parse_args()


def collect_video_data(args=None):
    if args is None:
        args = parse_args()

    current_data_args = dict(data_args)
    current_data_args.update(
        data_path=args.data_path,
        n_frames=args.frames,
        H=args.height,
        W=args.width,
        ee_goal_position=[args.goal_x, args.goal_y],
    )

    current_sim_args = dict(sim_args)
    current_sim_args["vis"] = args.vis

    all_args = {
        "sys_args": sys_args,
        "data_args": current_data_args,
        "render_args": render_args,
        "sim_args": current_sim_args,
        "policy_args": policy_args,
    }

    builder = Builder(all_args)
    data_collect_env = builder.build_data_collect_env()
    try:
        data_collect_env.collect_video_data(builder.data_args.ee_goal_position)
    finally:
        data_collect_env.close()

    print("Video data saved to:", builder.data_args.data_path)


if __name__ == "__main__":
    collect_video_data()
