from argparse import ArgumentParser, Namespace
from typing import NamedTuple
import os
import argparse
import torch

from diff_rendering.gaussian_splatting_2d.arguments import ModelParams, PipelineParams, OptimizationParams,GroupParams

def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    return parser

# combined args2 in args1
def get_combined_args(args1,args2):
    args1_dict = vars(args1)
    args2_dict = vars(args2)
    for k,v in args2_dict.items():
        if(k in args1_dict):
            args1_dict[k] = v
    return Namespace(**args1_dict)

class Gaussian_args(NamedTuple):
    dataset : GroupParams
    opt : GroupParams
    pipe : GroupParams
    testing_iterations : int
    saving_iterations : int
    checkpoint_iterations : int
    checkpoint : str

def get_gaussian_args(data_args,render_args,sys_args):
    # Set up command line argument parser
    parser = ArgumentParser(description="2dgs script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    # parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args([])
    
    args = get_combined_args(args,data_args)
    args = get_combined_args(args,render_args)
    args.save_iterations.append(3000)
    args.save_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations/2)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations/2)

    if sys_args.output_path != None:
        args.model_path = os.path.join(sys_args.output_path, "static")
    
    return Gaussian_args(lp.extract(args), op.extract(args), pp.extract(args), 
                         args.test_iterations, args.save_iterations, 
                         args.checkpoint_iterations, args.start_checkpoint)
