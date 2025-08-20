import numpy as np
import os
import enum
import types
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm
import imageio


def create_videos_with_rgbs(rgbs,out_dir):
    video_kwargs = {
      'shape': rgbs[0].shape[1:3],
      'codec': 'h264',
      'fps': 6,
      'crf': 18,
    }
    input_format='rgb'
    # import pdb;pdb.set_trace()
    with media.VideoWriter(
        out_dir, **video_kwargs, input_format=input_format) as writer:
        for rgb in rgbs:
            rgb=rgb.permute(1, 2, 0).cpu().detach().numpy()
            rgb = (np.clip(np.nan_to_num(rgb), 0., 1.) * 255.).astype(np.uint8)
            writer.add_image(rgb)

def create_gif_with_rgbs(rgbs,out_dir):
    rgbs = [rgb.permute(1, 2, 0).cpu().detach().numpy() for rgb in rgbs]
    rgbs = (np.clip(np.nan_to_num(rgbs), 0., 1.) * 255.).astype(np.uint8)
    imageio.mimsave(out_dir, rgbs, fps=6) 