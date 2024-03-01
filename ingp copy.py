#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json

import numpy as np

import shutil
import time
import sys
ngp_path = '/home/koala/AgriSim-virtual-NeRF-training/models/instant-ngp'
sys.path.append(os.path.join(ngp_path, "build"))
import pyngp as ngp # noqa

sys.path.append(os.path.join(ngp_path, "scripts"))
from common import *
from scenes import *

from tqdm import tqdm


"""
python ../models/instant-ngp/scripts/run.py 
--scene $dest_dir --network base.json --n_steps 5000 
--save_mesh $screenshot_dir"mesh.obj" --screenshot_dir $screenshot_dir 
--screenshot_transforms $screenshot_transforms
    
"""
def get_scene(scene):
	for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
		if scene in scenes:
			return scenes[scene]
	return None

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
		
if __name__ == "__main__":
	
    # args = parse_args()
    args = Namespace(exposure=0.0, files=[], gui=False, height=0, load_snapshot='', 
                    marching_cubes_density_thresh=2.5, marching_cubes_res=256, mode='', n_steps=50000, 
                    near_distance=-1, nerf_compatibility=False, network='base.json', 
                    save_mesh='', save_snapshot='', scene='', 
                    screenshot_dir='', screenshot_frames=None, screenshot_spp=16, 
                    screenshot_transforms='', second_window=False, 
                    sharpen=0, test_transforms='', train=False, video_camera_path='', video_camera_smoothing=False, 
                    video_fps=60, video_n_seconds=1, video_output='video.mp4', video_render_range=(-1, -1), 
                    video_spp=8, vr=False, width=0)
    exp_path = "logs/test"
    img_dataset_path = "../data/unity_images"
    args.scene = img_dataset_path
    args.screenshot_transforms = os.path.join(img_dataset_path, "transforms.json")
    args.save_mesh = os.path.join(exp_path, "mesh.obj")
    args.screenshot_dir = os.path.join(exp_path)

    testbed = ngp.Testbed()
    testbed.root_dir = ROOT_DIR


    if args.scene:
        scene_info = get_scene(args.scene)
        if scene_info is not None:
            args.scene = os.path.join(scene_info["data_dir"], scene_info["dataset"])
            if not args.network and "network" in scene_info:
                args.network = scene_info["network"]

        testbed.load_training_data(args.scene)

    ref_transforms = {}
    if args.screenshot_transforms: # try to load the given file straight away
        print("Screenshot transforms from ", args.screenshot_transforms)
        with open(args.screenshot_transforms) as f:
            ref_transforms = json.load(f)

    testbed.nerf.sharpen = float(args.sharpen)
    testbed.exposure = args.exposure
    testbed.shall_train = args.train if args.gui else True


    testbed.nerf.render_with_lens_distortion = True

    network_stem = os.path.splitext(os.path.basename(args.network))[0] if args.network else "base"


    old_training_step = 0
    n_steps = args.n_steps

    # If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
    # don't train by default and instead assume that the goal is to render screenshots,
    # compute PSNR, or render a video.

    tqdm_last_update = 0
    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="steps") as t:
            while testbed.frame():
                if testbed.want_repl():
                    repl(testbed)

                if testbed.training_step >= n_steps:
                    break

                # Update progress bar
                if testbed.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()

                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now

    if args.save_mesh:
        res = args.marching_cubes_res or 256
        thresh = args.marching_cubes_density_thresh or 2.5
        print(f"Generating mesh via marching cubes and saving to {args.save_mesh}. Resolution=[{res},{res},{res}], Density Threshold={thresh}")
        testbed.compute_and_save_marching_cubes_mesh(args.save_mesh, [res, res, res], thresh=thresh)

    if ref_transforms:
        testbed.fov_axis = 0
        testbed.fov = ref_transforms["camera_angle_x"] * 180 / np.pi
        args.screenshot_frames = range(len(ref_transforms["frames"]))
        print(args.screenshot_frames)
        for idx in args.screenshot_frames:
            f = ref_transforms["frames"][int(idx)]
            cam_matrix = f.get("transform_matrix", f["transform_matrix_start"])
            testbed.set_nerf_camera_matrix(np.matrix(cam_matrix)[:-1,:])
            outname = os.path.join(args.screenshot_dir, os.path.basename(f["file_path"]))

            # Some NeRF datasets lack the .png suffix in the dataset metadata
            if not os.path.splitext(outname)[1]:
                outname = outname + ".png"

            print(f"rendering {outname}")
            image = testbed.render(args.width or int(ref_transforms["w"]), args.height or int(ref_transforms["h"]), args.screenshot_spp, True)
            os.makedirs(os.path.dirname(outname), exist_ok=True)
            write_image(outname, image)

    if args.video_camera_path:
        testbed.load_camera_path(args.video_camera_path)

        resolution = [args.width or 1920, args.height or 1080]
        n_frames = args.video_n_seconds * args.video_fps
        save_frames = "%" in args.video_output
        start_frame, end_frame = args.video_render_range

        if "tmp" in os.listdir():
            shutil.rmtree("tmp")
        os.makedirs("tmp")

        for i in tqdm(list(range(min(n_frames, n_frames+1))), unit="frames", desc=f"Rendering video"):
            testbed.camera_smoothing = args.video_camera_smoothing

            if start_frame >= 0 and i < start_frame:
                # For camera smoothing and motion blur to work, we cannot just start rendering
                # from middle of the sequence. Instead we render a very small image and discard it
                # for these initial frames.
                # TODO Replace this with a no-op render method once it's available
                frame = testbed.render(32, 32, 1, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
                continue
            elif end_frame >= 0 and i > end_frame:
                continue

            frame = testbed.render(resolution[0], resolution[1], args.video_spp, True, float(i)/n_frames, float(i + 1)/n_frames, args.video_fps, shutter_fraction=0.5)
            if save_frames:
                write_image(args.video_output % i, np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)
            else:
                write_image(f"tmp/{i:04d}.jpg", np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)

        if not save_frames:
            os.system(f"ffmpeg -y -framerate {args.video_fps} -i tmp/%04d.jpg -c:v libx264 -pix_fmt yuv420p {args.video_output}")

        shutil.rmtree("tmp")
