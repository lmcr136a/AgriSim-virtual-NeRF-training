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
import time
import sys
import yaml
import shutil
import commentjson as json
import numpy as np

ngp_path = '/home/koala/AgriSim-virtual-NeRF-training/models/instant-ngp'
sys.path.append(os.path.join(ngp_path, "build"))
import pyngp as ngp # noqa

sys.path.append("..")
from AgriSim.unity_sampler.sampler import UnitySampler, CameraViewpoint

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

def get_configs():
    from argparse import Namespace

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', '-c', default="configs/test.yaml",
                        help='config file path')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        configs = yaml.safe_load(f)

    ns = Namespace(**configs)
    return ns
    

def get_scene(scene):
	for scenes in [scenes_sdf, scenes_nerf, scenes_image, scenes_volume]:
		if scene in scenes:
			return scenes[scene]
	return None

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class INGPTrainer():
    def __init__(self, c, rl_args=None):
        self.rl_args = rl_args
        args = Namespace(exposure=0.0, files=[], gui=False, height=0, load_snapshot='', 
                        marching_cubes_density_thresh=2.5, marching_cubes_res=256, mode='', n_steps=c.total_training_steps, 
                        near_distance=-1, nerf_compatibility=False, network='base.json', 
                        save_mesh='', save_snapshot='', scene='', 
                        screenshot_dir='', screenshot_frames=None, screenshot_spp=16, 
                        screenshot_transforms='', second_window=False, 
                        sharpen=0, test_transforms='', train=False, video_camera_path='', video_camera_smoothing=False, 
                        video_fps=60, video_n_seconds=1, video_output='video.mp4', video_render_range=(-1, -1), 
                        video_spp=8, vr=False, width=0)
        args.scene = c.img_dataset_path
        args.screenshot_transforms = os.path.join(c.img_dataset_path, "transforms.json")
        args.save_mesh = os.path.join(c.exp_path, "mesh.obj")
        args.screenshot_dir = os.path.join(c.exp_path)
        args.test_transforms = c.test_dataset_path

        if not os.path.exists(c.exp_path):
             os.mkdir(c.exp_path)        

        self.testbed = ngp.Testbed()
        self.testbed.root_dir = ROOT_DIR
        
        ## Unity dataset
        # self.dataset_info = json.load(open(os.path.join(args.scene, "all_transforms.json")))
        self.usampler = UnitySampler(
                                object_family='02843684', 
                                object_id='1b73f96cf598ef492cba66dc6aeabcd4', 
                                screenshot_dir=c.img_dataset_path,
                                config_path='../AgriSim/config.yaml'
                                )

        # self.ref_transforms = {}
        # print("Screenshot transforms from ", args.screenshot_transforms)
        # with open(args.screenshot_transforms) as f:
        #     self.ref_transforms = json.load(f)

        self.testbed.nerf.sharpen = float(args.sharpen)
        self.testbed.exposure = args.exposure
        self.testbed.shall_train = args.train if args.gui else True
        self.testbed.nerf.render_with_lens_distortion = True

        self.print_interval = 5
        self.args = args
        with open(os.path.join(c.exp_path, "config.yaml"), 'w') as f:
            yaml.dump(c)


    def set_selected_dataset(self, selec_poses):
        self.usampler.generate_imgs(additional_poses=selec_poses)  # x, y, z, theta, phi

    def train(self, selec_poses):
        old_training_step = 0
        n_steps = self.args.n_steps

        self.set_selected_dataset(selec_poses)
        print(self.args.scene)
        self.testbed.load_training_data(self.args.scene)


        tqdm_last_update = 0
        if n_steps > 0:
            with tqdm(desc="Training", total=n_steps, unit="steps") as t:
                while self.testbed.frame():
                    if self.testbed.want_repl():
                        repl(self.testbed)

                    if self.testbed.training_step >= n_steps:
                        break

                    # Update progress bar
                    if self.testbed.training_step < old_training_step or old_training_step == 0:
                        old_training_step = 0
                        t.reset()

                    now = time.monotonic()
                    if now - tqdm_last_update > 0.1:
                        t.update(self.testbed.training_step - old_training_step)
                        t.set_postfix(loss=self.testbed.loss)
                        old_training_step = self.testbed.training_step
                        tqdm_last_update = now

        if self.args.save_mesh:
            res = self.args.marching_cubes_res or 256
            thresh = self.args.marching_cubes_density_thresh or 2.5
            print(f"Generating mesh via marching cubes and saving to {self.args.save_mesh}. Resolution=[{res},{res},{res}], Density Threshold={thresh}")
            self.testbed.compute_and_save_marching_cubes_mesh(
                 self.args.save_mesh, [res, res, res], thresh=thresh)
  
    def val(self, epoch):
        print("Evaluating test transforms from ", self.args.test_transforms)
        
        totmse = 0
        totpsnr = 0
        totssim = 0
        totcount = 0
        minpsnr = 1000
        maxpsnr = 0

        # Evaluate metrics on black background
        self.testbed.background_color = [0.0, 0.0, 0.0, 1.0]

        # Prior nerf papers don't typically do multi-sample anti aliasing.
        # So snap all pixels to the pixel centers.
        self.testbed.snap_to_pixel_centers = True
        spp = 8

        self.testbed.nerf.render_min_transmittance = 1e-4

        self.testbed.shall_train = False
        self.testbed.load_training_data(self.args.test_transforms)
        diffimg = np.zeros((10, 10))
        with tqdm(range(self.testbed.nerf.training.dataset.n_images), unit="images", desc=f"Rendering test frame") as t:
            for i in t:
                resolution = self.testbed.nerf.training.dataset.metadata[i].resolution
                self.testbed.render_ground_truth = True
                self.testbed.set_camera_to_training_view(i)
                ref_image = self.testbed.render(resolution[0], resolution[1], 1, True)
                self.testbed.render_ground_truth = False
                image = self.testbed.render(resolution[0], resolution[1], spp, True)
                if epoch%self.print_interval == 0:
                    write_image(f"{self.args.screenshot_dir}/scene{i}_ref.png", ref_image)
                    write_image(f"{self.args.screenshot_dir}/scene{i}_out.png", image)

                    diffimg = np.absolute(image - ref_image)
                    diffimg[...,3:4] = 1.0
                    # write_image(f"{self.args.screenshot_dir}/diff.png", diffimg)

                A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
                R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
                mse = float(compute_error("MSE", A, R))
                ssim = float(compute_error("SSIM", A, R))
                totssim += ssim
                totmse += mse
                psnr = mse2psnr(mse)
                totpsnr += psnr
                minpsnr = psnr if psnr<minpsnr else minpsnr
                maxpsnr = psnr if psnr>maxpsnr else maxpsnr
                totcount = totcount+1
                t.set_postfix(psnr = totpsnr/(totcount or 1))

        psnr_avgmse = mse2psnr(totmse/(totcount or 1))
        psnr = totpsnr/(totcount or 1)
        ssim = totssim/(totcount or 1)
        print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")
        write_image(f"{self.args.screenshot_dir}/0_P={round(psnr, 2)}_S={round(ssim, 4)}_diff.png", diffimg)
        return ssim


if __name__ == "__main__":
    from utils import get_configs
    import time
    start = time.time()
    np.random.seed(0)
    confs = get_configs()
    for c in confs:
        ### This is for the baseline
        if c.baseline:
            rand_poses = np.random.random_sample((c.total_images, 3))   # 0~1
            rand_poses = rand_poses * (c.pose_max - c.pose_min) + c.pose_min
            minous = (2*np.random.randint(0,2,size=(rand_poses.shape))-1)#.reshape(rand_poses.shape)
            rand_poses = minous*rand_poses 
            rand_poses[:,1] = -np.abs(rand_poses[:,1])

            trainer = INGPTrainer(c)
            trainer.train(rand_poses)
            trainer.val(0)
        else:
            from rl_test import run_rl
            run_rl(c)
        print(f"\n \t{(time.time() - start)/60} min\n")