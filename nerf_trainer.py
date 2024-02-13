# from AgriSim.unity_sampler import Sampler, CameraViewpoint
import os, sys
import numpy as np
import imageio
import shutil
import time
import torch
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from yenchenlin.load_llff import load_llff_data
from yenchenlin.load_deepvoxels import load_dv_data
from yenchenlin.load_blender import load_blender_data
from yenchenlin.load_LINEMOD import load_LINEMOD_data

from utils import *
    

class NeRFTrainer():
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = os.path.join(self.args.basedir, self.args.expname)
        # if os.path.exists(self.log_dir):
        #     shutil.rmtree(self.log_dir)
        #     os.makedirs(self.log_dir)
        self.images, self.poses, self.bds, \
            self.render_poses, self.i_test = load_llff_data(
                                                    self.args.datadir, self.args.factor,
                                                    recenter=True, bd_factor=.75,
                                                    spherify=args.spherify)

        # Create nerf model
        self.render_kwargs_train, self.render_kwargs_test, \
                            start, self.grad_vars, self.optimizer = create_nerf(args)
        self.define_bounds()
        
        self.N_rand = self.args.N_rand
        self.global_step = start

        self.val_iter = self.args.val_iter
        self.train_loss = []
        self.val_loss = []


    def train(self, selec_poses, selec_idxs):
        images, poses = self.images[selec_idxs, :, :, :], selec_poses

        rays_rgb, i_batch = train_preprocess(
                                                self.args, images, poses, self.bds, 
                                                self.render_poses, self.i_test)
        hwf = poses[0,:3,-1]
        H, W, self.K = get_K(hwf)
        self.start = 1
        losses = []
        print("___NeRF training start___")
        for i in trange(self.start, self.start+ self.args.n_iters):

            i_batch, batch_rays, target_s= get_batch(rays_rgb, i_batch, self.N_rand)

            rgb, disp, acc, extras = render(H, W, self.K, chunk=self.args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **self.render_kwargs_train)
            self.optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            # if i%self.args.i_testset==0 and i > 0:
            #     rgb8_gt = to8b(images[self.i_test][i])
            #     filename = os.path.join(self.log_dir, '{:03d}_model_output.png'.format(i))
            #     imageio.imwrite(filename, rgb8_gt)
            
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)

            loss.backward()
            self.optimizer.step()
            self.optimizer = update_lr(self.args, self.optimizer, self.global_step)

            if i%self.val_iter==0 and i > 0:
                self.val()
            # logs(self.args, i, render_poses, images, self.global_step, self.render_kwargs_train, self.optimizer, self.render_kwargs_test, 
            #         self.hwf, K, poses, loss, psnr, self.i_test)
            self.global_step += 1
            self.train_loss.append(loss.detach().cpu().numpy())
        # self.train_loss.append(np.mean(losses))
        
    
    def val(self,):
        with torch.no_grad():
            if type(self.i_test) != list:
                self.i_test = [self.i_test]
            val_gt_imgs = self.images[self.i_test]
            val_gt_poses = self.poses[self.i_test]
            hwf = val_gt_poses[0, :3,-1]

            rgbs, disps = render_path(torch.Tensor(val_gt_poses).to(self.device), hwf, self.K, self.args.chunk, 
                        self.render_kwargs_test, gt_imgs=val_gt_imgs, savedir=self.log_dir)
            img_loss = img2mse(rgbs, val_gt_imgs)
            psnr = mse2psnr(img_loss)
        loss = np.mean(psnr.detach().cpu().numpy())
        self.val_loss.append(loss)
        # Reward, State
        self.save_plot()
        return loss, (rgbs, val_gt_poses)
    
    def save_plot(self, postfix=""):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,7))
        fig.suptitle("Train&Val Loss")
        axs[0].set_title('Training Metric')
        x = list(range(len(self.train_loss)))
        axs[0].plot(x, self.train_loss)
        axs[0].set_xlabel("Iter")
        axs[0].set_ylabel("Loss (PSNR)")
        axs[1].set_title('Validation Metric')
        x = list(range(len(self.val_loss)))
        axs[1].plot(x, self.val_loss)
        axs[1].set_xlabel("Iter")
        axs[1].set_ylabel("Loss (PSNR)")
        plt.savefig(os.path.join(self.log_dir, f"lossplot_{postfix}.jpg"))



    def define_bounds(self):
        print('DEFINING BOUNDS')
        if self.args.no_ndc:
            near = np.ndarray.min(self.bds) * .9
            far = np.ndarray.max(self.bds) * 1.
            
        else:
            near = 0.
            far = 1.
        bds_dict = {
            'near' : near,
            'far' : far,
        }
        self.render_kwargs_train.update(bds_dict)
        self.render_kwargs_test.update(bds_dict)