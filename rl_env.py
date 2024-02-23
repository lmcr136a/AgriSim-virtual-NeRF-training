import numpy as np
from gym import Env
from gym.spaces import Box
from itertools import combinations

from nerf_trainer import NeRFTrainer
from utils import *


"""
observation (state): [validation_img, positions_used]
poses = [pos1, ..., posN]
action_space: [(pos_idx1,.., pos_idx4)_1, (pos_idx1,.., pos_idx4)_2, ...]
action: [pos1, pos2, pos3, pos4]
"""


class NeRFENV(Env):
    def __init__(self, args, epi_length=5):
        self.args = args
        self.nerf_trainer = NeRFTrainer(args)

        self.epi_length = epi_length
        self.poses = self.nerf_trainer.poses
        # self.pose_idx_combs = np.array(list(combinations(list(range(len(self.poses))), args.n)))  # combination of pose index C*4



        initial_idxs = np.random.randint(0, self.poses.shape[0], args.n0)
        self.selec_idx = initial_idxs
        self.ks = get_Ntrain_data_nums(args, self.poses)
        
        self.action_space = self.poses
        # self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.int8)

        initial_poses_idx = self.randomly_select_pose_idx(args.n0)
        initial_poses = self.poses[initial_poses_idx]
        reward, self.state = self.get_state_reward(initial_poses, initial_poses_idx)   # loss, (rgbs, val_gt_poses)
        self.prev_state = self.state

        print(f"""
RL Start{"="*10}
    Action Space: {self.action_space.shape}
    Episode Len: {self.epi_length}
              """)

    def get_actions(self, action_space_selection):
        return self.poses[action_space_selection, :, :]
    
    def get_state_reward(self, new_poses, new_poses_idxs):
        # print(type(new_poses), type(new_poses_idxs))
        # print(new_poses.shape, len(new_poses_idxs))
        self.nerf_trainer.train(new_poses, new_poses_idxs)
        return self.nerf_trainer.val()
    
    def randomly_select_pose_idx(self, how_many_pose):
        return np.random.choice(list(range(len(self.poses))), how_many_pose, replace=False)
    
    def step(self, action):
        self.epi_length -= 1
        if type(action) != list and type(action) != np.ndarray:
            action = np.array([action])
        new_pos = self.action_space[action]  # action 1 -> one case that grabs n poses, there is nC_N actions
        # self.action_space.pop(action)
        action = np.array(action)
        print(type(new_pos), type(action))
        print(new_pos.shape, action.shape)
        self.state.append(new_pos)
        reward = self.get_state_reward(new_pos, action)   # loss (psnr)
        
        self.prev_state = self.state
        
        if self.epi_length <= 0:
            done = True
        else:
            done = False
        
        info = {}
        
        return self.get_obs(), reward, done, info
    
    
    def reset(self):
        self.action_space = self.poses

        initial_poses_idx = self.randomly_select_pose_idx(self.args.n0)
        initial_poses = self.poses[initial_poses_idx]
        reward, self.state = self.get_state_reward(initial_poses, initial_poses_idx)   # loss, (rgbs, val_gt_poses)
        self.prev_state = self.state
        self.epi_length = self.epi_length
        return self.get_obs()
    
    def get_obs(self):  # This depends on RL models
        return self.state