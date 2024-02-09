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
        self.nerf_trainer = NeRFTrainer(args)

        self.epi_length = epi_length
        self.poses = self.nerf_trainer.poses
        self.pose_idx_combs = np.array(list(combinations(list(range(len(self.poses))), args.n)))  # combination of pose index C*4



        initial_idxs = np.random.randint(0, self.poses.shape[0], args.n0)
        self.selec_idx = initial_idxs
        self.ks = get_Ntrain_data_nums(args, self.poses)
        
        self.action_space = self.pose_idx_combs
        # self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.int8)
        self.state = self.poses[self.pose_idx_combs[np.random.choice(list(range(len(self.pose_idx_combs))))]]  # n*3*5
        self.prev_state = self.state

        print(f"""
RL Start{"="*10}
    Action Space: {self.action_space}
    Episode Len: {self.epi_length}
              """)

    def get_actions(self, action_space_selection):
        return self.poses[action_space_selection, :, :]
    
    def step(self, action):
        print("self.action_space", self.action_space.shape)
        print("action", action)
        print(action.shape)
        new_pos_idx = self.action_space[action]  # action 1 -> one case that grabs n poses, there is nC_N actions
        print("new_pos_idx",new_pos_idx.shape)
        new_pos = self.poses[new_pos_idx]
        print(new_pos.shape, self.state.shape)
        exit()
        self.state = np.concatenate((self.state, new_pos))
        # self.action_space.pop(action)
        self.epi_length -= 1
        
        # Train the nerf with new pos imgs
        self.nerf_trainer.train(new_pos)

        # Reward
        reward, new_state_img = self.nerf_trainer.val()
        
        self.prev_state = self.state
        
        if self.epi_length <= 0:
            done = True
        else:
            done = False
        
        info = {}
        
        return self.get_obs(), reward, done, info
    
    
    def reset(self):
        self.action_space = self.pose_idx_combs
        self.state = self.poses[self.pose_idx_combs[np.random.choice(list(range(len(self.pose_idx_combs))))]]
        self.prev_state = self.state
        self.epi_length = self.epi_length
        return self.get_obs()
    
    def get_obs(self):  # This depends on RL models
        return np.array([self.state], dtype=int)