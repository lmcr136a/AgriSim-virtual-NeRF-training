import numpy as np
from gym import Env
from gym.spaces import Box
from nerf_trainer import NeRFTrainer


class NeRFENV(Env):
    def __init__(self, args, position_candidates, epi_length=5):
        self.epi_length = epi_length
        self.position_candidates = position_candidates
        
        self.action_space = position_candidates
        self.observation_space = Box(low=np.array([0]), high=np.array([100]), dtype=np.int8)
        self.state = np.random.choice(position_candidates)
        self.prev_state = self.state
        self.epi_length = epi_length


        self.nerf_trainer = NeRFTrainer(args)
    
    def step(self, action):
        new_pos = self.action_space[action]
        self.state += new_pos
        self.action_space.pop(action)
        self.epi_length -= 1
        
        # Train the nerf with new pos imgs
        self.nerf_trainer.train(new_pos)

        # Reward
        reward = self.nerf_trainer.val()
        
        self.prev_state = self.state
        
        if self.epi_length <= 0:
            done = True
        else:
            done = False
        
        info = {}
        
        return self.get_obs(), reward, done, info
    
    
    def reset(self):
        self.action_space = self.position_candidates
        self.state = np.random.choice(self.position_candidates)
        self.prev_state = self.state
        self.epi_length = self.epi_length
        return self.get_obs()
    
    def get_obs(self):  # This depends on RL models
        return np.array([self.state], dtype=int)