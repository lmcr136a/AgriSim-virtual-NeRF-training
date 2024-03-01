import torch
import argparse
import yaml
import numpy as np
from argparse import Namespace
from utils import get_configs
from ingp import INGPTrainer


if __name__ == "__main__":

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    """
    Total sample num of fern: 20
    16 = 4 + 4 + 4 + 4
        n0  n1  n2  n3
    """
    np.random.seed(0)
    confs = get_configs()
    
    for c in confs:
        ### This is for the baseline
        if c.baseline:
            random_poses = (np.random.random_sample((c.total_images, 3))-0.5)*c.limit*2
            random_poses[:, 1] = -np.abs(random_poses[:, 1])
            random_poses

            trainer = INGPTrainer(c)
            trainer.train(random_poses)
            trainer.val(0)
        else:
            from rl_test import run_rl
            run_rl(c)