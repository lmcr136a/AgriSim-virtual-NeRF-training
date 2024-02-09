# from AgriSim.unity_sampler import Sampler, CameraViewpoint
import os, sys
import numpy as np
import imageio
import json
import random
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt



from utils import *
    

if __name__ == "__main__":

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    """
    Total sample num of fern: 20
    16 = 4 + 4 + 4 + 4
        n0  n1  n2  n3
    """
            
    parser = config_parser()
    args = parser.parse_args()



    initial_idxs = np.random.randint(0, poses.shape[0], args.n0)
    selec_idx = initial_idxs
    
    ks = get_Ntrain_data_nums(args, poses)
