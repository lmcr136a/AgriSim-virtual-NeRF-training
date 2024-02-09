import torch
from utils import config_parser
from rl import RL


if __name__ == "__main__":

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    """
    Total sample num of fern: 20
    16 = 4 + 4 + 4 + 4
        n0  n1  n2  n3
    """
            
    parser = config_parser()
    args = parser.parse_args()
    
    rl = RL(args)
    rl.train()

    ## Baseline