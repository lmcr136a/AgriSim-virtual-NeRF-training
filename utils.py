import argparse
import yaml
import os
from argparse import Namespace

def get_configs():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', '-c', default="test.yaml",nargs='+',
                        help='config file path')
    args = parser.parse_args()
    
    if type(args.config) == list:
        configs = []
        for conf in args.config:
            with open(os.path.join("configs", conf), 'r') as f:
                conf = yaml.safe_load(f)
            configs.append(Namespace(**conf))
        return configs

    else:
        with open(os.path.join("configs", args.config), 'r') as f:
            config = yaml.safe_load(f)
        config = Namespace(**config)
        return [config]
    
