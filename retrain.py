
import argparse
import os
from argparse import ArgumentParser

import numpy as np
import torch

import nni
from configs import add_config, get_cfg
from evaluator import DefaultEvaluator
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeter
from utils import MyLogger


def setup_cfg(args):
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file((args.config_file))
    cfg.merge_from_list(args.opts)
    if cfg.model.resume_path:
        cfg.logger.path = os.path.dirname(cfg.model.resume_path)
    else:
        index = 0
        path = os.path.dirname(args.arc_path)+'_train_{}'.format(index)
        while os.path.exists(path):
            index += 1
            path = os.path.dirname(args.arc_path)+'_train_{}'.format(index)
        cfg.logger.path = path
    cfg.logger.log_file = os.path.join(cfg.logger.path, 'log_train.txt')
    os.makedirs(cfg.logger.path, exist_ok=True)
    cfg.freeze()
    SEED = cfg.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return cfg


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--config_file", default='./config/retrain.yaml', type=str)
    parser.add_argument("--train_epochs", default=2, type=int)
    parser.add_argument("--arc_path") # "./outputs/checkpoint_0" or "./outputs/checkpoint_0/epoch_1.json"
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    config_file = args.config_file
    if os.path.isdir(args.arc_path) and args.arc_path[-1] != '/':
        args.arc_path += '/'
    arc_path = args.arc_path
    
    assert config_file and arc_path, f"please check whether {config_file} and {arc_path} exists"

    # configuration
    cfg = setup_cfg(args)
    with open(os.path.join(cfg.logger.path, 'cfg_retrain.yaml'), 'w') as f:
        f.write(str(cfg))
    cfg.update({'args': args})
    logger = MyLogger(__name__, cfg).getlogger()
    logger.info('args:{}'.format(args))

    evaluator = DefaultEvaluator(cfg, args.arc_path)
    
    if os.path.isdir(arc_path):
        best_arch_info = evaluator.compare()
        evaluator.run(best_arch_info['arc'])
    elif os.path.isfile(arc_path):
        evaluator.run(arc_path)
