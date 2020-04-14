# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
from argparse import ArgumentParser

import numpy as np
import torch

from configs import get_cfg, add_config
from trainer import build_trainer
from utils import MyLogger, reward_function


def setup_cfg(args):
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file((args.config_file))
    cfg.merge_from_list(args.opts)
    if cfg.model.resume_path:
        cfg.logger.path = os.path.dirname(cfg.model.resume_path)
    cfg.logger.log_file = os.path.join(cfg.logger.path, 'log_search.txt')
    os.makedirs(cfg.logger.path, exist_ok=True)
    cfg.freeze()
    SEED = cfg.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return cfg

if __name__ == "__main__":
    parser = ArgumentParser("enas")
    parser.add_argument("--config_file", default="./configs/search.yaml", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = setup_cfg(args)
    logger = MyLogger('NNI_NAS', cfg).getlogger()
    logger.info(cfg)
    logger.info(args)
    with open(os.path.join(cfg.logger.path, 'search.yaml'), 'w') as f:
        f.write(str(cfg))

    trainer = build_trainer(cfg)
    trainer.train()