# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from nni.nas.pytorch import enas
from nni.nas.pytorch.callbacks import (ArchitectureCheckpoint,
                                       LRSchedulerCallback)

from callbacks import CheckpointCallback
from configs import get_cfg, add_config
from datasets import build_dataset
from losses import build_loss_fn
from mutator import build_mutator
from networks import build_model
from trainer import EnasTrainer, RandomTrainer
from utils import MyLogger, metrics, reward_function


def setup_cfg(args):
    cfg = get_cfg()
    add_config(cfg)
    cfg.merge_from_file((args.config_file))
    cfg.merge_from_list(args.opts)
    if cfg.model.resume_path:
        cfg.logger.path = os.path.dirname(cfg.model.resume_path)
        cfg.logger.log_file = os.path.join(cfg.logger.path, 'log.txt')
    os.makedirs(cfg.logger.path, exist_ok=True)
    cfg.freeze()
    SEED = cfg.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return cfg

if __name__ == "__main__":
    parser = ArgumentParser("enas")
    parser.add_argument("--config_file", default="./configs/search.yaml", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = setup_cfg(args)
    logger = MyLogger('EnasSkin', cfg).getlogger()
    logger.info(args)
    logger.info(cfg)
    with open(os.path.join(cfg.logger.path, 'cfg_search.yaml'), 'w') as f:
        f.write(str(cfg))

    dataset_train, dataset_valid = build_dataset(cfg)
    model = build_model(cfg)
    logger.info('Cell choices: {}'.format(model.layers[0].nodes[0].cell_x.op_choice.choices))
    mutator = build_mutator(model, cfg)
    loss_function = build_loss_fn(cfg)
    OPTIM = cfg.optim
    if cfg.optim.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), cfg.optim.base_lr, momentum=OPTIM.momentum, weight_decay=OPTIM.weight_decay, nesterov=True)
    elif cfg.optim.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), cfg.optim.base_lr, weight_decay=OPTIM.weight_decay)
    else:
        raise NotImplementedError
    if cfg.optim.scheduler.name.lower() == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.optim.scheduler.milestones)
    elif cfg.optim.scheduler.name.lower() == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optim.scheduler.t_max, eta_min=1e-5)
    else:
        raise NotImplementedError

    if cfg.trainer.name == 'EnasTrainer':
        trainer = EnasTrainer(
            cfg=cfg,
            model=model,
            mutator=mutator,
            loss=loss_function,
            metrics=metrics,
            reward_function=reward_function,
            optimizer=optimizer,
            callbacks=[LRSchedulerCallback(lr_scheduler),
                       ArchitectureCheckpoint(cfg.logger.path),
                       CheckpointCallback(cfg.logger.path, name='best_search.pth', mode=cfg.callback.checkpoint.mode)],
            dataset_train=dataset_train,
            dataset_valid=dataset_valid,
            debug=args.debug)
    elif cfg.trainer.name == 'RandomTrainer':
        trainer = RandomTrainer(
            cfg=cfg,
            model=model,
            mutator=mutator,
            loss=loss_function,
            metrics=metrics,
            reward_function=reward_function,
            optimizer=optimizer,
            callbacks=[LRSchedulerCallback(lr_scheduler),
                       ArchitectureCheckpoint(cfg.logger.path),
                       CheckpointCallback(cfg.logger.path, optimizer, save_flag=True, cfg=cfg)],
            dataset_train=dataset_train,
            dataset_valid=dataset_valid,
            debug=args.debug)
    else:
        raise NotImplementedError
    trainer.train()
