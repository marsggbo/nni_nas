# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torchline
from torch.utils.tensorboard import SummaryWriter

from configs import get_cfg, add_config
from datasets import build_dataset
from kd_model import load_kd_model, loss_fn_kd
from losses import build_loss_fn
from networks import build_model
from callbacks import CheckpointCallback
import nni
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeter
from utils import (MyLogger, calc_real_model_size, metrics, mixup_loss_fn,
                   mixup_data)

global logger
global writter


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

class Evaluator(BaseEvaluator):
    def __init__(self, cfg, arc_path, callbacks=[]):
        super(Evaluator, self).__init__()
        self.cfg = cfg
        if not callbacks:
            self.callbacks = [self.ckpt_callback]
        else:
            self.callbacks = callbacks
        self.arcs = self.load_arcs(arc_path)
        self.writter = SummaryWriter(os.path.join(cfg.logger.path, 'summary_runs'))
        self.logger = MyLogger(__name__, cfg).getlogger()
        self.size_acc = {} # {'epoch1': [model_size, acc], 'epoch2': [model_size, acc], ...}
        self.init_basic_settings()

    def init_basic_settings(self):
        '''init train_epochs, device, loss_fn, dataset, and dataloaders
        '''
        # train epochs
        try:
            self.train_epochs = cfg.args.train_epochs
        except:
            self.train_epochs = 1

        # CheckpointCallback
        self.ckpt_callback = CheckpointCallback(
            self.cfg.logger.path,
            'best_retrain.pth',
            mode=self.cfg.callback.checkpoint.mode)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # loss_fn
        self.loss_fn = build_loss_fn(cfg)
        self.loss_fn.to(self.device)
        self.logger.info(f"Building loss function ...")

        # dataset
        self.train_dataset, self.test_dataset = build_dataset(cfg)

        # dataloader
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=cfg.dataset.batch_size, 
            shuffle=True, num_workers=cfg.dataset.workers, pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=cfg.dataset.batch_size, 
            shuffle=False, num_workers=cfg.dataset.workers, pin_memory=True)
        self.logger.info(f"Building dataset and dataloader ...")

    def load_arcs(self, arc_path):
        '''load arch json files
        Args:
            arc_path:
                (file): [arc_path]
                (dir): [arc_path/epoch_0.json, arc_path/epoch_1.json, ...]
        '''
        if os.path.isfile(arc_path):
            return [arc_path]
        else:
            arcs = os.listdir(arc_path)
            arcs = [os.path.join(arc_path, arc) for arc in arcs 
                    if arc.split('.')[-1]=='json']
            arcs = sorted(arcs, 
                          key=lambda x: int( os.path.splitext( os.path.basename(x) )[0].split('_')[1] ) )
            return arcs

    def reset(self):
        '''mutable can be only initialized for once, hence it needs to
        reset model, optimizer, scheduler when run a new trial.
        '''
        # model
        self.model = build_model(self.cfg)
        self.model.to(self.device)
        self.logger.info(f"Building model {self.cfg.model.name} ...")

        # optimizer
        self.optimizer = generate_optimizer(
            model=self.model,
            optim_name=self.cfg.optim.name,
            lr=self.cfg.optim.base_lr,
            momentum=self.cfg.optim.momentum,
            weight_decay=self.cfg.optim.weight_decay)
        self.logger.info(f"Building optimizer {self.cfg.optim.name} ...")

        # scheduler
        self.scheduler_params = parse_cfg_for_scheduler(self.cfg, self.cfg.optim.scheduler.name)
        self.lr_scheduler = generate_scheduler(
            optimizer,
            self.cfg.optim.scheduler.name,
            **self.scheduler_params)
        self.logger.info(f"Building optim.scheduler {self.cfg.optim.scheduler.name} ...")

    def compare(self):
        self.enable_writter = False
        # split train dataset into train and valid dataset
        train_size = int(0.8 * len(self.train_dataset))
        valid_size = len(self.train_dataset) - train_size
        self.train_dataset_part, self.valid_dataset_part = torch.utils.data.random_split(
            self.train_dataset, [train_size, valid_size])

        # dataloader
        self.train_loader_part = torch.utils.data.DataLoader(
            self.train_dataset_part, batch_size=cfg.dataset.batch_size, 
            shuffle=True, num_workers=cfg.dataset.workers, pin_memory=True)
        self.valid_loader_part = torch.utils.data.DataLoader(
            self.valid_dataset_part, batch_size=cfg.dataset.batch_size, 
            shuffle=True, num_workers=cfg.dataset.workers, pin_memory=True)

        # choose the best architecture
        for arc in self.arcs:
            self.reset()
            self.mutator = apply_fixed_architecture(self.model, arc)
            size = calc_real_model_size(self.model, self.mutator)
            arc_name = os.path.basename(arc)
            self.logger.info(f"{arc} Model size={size*4/1024**2} MB")

            # train
            for epoch in range(self.train_epochs):
                self.train_one_epoch(epoch, self.train_loader_part)
            val_acc = self.validate(-1, self.valid_loader_part)
            self.size_acc[arc_name] = {'size': size,
                                       'val_acc': val_acc,
                                       'arc': arc}
        sorted_size_acc = sorted(self.size_acc.items(),
                                 key=lambda x: x[1]['val_acc'], reverse=True)
        return sorted_size_acc[0][1]

    def run(self, arc, validate=True):
        '''retrain the best-performing arch from scratch
            arc: the json file path of the best-performing arch 
        '''
        self.enable_writter = True
        self.reset()

        # init model and mutator
        self.mutator = apply_fixed_architecture(self.model, arc)
        size = calc_real_model_size(self.model, self.mutator)
        arc_name = os.path.basename(arc)
        self.logger.info(f"{arc_name} Model size={size*4/1024**2} MB")

        # callbacks
        for callback in self.callbacks:
            callback.build(self.model, self.mutator, self)

        # resume
        start_epoch, self.best_metric = self.resume()

        # fintune
        # todo： improve robustness, bug of optimizer resume
        # if self.cfg.model.finetune:
        #     self.logger.info("Freezing params of conv part ...")
        #     for name, param in self.model.named_parameters():
        #         if 'dense' not in name:
        #             param.requires_grad = False

        # load teacher model if using knowledge distillation
        if hasattr(self.cfg, 'kd') and self.cfg.kd.enable:
            self.kd_model = load_kd_model(self.cfg).to(self.device)
            self.kd_model.eval()
        else:
            self.kd_model = None

        # dataparallel
        if len(self.cfg.trainer.device_ids) > 1:
            device_ids = cfg.trainer.device_ids
            num_gpus_available = torch.cuda.device_count()
            assert num_gpus_available >= len(device_ids), "you can only use {} device(s)".format(num_gpus_available)
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            if self.kd_model:
                self.kd_model = torch.nn.DataParallel(self.kd_model, device_ids=device_ids)

        # start training
        for epoch in range(start_epoch, self.cfg.trainer.num_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            self.logger.info("Epoch %d Training", epoch)
            self.train_one_epoch(epoch)

            if validate:
                self.logger.info("Epoch %d Validating", epoch)
                top1 = self.validate_one_epoch(epoch)

            self.lr_scheduler.step()

            for callback in self.callbacks:
                callback.on_epoch_end(epoch)

        self.logger.info("Final best Prec@1 = {:.4%}".format(self.best_metric))

    def train_one_epoch(self, epoch, dataloader):
        config = self.cfg
        top1 = AverageMeter("top1")
        top3 = AverageMeter("top3")
        losses = AverageMeter("losses")

        cur_lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info("Epoch %d LR %.6f", epoch, cur_lr)
        if self.enable_writter:
            self.writter.add_scalar("lr", cur_lr, global_step=cur_step)

        self.model.train()

        for step, (x, y) in enumerate(dataloader):
            if config.args.debug and step > 1:
                break
            for callback in self.callbacks:
                callback.on_batch_begin(epoch)
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            bs = x.size(0)
            # mixup data
            if config.mixup.enable:
                x, y_a, y_b, lam = mixup_data(x, y, config.mixup.alpha)
                mixup_y = [y_a, y_b, lam]

            logits = self.model(x)
            if isinstance(logits, tuple):
                logits, aux_logits = logits
                if config.mixup.enable:
                    aux_loss = mixup_loss_fn(loss_fn, aux_logits, *mixup_y)
                else:
                    aux_loss = loss_fn(aux_logits, y)
            else:
                aux_loss = 0.
            if config.mixup.enable:
                loss = mixup_loss_fn(loss_fn, logits, *mixup_y)
            else:
                loss = loss_fn(logits, y)
            if config.model.aux_weight > 0:
                loss += config.model.aux_weight * aux_loss
            if self.kd_model is not None:
                teacher_output = self.kd_model(x)
                loss += (1-config.kd.loss.alpha)*loss + loss_fn_kd(logits, teacher_output, cfg.kd.loss)

            loss.backward()
            # gradient clipping
            # nn.utils.clip_grad_norm_(model.parameters(), 20)

            if (step+1) % config.trainer.accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            accuracy = metrics(logits, y, topk=(1, 3))
            losses.update(loss.item(), bs)
            top1.update(accuracy["acc1"], bs)
            top3.update(accuracy["acc3"], bs)
            if self.enable_writter:
                self.writter.add_scalar("loss/train", loss.item(), global_step=epoch)
                self.writter.add_scalar("acc1/train", accuracy["acc1"], global_step=epoch)
                self.writter.add_scalar("acc3/train", accuracy["acc3"], global_step=epoch)

            if step % config.logger.log_frequency == 0 or step == len(self.train_loader) - 1:
                self.logger.info(
                    "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top3.avg:.1%})".format(
                        epoch + 1, config.trainer.num_epochs, step, len(self.train_loader) - 1, losses=losses,
                        top1=top1, top3=top3))

            for callback in self.callbacks:
                callback.on_batch_end(epoch)
        self.logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(
            epoch + 1, config.trainer.num_epochs, top1.avg))

    def valid_one_epoch(self, epoch, dataloader):
        config = self.cfg
        top1 = AverageMeter("top1")
        top3 = AverageMeter("top3")
        losses = AverageMeter("losses")

        self.model.eval()

        with torch.no_grad():
            for step, (X, y) in enumerate(dataloader):
                if config.args.debug and step > 1:
                    break
                X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                bs = X.size(0)

                logits = self.model(X)
                if isinstance(logits, tuple):
                    logits, aux_logits = logits
                    aux_loss = self.loss_fn(aux_logits, y)
                else:
                    aux_loss = 0.
                loss = self.loss_fn(logits, y)
                if config.model.aux_weight > 0:
                    loss = loss + config.model.aux_weight * aux_loss

                accuracy = metrics(logits, y, topk=(1, 3))
                losses.update(loss.item(), bs)
                top1.update(accuracy["acc1"], bs)
                top3.update(accuracy["acc3"], bs)

                if step % config.logger.log_frequency == 0 or step == len(self.valid_loader) - 1:
                    self.logger.info(
                        "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top3.avg:.1%})".format(
                            epoch + 1, config.trainer.num_epochs, step, len(self.valid_loader) - 1, 
                            losses=losses, top1=top1, top3=top3))

        if self.enable_writter and epoch > 0:
            self.writter.add_scalar("loss/test", losses.avg, global_step=epoch)
            self.writter.add_scalar("acc1/test", top1.avg, global_step=epoch)
            self.writter.add_scalar("acc3/test", top3.avg, global_step=epoch)

        self.logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.trainer.num_epochs, top1.avg))

        return top1.avg

    def resume(self, mode=True):
        path = cfg.model.resume_path
        if path:
            assert os.path.exists(path), "{} does not exist".format(path)
            ckpt = torch.load(path)
            start_epoch = ckpt['epoch'] + 1
            try:
                self.model.load_state_dict(ckpt['model_state_dict'])
            except:
                self.logger.info('Loading from DataParallel model...')
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in ckpt['model_state_dict'].items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.model.load_state_dict(new_state_dict)
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            self.logger.info('Resuming training from epoch {}'.format(start_epoch))
            return ckpt['epoch'], ckpt['best_metric']
        else:
            if self.mode: # the more the better, e.g. acc
                self.best_metric = -1. * np.inf
            else: # the less, the better, e.g. epe
                self.best_metric = np.inf
            return 0, self.best_metric


if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--config_file", default='./config/skin10.yaml', type=str)
    parser.add_argument("--train_epochs", default=2, type=int)
    parser.add_argument("--arc_path", default="./outputs/checkpoints_0")
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
    with open(os.path.join(cfg.logger.path, 'cfg_train.yaml'), 'w') as f:
        f.write(str(cfg))
    cfg.update({'args': args})
    logger = MyLogger(__name__, cfg).getlogger()
    writer = SummaryWriter(os.path.join(cfg.logger.path, 'summary_runs'))
    self.logger.info('args:{}'.format(args))

    evaluator = Evaluator(cfg, args.arc_path)
    
    if os.path.isdir(arc_path):
        best_arch_info = evaluator.compare()
        evaluator.run(best_arch_info['arc'])
    elif os.path.isfile(arc_path):
        evaluator.run(arc_path)
