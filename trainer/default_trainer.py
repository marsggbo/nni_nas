# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
from abc import abstractmethod

import numpy as np
import torch

from nni.nas.pytorch.callbacks import ArchitectureCheckpoint
from nni_nas.callbacks import CheckpointCallback
from nni_nas.datasets import build_dataset
from nni_nas.losses import build_loss_fn
from nni_nas.mutator import build_mutator
from nni_nas.networks import build_model
from nni_nas.utils import (MyLogger, calc_real_model_size, generate_optimizer,
                           generate_scheduler, metrics,
                           parse_cfg_for_scheduler)

from .base_trainer import BaseTrainer

__all__ = [
    'TorchTensorEncoder',
    'Trainer'
]


class TorchTensorEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=method-hidden
        if isinstance(o, torch.Tensor):
            olist = o.tolist()
            if "bool" not in o.type().lower() and all(map(lambda d: d == 0 or d == 1, olist)):
                print("Every element in %s is either 0 or 1. "
                                "You might consider convert it into bool.", olist)
            return olist
        return super().default(o)


class Trainer(BaseTrainer):
    def __init__(self, cfg, **kwargs):
        """
        Trainer initialization.

            Parameters
            ----------
                model : nn.Module
                    Model with mutables.
                mutator : BaseMutator
                    A mutator object that has been initialized with the model.
                loss : callable
                    Called with logits and targets. Returns a loss tensor.
                metrics : callable
                    Returns a dict that maps metrics keys to metrics data.
                optimizer : Optimizer
                    Optimizer that optimizes the model.
                num_epochs : int
                    Number of epochs of training.
                dataset_train : torch.utils.data.Dataset
                    Dataset of training.
                dataset_valid : torch.utils.data.Dataset
                    Dataset of validation/testing.
                batch_size : int
                    Batch size.
                workers : int
                    Number of workers used in data preprocessing.
                device : torch.device
                    Device object. Either ``torch.device("cuda")`` or ``torch.device("cpu")``. When ``None``, trainer will
                    automatic detects GPU and selects GPU first.
                log_frequency : int
                    Number of mini-batches to log metrics.
                callbacks : list of Callback
                    Callbacks to plug into the trainer. See Callbacks.
        """
        self.cfg = cfg
        self.logger = MyLogger(__name__, cfg).getlogger()
        self.set_up()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.mutator.to(self.device)
        self.loss.to(self.device)

        self.callbacks = self.generate_callbacks()
        for callback in self.callbacks:
            callback.build(self.model, self.mutator, self)

    def set_up(self):
        # model
        self.model = build_model(self.cfg)
        self.logger.info('Cell choices: {}'.format(self.model.layers[0].nodes[0].cell_x.op_choice.choices))
        self.logger.info(f"Building model {self.cfg.model.name} ...")

        # mutator
        # self.logger.info('Cell choices: {}'.format(model.layers[0].nodes[0].cell_x.op_choice.choices))
        self.mutator = build_mutator(self.model, self.cfg)
        self.logger.info(f"Building mutator {self.cfg.mutator.name} ...")

        # dataset
        self.batch_size = self.cfg.dataset.batch_size
        self.workers = self.cfg.dataset.workers
        self.dataset_train, self.dataset_valid = build_dataset(self.cfg)
        self.logger.info(f"Building dataset {self.cfg.dataset.name} ...")

        # loss
        self.loss = build_loss_fn(self.cfg)
        self.logger.info(f"Building loss function {self.cfg.loss.name} ...")

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
            self.optimizer,
            self.cfg.optim.scheduler.name,
            **self.scheduler_params)
        self.logger.info(f"Building optimizer scheduler {self.cfg.optim.scheduler.name} ...")

        # miscellaneous
        self.num_epochs = self.cfg.trainer.num_epochs
        self.log_frequency = self.cfg.logger.log_frequency
        self.start_epoch = 0
   
    @abstractmethod
    def train_one_epoch(self, epoch):
        pass

    @abstractmethod
    def validate_one_epoch(self, epoch):
        pass

    def train(self, validate=True):
        self.resume()
        for epoch in range(self.start_epoch, self.num_epochs):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # training
            self.logger.info("Epoch {} Training".format(epoch))
            meters = self.train_one_epoch(epoch)
            self.logger.info("Final training metric: {}".format(meters))

            if validate:
                # validation
                self.logger.info("Epoch {} Validatin".format(epoch))
                self.validate_one_epoch(epoch)

            for callback in self.callbacks:
                if isinstance(callback, CheckpointCallback):
                    callback.update_best_metric(meters.meters['save_metric'].avg)
                callback.on_epoch_end(epoch)

    def validate(self):
        self.validate_one_epoch(-1)

    def export(self, file):
        mutator_export = self.mutator.export()
        with open(file, "w") as f:
            json.dump(mutator_export, f, indent=2, sort_keys=True, cls=TorchTensorEncoder)

    def checkpoint(self):
        raise NotImplementedError("Not implemented yet")

    def model_size(self):
        return calc_real_model_size(self.model, self.mutator)*4/1024**2

    def generate_callbacks(self):
        '''
        Args：
            func: a function to generate other callbacks, must return a list
        Return:
            a list of callbacks.
        '''
        self.ckpt_callback = CheckpointCallback(
            checkpoint_dir=self.cfg.logger.path,
            name='best_search.pth',
            mode=self.cfg.callback.checkpoint.mode)
        self.arch_callback = ArchitectureCheckpoint(self.cfg.logger.path)
        callbacks = [self.ckpt_callback, self.arch_callback]
        return callbacks
    
    def metrics(self, *args, **kwargs):
        return metrics( *args, **kwargs)

    def resume(self):
        self.best_metric = -999
        path = self.cfg.model.resume_path
        if path:
            assert os.path.exists(path), "{} does not exist".format(path)
            ckpt = torch.load(path)
            self.start_epoch = ckpt['epoch'] + 1
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.mutator.load_state_dict(ckpt['mutator_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
            self.best_metric = ckpt['best_metric']
            self.logger.info('Resuming training from epoch {}'.format(self.start_epoch))

        for callback in self.callbacks:
            if isinstance(callback, CheckpointCallback):
                if self.best_metric == -999:
                    self.best_metric = callback.best_metric
                else:
                    callback.best_metric = self.best_metric

        if len(self.cfg.trainer.device_ids) > 1:
            device_ids = self.cfg.trainer.device_ids
            num_gpus_available = torch.cuda.device_count()
            assert num_gpus_available >= len(device_ids), "you can only use {} device(s)".format(num_gpus_available)
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            # self.mutator = torch.nn.DataParallel(self.mutator, device_ids=device_ids) # mutator doesn't support dataparallel yet.
