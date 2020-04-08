# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import random
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.optim as optim

from mutator import EnasMutator
from nni.nas.pytorch.trainer import Trainer
from nni.nas.pytorch.utils import AverageMeter, AverageMeterGroup
from utils import MyLogger, calc_real_model_size
from kd_model import load_kd_model, loss_fn_kd

__all__ = [
    'RandomTrainer',
]

class RandomTrainer(Trainer):
    def __init__(self, cfg, model, loss, metrics, reward_function, optimizer,
                 dataset_train, dataset_valid, mutator=None, callbacks=None, debug=False):
        """Initialize an RandomTrainer.
            Parameters
            ----------
            model : nn.Module
                PyTorch model to be trained.
            loss : callable
                Receives logits and ground truth label, return a loss tensor.
            metrics : callable
                Receives logits and ground truth label, return a dict of metrics.
            reward_function : callable
                Receives logits and ground truth label, return a tensor, which will be feeded to RL controller as reward.
            optimizer : Optimizer
                The optimizer used for optimizing the model.
            num_epochs : int
                Number of epochs planned for training.
            dataset_train : Dataset
                Dataset for training. Will be split for training weights and architecture weights.
            dataset_valid : Dataset
                Dataset for testing.
            mutator : EnasMutator
                Use when customizing your own mutator or a mutator with customized parameters.
            batch_size : int
                Batch size.
            workers : int
                Workers for data loading.
            device : torch.device
                ``torch.device("cpu")`` or ``torch.device("cuda")``.
            log_frequency : int
                Step count per logging.
            callbacks : list of Callback
                list of callbacks to trigger at events.
            entropy_weight : float
                Weight of sample entropy loss.
            skip_weight : float
                Weight of skip penalty loss.
            baseline_decay : float
                Decay factor of baseline. New baseline will be equal to ``baseline_decay * baseline_old + reward * (1 - baseline_decay)``.
            mutator_lr : float
                Learning rate for RL controller.
            mutator_steps_aggregate : int
                Number of steps that will be aggregated into one mini-batch for RL controller.
            mutator_steps : int
                Number of mini-batches for each epoch of RL controller learning.
            aux_weight : float
                Weight of auxiliary head loss. ``aux_weight * aux_loss`` will be added to total loss.
        """
        device = cfg.trainer.device
        log_frequency = cfg.logger.log_frequency
        batch_size = cfg.dataset.batch_size
        num_epochs = cfg.trainer.num_epochs
        workers = cfg.dataset.workers
        super().__init__(model, mutator if mutator is not None else EnasMutator(model),
                         loss, metrics, optimizer, num_epochs, dataset_train, dataset_valid,
                         batch_size, workers, device, log_frequency, callbacks)

        self.cfg = cfg
        self.debug = debug
        self.logger = MyLogger(__name__, cfg).getlogger()
        self.start_epoch = 0
        self.warm_start_epoch = cfg.trainer.warm_start_epoch
        self.reward_function = reward_function

        # preparing dataset
        n_train = len(self.dataset_train)
        split = n_train // 10
        indices = list(range(n_train))
        random.shuffle(indices)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:-split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[-split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=workers,
                                                        pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=workers,
                                                        pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                       batch_size=batch_size,
                                                       num_workers=workers,
                                                       pin_memory=True)
        self.num_batches_per_epoch = len(self.train_loader)

        if hasattr(self.cfg, 'kd') and self.cfg.kd.enable:
            self.kd_model = load_kd_model(self.cfg).to(self.device)
        else:
            self.kd_model = None

    def resume(self):
        path = self.cfg.model.resume_path
        if path:
            assert os.path.exists(path), "{} does not exist".format(path)
            ckpt = torch.load(path)
            self.start_epoch = ckpt['epoch'] + 1
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            for callback in self.callbacks:
                if hasattr(callback, 'best_metric'):
                    callback.best_metric = ckpt['best_metric']
            self.logger.info('Resuming training from epoch {}'.format(self.start_epoch))

        if len(self.cfg.trainer.device_ids) > 1:
            device_ids = self.cfg.trainer.device_ids
            num_gpus_available = torch.cuda.device_count()
            assert num_gpus_available >= len(device_ids), "you can only use {} device(s)".format(num_gpus_available)
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            if self.kd_model:
                self.kd_model = torch.nn.DataParallel(self.kd_model, device_ids=device_ids)
                self.kd_model.eval()

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
                if hasattr(callback, 'best_metric'):
                    callback.on_epoch_end(epoch, meters.meters['acc1'].avg)
                else:
                    callback.on_epoch_end(epoch)

    def train_one_epoch(self, epoch):
        # Sample model and train
        self.model.train()
        self.mutator.eval()
        meters = AverageMeterGroup()
        for step, sample_batched in enumerate(self.train_loader):
            if self.debug and step > 0:
                break
            inputs, targets = sample_batched
            inputs = inputs.cuda()
            targets = targets.cuda()

            with torch.no_grad():
                self.mutator.reset()

            output = self.model(inputs)
            if isinstance(output, tuple):
                output, aux_output = output
                aux_loss = self.loss(aux_output, targets)
            else:
                aux_loss = 0.
            loss = self.loss(output, targets)
            loss = loss + self.cfg.model.aux_weight * aux_loss
            if self.kd_model:
                teacher_output = self.kd_model(inputs)
                loss = (1-self.cfg.kd.loss.alpha)*loss + loss_fn_kd(output, teacher_output, self.cfg.kd.loss)
            metrics = self.metrics(output, targets)

            # record loss and EPE
            metrics['loss'] = loss.item()
            meters.update(metrics)

            loss.backward()
            if (step+1) % self.cfg.trainer.accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.log_frequency is not None and step % self.log_frequency == 0:
                self.logger.info("Model Epoch [{}/{}] Step [{}/{}] Model size: {} {}".format(
                    epoch + 1, self.num_epochs, step + 1, len(self.train_loader), self.model_size(), meters))

        return meters

    def validate_one_epoch(self, epoch):
        pass

    def test_one_epoch(self, epoch):
        self.model.eval()
        self.mutator.eval()
        meters = AverageMeterGroup()
        for step, sample_batched in enumerate(self.test_loader):
            if self.debug and step > 0:
                break
            inputs, targets = sample_batched
            inputs = inputs.cuda()
            targets = targets.cuda()

            self.optimizer.zero_grad()
            output = self.model(inputs)
            if isinstance(output, tuple):
                output, aux_output = output
                aux_loss = self.loss(aux_output, targets)
            else:
                aux_loss = 0.
            loss = self.loss(output, targets)
            loss = loss + self.cfg.model.aux_weight * aux_loss
            if self.kd_model:
                teacher_output = self.kd_model(inputs)
                loss = (1-self.cfg.kd.loss.alpha)*loss + loss_fn_kd(output, teacher_output, self.cfg.kd.loss)

            metrics = self.metrics(output, targets)

            # record loss and EPE
            metrics['loss'] = loss.item()
            meters.update(metrics)

            if self.log_frequency is not None and step % self.log_frequency == 0:
                self.logger.info("Test: Step [{}/{}]  {}".format(step + 1, len(self.test_loader), meters))
        self.logger.info("Finwal model EPE = {}".format(meters.meters['epe'].avg))

    def get_model(self):
        return self.model.state_dict()

    def model_size(self):
        return calc_real_model_size(self.model, self.mutator)*4/1024**2
