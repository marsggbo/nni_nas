# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
import sys
sys.path.append('..')

import torch
import torch.optim as optim

from kd_model import load_kd_model, loss_fn_kd
from trainer import TRAINER_REGISTRY
from utils import AverageMeterGroup, reward_function

from .build import TRAINER_REGISTRY
from .default_trainer import Trainer


__all__ = [
    'EnasTrainer',
]


@TRAINER_REGISTRY.register()
class EnasTrainer(Trainer):
    def __init__(self, cfg):
        """Initialize an EnasTrainer.
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
        cfg.defrost()
        cfg.mutator.name = 'EnasMutator'
        cfg.freeze()
        super(EnasTrainer, self).__init__(cfg)
        trainer_cfg = cfg.trainer.EnasTrainer

        self.debug = cfg.debug
        self.warm_start_epoch = cfg.trainer.warm_start_epoch

        self.mutator_optim = optim.Adam(self.mutator.parameters(), lr=trainer_cfg.mutator_lr)
        self.entropy_weight = trainer_cfg.entropy_weight
        self.skip_weight = trainer_cfg.skip_weight
        self.baseline_decay = trainer_cfg.baseline_decay
        self.baseline = 0.
        self.mutator_steps_aggregate = trainer_cfg.mutator_steps_aggregate
        self.mutator_steps = 2 if self.debug else trainer_cfg.mutator_steps

        # preparing dataset
        n_train = len(self.dataset_train)
        split = n_train // 10
        indices = list(range(n_train))
        random.shuffle(indices)
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:-split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[-split:])
        self.train_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=self.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=self.workers,
                                                        pin_memory=True)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_train,
                                                        batch_size=self.batch_size,
                                                        sampler=valid_sampler,
                                                        num_workers=self.workers,
                                                        pin_memory=True)
        self.test_loader = torch.utils.data.DataLoader(self.dataset_valid,
                                                       batch_size=self.batch_size,
                                                       num_workers=self.workers,
                                                       pin_memory=True)
        self.num_batches_per_epoch = len(self.train_loader)

        if hasattr(self.cfg, 'kd') and self.cfg.kd.enable:
            self.kd_model = load_kd_model(self.cfg).to(self.device)
            if len(self.cfg.trainer.device_ids) > 1:
                self.kd_model = torch.nn.DataParallel(self.kd_model, device_ids=device_ids)
                self.kd_model.eval()
        else:
            self.kd_model = None

    def reward_function(self):
        return reward_function

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

            # record loss and other metrics
            metrics['loss'] = loss.item()
            meters.update(metrics)

            loss.backward()
            if (step+1) % self.cfg.trainer.accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.log_frequency is not None and step % self.log_frequency == 0:
                self.logger.info("Model Epoch [{}/{}] Step [{}/{}] Model size: {} {}".format(
                    epoch + 1, self.num_epochs, step + 1, len(self.train_loader), self.model_size(), meters))

        # warm up training
        if self.warm_start_epoch >= epoch:
            return meters

        # Train sampler (mutator)
        self.model.eval()
        self.mutator.train()
        meters = AverageMeterGroup()
        mutator_step, total_mutator_steps = 0, self.mutator_steps * self.mutator_steps_aggregate
        while mutator_step < total_mutator_steps:
            for step, sample_batched in enumerate(self.valid_loader):
                if self.debug and step > 0:
                    break
                inputs, targets = sample_batched
                inputs = inputs.cuda()
                targets = targets.cuda()

                self.mutator.reset()
                with torch.no_grad():
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
                    reward = self.reward_function(output, targets) * self.cfg.mutator.EnasMutator.reward_weight

                if self.entropy_weight is not None:
                    reward += self.entropy_weight * self.mutator.sample_entropy
                self.baseline = self.baseline * self.baseline_decay + reward * (1 - self.baseline_decay)
                self.baseline = self.baseline.detach().item()
                loss = self.mutator.sample_log_prob * (reward - self.baseline)
                if self.skip_weight:
                    loss += self.skip_weight * self.mutator.sample_skip_penalty
                loss += self.cfg.mutator.EnasMutator.arch_loss_weight * self.model_size()
                metrics["reward"] = reward.item()
                metrics["loss"] = loss.item()
                metrics["ent"] = self.mutator.sample_entropy.item()
                metrics["baseline"] = self.baseline
                metrics["skip"] = self.mutator.sample_skip_penalty

                loss = loss / self.mutator_steps_aggregate
                loss.backward()
                meters.update(metrics)

                if mutator_step % self.mutator_steps_aggregate == 0:
                    self.mutator_optim.step()
                    self.mutator_optim.zero_grad()

                if self.log_frequency is not None and mutator_step % self.log_frequency == 0:
                    self.logger.info("RL Epoch [%s/%s] Step [%s/%s]  %s", epoch + 1, self.num_epochs,
                                mutator_step+1, total_mutator_steps, meters)
                mutator_step += 1
                if mutator_step >= total_mutator_steps:
                    break

            if self.debug:
                break

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

            metrics = self.metrics(output, targets)

            # record loss and other metrics
            metrics['loss'] = loss.item()
            meters.update(metrics)

            if self.log_frequency is not None and step % self.log_frequency == 0:
                self.logger.info("Test: Step [{}/{}]  {}".format(step + 1, len(self.test_loader), meters))
        self.logger.info("Finwal model metric = {}".format(meters.meters['save_metric'].avg))
