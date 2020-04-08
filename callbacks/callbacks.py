import os
import sys
sys.path.append('..')

import numpy as np
import torch
import torch.nn as nn
from utils import MyLogger

class Callback:
    
    def __init__(self):
        self.model = None
        self.mutator = None
        self.trainer = None

    def build(self, model, mutator, trainer):
        self.model = model
        self.mutator = mutator
        self.trainer = trainer

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        pass

    def on_batch_begin(self, epoch):
        pass

    def on_batch_end(self, epoch):
        pass

class CheckpointCallback(Callback):
    def __init__(self, checkpoint_dir, name, mode=True):
        super(CheckpointCallback, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.mode = self.parse_mode(mode)
        self.name = name
        self.warn_flag = True
        if self.mode: # the more the better, e.g. acc
            self.best_metric = -1. * np.inf
        else: # the less, the better, e.g. epe
            self.best_metric = np.inf

    def parse_mode(self, mode):
        if mode.lower() == 'min':
            return False
        elif mode.lower() == 'max':
            return True
        elif isinstance(mode, bool):
            return mode
        else:
            print(f'''Mode only supports [True / max, False / min], but got {mode, type(mode)}''')
            raise NotImplementedError

    def on_epoch_end(self, epoch, cur_metric):
        if self.mode:
            if cur_metric > self.trainer.best_metric:
                self.trainer.best_metric = cur_metric
                self.save(epoch)
        else:
            if cur_metric < self.trainer.best_metric:
                self.trainer.best_metric = cur_metric
                self.save(epoch)
        self.best_metric = self.trainer.best_metric

    def save(self, epoch):
        model_state_dict = self.get_state_dict(self.model) if hasattr(self.trainer, 'model') else "No state_dict"
        mutator_state_dict = self.get_state_dict(self.mutator) if hasattr(self.trainer, 'mutator') else "No state_dict"
        optimizer_state_dict = self.get_state_dict(self.trainer.optimizer) if hasattr(self.trainer, 'optimizer') else "No state_dict"
        lr_scheduler_state_dict = self.get_state_dict(self.trainer.lr_scheduler) if hasattr(self.trainer, 'lr_scheduler') else "No state_dict"
        dest_path = os.path.join(self.checkpoint_dir, self.name)
        ckpt = {
            'model_state_dict': model_state_dict, # model state_dict
            'mutator_state_dict': mutator_state_dict, # mutator state_dict
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_scheduler_state_dict,
            'epoch': epoch,
            'best_metric': self.trainer.best_metric
        }
        torch.save(ckpt, dest_path)
        self.trainer.logger.info(f"Saving model to {dest_path} at epoch {epoch} with metric {self.trainer.best_metric}")
        self.warn_flag = False

    def get_state_dict(self, module):
        try:
            if isinstance(module, nn.DataParallel):
                state_dict = module.module.state_dict()
            else:
                state_dict = module.state_dict()
        except:
            if self.warn_flag:
                self.trainer.logger.info(f"{module} has no attribution of 'state_dict'")
            state_dict = 'No state_dict'
        return state_dict
