# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np
import torch

import nni


class MyLogger(object):
    def __init__(self, name, cfg=None):
        self.file = cfg.logger.log_file if cfg is not None else 'log.txt'
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if (self.logger.hasHandlers()):
            self.logger.handlers.clear()
        formatter = logging.Formatter('[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')

        hdlr = logging.FileHandler(self.file, 'a', encoding='utf-8')
        hdlr.setLevel(logging.INFO)
        hdlr.setFormatter(formatter)
        self.logger.addHandler(hdlr)

        strhdlr = logging.StreamHandler()
        strhdlr.setLevel(logging.INFO)
        strhdlr.setFormatter(formatter)
        self.logger.addHandler(strhdlr)

        hdlr.close()
        strhdlr.close()

    def getlogger(self):
        return self.logger

def calc_real_model_size(model, mutator):
    '''calculate the size of real model
        real_size = size_choice + size_non_choice
    '''
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    else:
        model = model
    size_choice = 0 # the size of LayerChoice
    size_non_choice = 0 # the size of normal model part

    # the size of normal model part
    for name, param in model.named_parameters():
        if 'choice' not in name:
            size_non_choice += param.numel()

    # the mask for each operation, the masks looks like as follows:
    # {'normal_node_0_x_op': tensor([False, True, False, False, False, False]),
    #  'normal_node_0_y_op': tensor([True, False, False, False, False, False]),
    #  ...
    masks = {}
    for key in mutator._cache:
        if 'op' in key:
            masks[key] = mutator._cache[key]


    # the real size of all LayerChoice
    # for mutable in mutator.mutables:
    #     if isinstance(mutable, LayerChoice):
    for name, module in model.named_modules():
        if isinstance(module, nni.nas.pytorch.mutables.LayerChoice):
            size_ops = []
            for index, op in enumerate(module.choices):
                size_ops.append(sum([p.numel() for p in op.parameters()]))

            # parse the key for masks, which is needed to modified for different model.
            infos = name.split('.') # name=''layers.0.nodes.0.cell_x.op_choice''
            node_id = infos[3] # 0-4
            cell_id = infos[4][-1] # x or y
            prefix = 'normal'
            key = f"{prefix}_node_{node_id}_{cell_id}_op" # normal_node_0_x_op
            index = masks[key].int().argmax()
            size_choice += size_ops[index]
    real_size = size_choice + size_non_choice
    return real_size

def metrics(outputs, targets, topk=(1, 3)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res["acc{}".format(k)] = correct_k.mul_(1.0 / batch_size).item()
    return res

def reward_function(outputs, targets, topk=(1,)):
    batch_size = targets.size(0)
    _, predicted = torch.max(outputs.data, 1)
    return (predicted == targets).sum().item() / batch_size
