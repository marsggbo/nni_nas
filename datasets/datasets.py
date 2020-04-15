# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision.datasets import MNIST as _MNIST

from .build import DATASET_REGISTRY
from .transforms import build_transforms

__all__ = [
    'MNIST',
    'CIFAR10',
    'FakeData',
    'fakedata'
]

@DATASET_REGISTRY.register()
def MNIST(cfg):
    cfg.defrost()
    root = cfg.dataset.datapath

    datasets = []
    for is_train in [True, False]:
        cfg.dataset.is_train = is_train
        transform = build_transforms(cfg)
        datasets.append(_MNIST(root=root, train=is_train,
                        transform=transform.transform, download=True))
    dataset_train, dataset_valid = datasets
    cfg.freeze()
    return dataset_train, dataset_valid

@DATASET_REGISTRY.register()
def CIFAR10(cfg):
    cfg.defrost()
    root = cfg.dataset.datapath

    datasets = []
    for is_train in [True, False]:
        cfg.dataset.is_train = is_train
        transform = build_transforms(cfg)
        datasets.append(_CIFAR10(root=root, train=is_train,
                        transform=transform.transform, download=True))
    dataset_train, dataset_valid = datasets
    cfg.freeze()
    return dataset_train, dataset_valid

class FakeData(torch.utils.data.Dataset):
    def __init__(self, size=32, classes=10):
        self.size = size
        self.data = torch.rand(3, size, size)
        self.labels = 1

    def get_img_size(self):
        return self.size, self.size

    def get_scale_size(self):
        return self.size, self.size

    def __getitem__(self, index):
        return self.data, self.labels

    def __len__(self):
        return 10

@DATASET_REGISTRY.register()
def fakedata(cfg):
    classes = cfg.model.classes
    size = cfg.input.size
    if isinstance(size, list) or isinstance(size, tuple):
        size = size[0]
    assert isinstance(size, int), "the type of data size must be integer."
    dataset_train = FakeData(size, classes)
    dataset_valid = FakeData(size, classes)
    return dataset_train, dataset_valid
