# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from .build import DATASET_REGISTRY

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
