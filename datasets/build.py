import sys
sys.path.append('..')
from registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset, i.e. torch.utils.data.Dataset.

The registered object will be called with `obj(cfg)`
"""

__all__ = [
    'build_dataset',
    'DATASET_REGISTRY'
]

def build_dataset(cfg):
    """
    Built the dataset, defined by `cfg.dataset.name`.
    """
    name = cfg.dataset.name
    return DATASET_REGISTRY.get(name)(cfg)
