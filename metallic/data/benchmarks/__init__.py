from typing import Tuple
from .omniglot import omniglot_benchmark
from .miniimagenet import miniimagenet_benchmark
from ..datasets import MetaDataset

__all__ = ['list_benchmarks', 'get_benchmarks']

_BENCHMARKS = {
    'omniglot': omniglot_benchmark,
    'miniimagenet': miniimagenet_benchmark
}

def get_benchmarks(
    name: str,
    root: str,
    n_way: int = 5,
    k_shot: int = 1,
    **kwargs
) -> Tuple[MetaDataset]:
    """
    Return a most commonly used benchmark on the given dataset.

    Args:
        name (str): Name of the dataset
        root (str): Root directory of the dataset
        n_way (int): Number of the classes per tasks (same in 'trian' / 'val'
            / 'test' set)
        k_shot (int): Number of samples per class (same in 'support' / 'query'
            set)
        **kwargs: Other arguments if needed
    """
    train_dataset, val_dataset, test_dataset = _BENCHMARKS[name](
        root = root,
        n_way = n_way,
        k_shot = k_shot,
        **kwargs
    )
    return train_dataset, val_dataset, test_dataset

def list_benchmarks():
    """
    Return a list of available benchmarks.
    """
    return _BENCHMARKS.keys()
