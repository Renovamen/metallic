from typing import Optional, Callable
from torch.utils.data import DataLoader, Sampler

from .datasets import MetaDataset
from .sampler import *
from . import _utils

class MetaDataLoader(DataLoader):
    def __init__(
        self,
        dataset: MetaDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0.,
        worker_init_fn = Callable[[int], None]
    ) -> None:

        collate_fn = _utils.MetaCollate()

        if sampler is None:
            if shuffle:
                sampler = MetaRandomSampler(dataset)
            else:
                sampler = MetaSequentialSampler(dataset)
            shuffle = False

        super(MetaDataLoader, self).__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            sampler = sampler,
            batch_sampler = batch_sampler,
            num_workers = num_workers,
            collate_fn = collate_fn,
            pin_memory = pin_memory,
            drop_last = drop_last,
            timeout = timeout,
            worker_init_fn = worker_init_fn
        )
