import random
import warnings
from itertools import combinations
from torch.utils.data.sampler import SequentialSampler as TorchSequentialSampler
from torch.utils.data.sampler import RandomSampler as TorchRandomSampler

from metacraft.datasets.benchmarks.base import MetaDataset


class SequentialSampler(TorchSequentialSampler):
    def __init__(self, data_source):
        super(SequentialSampler, self).__init__(data_source)

    def __iter__(self):
        n_classes = len(self.data_source.dataset)
        n_classes_per_task = self.data_source.n_classes_per_task
        return combinations(range(n_classes), n_classes_per_task)


class RandomSampler(TorchRandomSampler):
    def __init__(self, data_source):
        # Temporarily disable the warning if the length of the length of the 
        # dataset exceeds the machine precision. This avoids getting this
        # warning shown with MetaDataLoader, even though MetaDataLoader itself
        # does not use the length of the dataset.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super(RandomSampler, self).__init__(data_source, replacement = True)

    def __iter__(self):
        n_classes = len(self.data_source.dataset)
        n_way = self.data_source.n_way
        for _ in combinations(range(n_classes), n_way):
            yield tuple(random.sample(range(n_classes), n_way))