import random
from itertools import combinations
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from .datasets import MetaDataset


class MetaSequentialSampler(SequentialSampler):
    def __init__(self, data_source: MetaDataset):
        super(MetaSequentialSampler, self).__init__(data_source)

    def __iter__(self):
        n_classes = self.data_source.n_classes
        n_way = self.data_source.n_way
        return combinations(range(n_classes), n_way)


class MetaRandomSampler(RandomSampler):
    def __init__(self, data_source: MetaDataset):
        super(MetaRandomSampler, self).__init__(data_source, replacement=True)

    def __iter__(self):
        n_classes = self.data_source.n_classes
        n_way = self.data_source.n_way
        for _ in combinations(range(n_classes), n_way):
            yield tuple(random.sample(range(n_classes), n_way))
