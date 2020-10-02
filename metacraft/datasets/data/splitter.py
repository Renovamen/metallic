import torch
import numpy as np
from collections import OrderedDict, defaultdict

from metacraft.datasets.benchmarks.base import Task, ConcatTask, SubsetTask


class MetaSplitter(object):
    '''
    Transforms a dataset into support / query splits, based on a fixed 
    number of samples per class for each split.

    This is a dataset transformation to be applied as a `dataset_transform`
    in a `MetaDataset`.

    attributes:
        shuffle (bool, default = True):
            Shuffle the data in the dataset before the split.

        k_shot_support (int):
            Number of samples per class in the support split. This corresponds
            to "K" in "K-shot" learning.

        k_shot_query (int):
            Number of samples per class in the query split.

        random_state_seed (int, optional):
            seed of the np.RandomState. Defaults to '0'.
    '''
    def __init__(self, shuffle = True, k_shot_support = None, 
                 k_shot_query = None, random_state_seed = 0):
    
        self.shuffle = shuffle

        self.splits = OrderedDict()
        self.splits['support'] = k_shot_support
        self.splits['query'] = k_shot_query

        self._min_samples_per_class = sum(self.splits.values())

        self.random_state_seed = random_state_seed
        self.seed(random_state_seed)


    def seed(self, seed):
        self.np_random = np.random.RandomState(seed = seed)


    def get_indices(self, task):
        if isinstance(task, ConcatTask):
            indices = self.get_indices_concattask(task)
        elif isinstance(task, Task):
            indices = self.get_indices_task(task)
        else:
            raise ValueError(
                'The task must be of type `ConcatTask` or `Task`, '
                'Got type `{0}`.'.format(type(task))
            )
        return indices


    def get_indices_task(self, task):
        all_class_indices = self._get_class_indices(task)
        indices = OrderedDict([(split, []) for split in self.splits])

        for name, class_indices in all_class_indices.items():
            n_samples = len(class_indices)
            if n_samples < self._min_samples_per_class:
                raise ValueError(
                    'The number of samples for class `{0}` ({1}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `ClassSplitter` ({2}).'.format(
                        name, n_samples, self._min_samples_per_class
                    )
                )

            if self.shuffle:
                seed = (hash(task) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(n_samples)
            else:
                dataset_indices = np.arange(n_samples)

            ptr = 0
            for split, n_split in self.splits.items():
                split_indices = dataset_indices[ptr:ptr + n_split]
                if self.shuffle:
                    self.np_random.shuffle(split_indices)
                indices[split].extend([class_indices[idx] for idx in split_indices])
                ptr += n_split

        return indices


    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0

        for dataset in task.datasets:
            n_samples = len(dataset)
            if n_samples < self._min_samples_per_class:
                raise ValueError(
                    'The number of samples for one class ({0}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `ClassSplitter` ({1}).'.format(
                        n_samples, self._min_samples_per_class
                    )
                )

            if self.shuffle:
                seed = (hash(task) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(n_samples)
            else:
                dataset_indices = np.arange(n_samples)

            ptr = 0
            for split, n_split in self.splits.items():
                split_indices = dataset_indices[ptr:ptr + n_split]
                if self.shuffle:
                    self.np_random.shuffle(split_indices)
                indices[split].extend(split_indices + cum_size)
                ptr += n_split
            cum_size += n_samples

        return indices
    

    def _get_class_indices(self, task):
        class_indices = defaultdict(list)
        if task.n_classes is None: # Regression task
            class_indices['regression'] = range(len(task))
        else:
            for index in range(len(task)):
                sample = task[index]
                if (not isinstance(sample, tuple)) or (len(sample) < 2):
                    raise ValueError(
                        'In order to split the dataset in train/'
                        'test splits, `Splitter` must access the targets. Each '
                        'sample from a task must be a tuple with at least 2 '
                        'elements, with the last one being the target.'
                    )
                class_indices[sample[-1]].append(index)

            if len(class_indices) != task.n_classes:
                raise ValueError(
                    'The number of classes detected in `Splitter` '
                    '({0}) is different from the property `n_classes` ({1}) '
                    'in task `{2}`.'.format(len(class_indices), task.n_classes, task)
                )

        return class_indices


    def __call__(self, task):
        indices = self.get_indices(task)
        return OrderedDict([
            (split, SubsetTask(task, indices[split]))
            for split in self.splits
        ])


    def __len__(self):
        return len(self.splits)