import os
import sys
from collections import OrderedDict, defaultdict
from typing import Optional, Callable, Iterator, List, Tuple, Any
from itertools import combinations
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import ConcatDataset, Subset

class Dataset(TorchDataset):
    """
    A dataset containing all of the samples from a given class:

    .. code-block::

        Dataset (a class)
        ├─────────┬─────────┐
        │         │         │
        sample1   sample2   ...
    """

    def __init__(
        self, index, data, class_label, transform=None, target_transform=None
    ) -> None:
        self.index = index
        self.data = data
        self.class_label = class_label
        self.transform = transform
        self.target_transform = target_transform

    def __hash__(self) -> int:
        return hash(self.index)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        image, target = self.data[index], self.class_label

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)


class ClassDataset:
    """
    Base class for a dataset composed of classes. Each item from a ``ClassDataset``
    is a ``Dataset`` containing samples from the given class:

    .. code-block::

        ClassDataset
        ├───────────────┬──────────────┐
        │               │              │
        class1          class2         ... (`Dataset`)
        ├─────────┬─────────┐
        │         │         │
        sample1   sample2   ...
    """

    def __init__(
        self,
        root: str,
        meta_split: str,
        cache_path: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ) -> None:
        if meta_split not in ['train', 'val', 'test']:
            raise ValueError(
                'Unknown meta-split name `{0}`. The meta-split '
                'must be in [`train`, `val`, `test`].'.format(meta_split)
            )
        self.meta_split = meta_split
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self._prepro_cache = os.path.join(self.root, cache_path)

    def __len__(self) -> int:
        return self.n_classes

    def preprocess(self) -> None:
        if self._check_cache():
            self.load_cache()
        else:
            self.create_cache()
            # self.save_cache()
        self.n_classes = len(self.labels[self.meta_split])

    def _check_cache(self) -> bool:
        """
        Check if cache file exists.
        """
        return os.path.isfile(self._prepro_cache)

    def create_labels(self) -> None:
        raise NotImplementError()

    def create_cache(self) -> None:
        """
        Iterates over the entire dataset and creates a map of target to samples
        from scratch.
        """
        print('Cache not found, creating from scratch...')
        self.create_labels()
        self.label_to_images = defaultdict(list)

        for (image, label) in self.dataset:
            self.label_to_images[label].append(image)

    def save_cache(self) -> None:
        state = {
            'label_to_images': self.label_to_images,
            'labels': self.labels
        }
        print('Saving cache to {}'.format(self._prepro_cache))
        torch.save(state, self._prepro_cache)

    def load_cache(self) -> None:
        """
        Load map of target to samples from cache.
        """
        print('Loading cache from {}'.format(self._prepro_cache))
        state = torch.load(self._prepro_cache)
        self.label_to_images = state['label_to_images']
        self.labels = state['labels']

    def __getitem__(self, index) -> Dataset:
        class_label = self.labels[self.meta_split][index % self.n_classes]
        data = self.label_to_images[class_label]

        return Dataset(
            index, data, class_label,
            transform = self.transform,
            target_transform = self.target_transform
        )


class TaskDataset(ConcatDataset):
    """
    A dataset for concatenating the given multiple classes, which means:

    .. code-block::

        TaskDataset
        ├────────┬────────┬────────┬────────┐
        │        │        │        │        │
        c1_s1    c1_s2    ...      c2_s1    ...
    """
    def __init__(self, datasets: Dataset, n_classes: int) -> None:
        super(TaskDataset, self).__init__(datasets)
        self.n_classes = n_classes

    def __getitem__(self, index: int) -> tuple:
        return ConcatDataset.__getitem__(self, index)


class SubTaskDataset(Subset):
    """
    Subset of a ``TaskDataset`` at specified indices.
    """
    def __init__(
        self, dataset: Dataset, indices: List[int], n_classes: int = None
    ) -> None:
        super(SubTaskDataset, self).__init__(dataset, indices)
        if n_classes is None:
            n_classes = dataset.n_classes
        self.n_classes = n_classes

    def __getitem__(self, index: int) -> tuple:
        return Subset.__getitem__(self, index)


class MetaDataset(TorchDataset):
    """
    A dataset for fast indexing of samples within classes.
    """
    def __init__(
        self,
        dataset: ClassDataset,
        n_way: int,
        k_shot_support: Optional[int] = None,
        k_shot_query: Optional[int] = None,
        shuffle: bool = True
    ) -> None:
        self.dataset = dataset
        self.n_way = n_way
        self.n_classes = len(dataset)

        # split a task into support / query set or not
        if (k_shot_support is None) or (k_shot_query is None):
            self.task_splits = None
        else:
            self.task_splits = OrderedDict()
            self.task_splits["support"] = k_shot_support
            self.task_splits["query"] = k_shot_query
            self._min_samples_per_class = sum(self.task_splits.values())
            self.shuffle = shuffle

    def split_task(self, task: TaskDataset) -> OrderedDict:
        """
        Split a ``TaskDataset`` into support / query set, each of ther set
        contains ``k_shot_suppor`` / ``k_shot_query`` samples per class.
        """
        indices = OrderedDict([(split, []) for split in self.task_splits])
        cumulative_size = 0

        # get indices of samples that will be wrapped in each split
        for dataset in task.datasets:
            n_samples = len(dataset)
            if n_samples < self._min_samples_per_class:
                raise ValueError(
                    'The number of samples for one class ({0}) '
                    'should be greater than the minimum number of'
                    'samples per class ({1}).'.format(
                        n_samples, self._min_samples_per_class
                    )
                )

            # shuffle samples in the current class before split
            if self.shuffle:
                seed = hash(task) % (2 ** 32)
                class_sample_indices = np.random.RandomState(seed).permutation(n_samples)
            else:
                class_sample_indices = np.arange(n_samples)

            # wrap k_shot samples for each class in split
            st = 0
            for split, k_shot in self.task_splits.items():
                split_indices = class_sample_indices[st : st + k_shot]
                indices[split].extend(
                    np.add(split_indices, cumulative_size).tolist()
                )
                st += k_shot
            cumulative_size += n_samples

        return OrderedDict([
            (split, SubTaskDataset(task, indices[split]))
            for split in self.task_splits
        ])

    def __getitem__(self, indices: tuple) -> TaskDataset:
        """
        Generate a task composed with the given ``n_way`` classes.

        Args:
            indices (tuple): The ``n_way`` indices of the classes to be
                sampled in the task.
        """
        # make sure the number of classes is equal to n_way
        assert len(indices) == self.n_way
        # get selected classes
        classes = [self.dataset[i] for i in indices]
        # concatenate them
        task = TaskDataset(
            datasets = classes,
            n_classes = self.n_way
        )

        # split the task into support / query set if needed
        if self.task_splits is not None:
            task = self.split_task(task)

        return task

    def __iter__(self) -> Iterator:
        """
        Iterate all possible tasks composed of ``n_way`` classes.
        """
        for index in combinations(range(self.n_classes), self.n_way):
            yield self[index]

    def __len__(self) -> int:
        """
        Number of all possible tasks composed of ``n_way`` classes.
        """
        length = 1
        # combination formula
        for i in range(1, self.n_way + 1):
            length *= (self.n_classes - i + 1) // i
        # the length exceeds machine precision
        if length > sys.maxsize:
            length = sys.maxsize
        return length
