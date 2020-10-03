import sys
import numpy as np
from copy import deepcopy
from itertools import combinations
from ordered_set import OrderedSet
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import ConcatDataset, Subset
from torchvision.transforms import Compose

from metacraft.datasets.transforms import FixTargetTransform, EncodeTarget


class Dataset(TorchDataset):
    '''
    Dataset (a class)
    ├─────────┬─────────┐
    │         │         │
    sample1   sample2   ...

    attributes:
        index (int): index of this class
    '''

    def __init__(self, index, transform = None, target_transform = None):
        self.index = index
        self.transform = transform
        self.target_transform = target_transform

    def target_transform_append(self, transform):
        '''
        Compose multiple target transforms.
        '''
        if transform is None:
            return
        if self.target_transform is None:
            self.target_transform = transform
        else:
            self.target_transform = Compose([self.target_transform, transform])

    def __hash__(self):
        return hash(self.index)


class Task(Dataset):
    '''
    Base class for a classification task.

    attributes:
        n_classes (int):
            Number of classes for the classification task.
    '''

    def __init__(self, index, n_classes, transform = None, target_transform = None):
        super(Task, self).__init__(index, transform = transform,
                                   target_transform = target_transform)
        self.n_classes = n_classes


class ConcatTask(Task, ConcatDataset):
    '''
    A classification task concatenated by `n_classes` classes:

    ConcatTask
    ├────────┬────────┬────────┬────────┐
    │        │        │        │        │
    c1_s1    c1_s2    ...      c2_s1    ...

    attributes:
        datasets (list[class1, class2, ...]): 
            A list of classes (Dataset) to be concatenated into a task.
        
        n_classes (int):
            Number of classes.
    '''

    def __init__(self, datasets, n_classes, target_transform = None):
        
        index = tuple(task.index for task in datasets)

        Task.__init__(self, index, n_classes)
        ConcatDataset.__init__(self, datasets)

        for task in self.datasets:
            task.target_transform_append(target_transform)

    def __getitem__(self, index):
        return ConcatDataset.__getitem__(self, index)


class SubsetTask(Task, Subset):
    def __init__(self, dataset, indices, n_classes = None, target_transform = None):
        
        if n_classes is None:
            n_classes = dataset.n_classes

        Task.__init__(self, dataset.index, n_classes)
        Subset.__init__(self, dataset, indices)

        self.dataset.target_transform_append(target_transform)

    def __getitem__(self, index):
        return Subset.__getitem__(self, index)

    def __hash__(self):
        return hash((self.index, tuple(self.indices)))


class ClassDataset(object):
    '''
    Base class for a dataset of classes. Each item from a `ClassDataset` is 
    a `Dataset` containing examples from the same class:

    ClassDataset
    ├───────────────┬──────────────┐
    │               │              │
    class1          class2         ... (`Dataset`)
    ├─────────┬─────────┐
    │         │         │
    sample1   sample2   ...

    attributes:
        meta_split (str):
            'train' / 'val' / 'test', name of the split to use.
        
        class_augmentations (list of callable, optional):
            A list of functions that augment the dataset with new classes (transformed
            from existing classes). E.g. `transforms.HorizontalFlip()`. 
    '''

    def __init__(self, meta_split = None, class_augmentations = None):
        if meta_split not in ['train', 'val', 'test']:
            raise ValueError(
                'Unknown meta-split name `{0}`. The meta-split '
                'must be in [`train`, `val`, `test`].'.format(meta_split)
            )

        self._meta_split = meta_split

        if class_augmentations is None:
            class_augmentations = []
        # expand class augmentation functions
        else:
            if not isinstance(class_augmentations, list):
                raise TypeError(
                    'Unknown type for `class_augmentations`. '
                    'Expected `list`, got `{0}`.'.format(type(class_augmentations))
                )
            unique_augmentations = OrderedSet()
            for augmentations in class_augmentations:
                for transform in augmentations:
                    unique_augmentations.add(transform)
            class_augmentations = list(unique_augmentations)
        
        self.class_augmentations = class_augmentations

    @property
    def meta_split(self):
        return self._meta_split

    def __getitem__(self, index):
        raise NotImplementedError()

    @property
    def n_classes(self):
        raise NotImplementedError()

    def __len__(self):
        return self.n_classes * (len(self.class_augmentations) + 1)
    
    def get_class_augmentation(self, index):
        transform_index = (index // self.n_classes) - 1
        if transform_index < 0:
            return None
        return self.class_augmentations[transform_index]
    
    def get_transform(self, index, transform = None):
        class_transform = self.get_class_augmentation(index)
        if class_transform is None:
            return transform
        if transform is None:
            return class_transform
        return Compose([class_transform, transform])

    def get_target_transform(self, index):
        class_transform = self.get_class_augmentation(index)
        return FixTargetTransform(class_transform)


class MetaDataset(object):
    '''
    Base class for a meta-dataset.

    attributes:
        dataset (ClassDataset):
            Each item of it is a dataset containing all the examples 
            from the same class.
        
        meta_split (str):
            'train' / 'val' / 'test', name of the split to use.
        
        n_way (int):
            Number of classes per tasks. This corresponds to "N" in "N-way" 
            classification.
        
        target_transform (callable, optional)
            A function/transform that takes a target, and returns a transformed 
            version. See also `torchvision.transforms`.
        
        dataset_transform (callable, optional):
            A function/transform that takes a dataset (ie. a task), and returns
            a transformed version of it. E.g. `MetaSplitter()`.
    '''

    def __init__(self, dataset, n_way, meta_split = None,
                 target_transform = None, dataset_transform = None):
        
        if dataset.meta_split not in ['train', 'val', 'test']:
            raise ValueError('Unknown meta-split name `{0}`. The meta-split '
                'must be in [`train`, `val`, `test`].'.format(meta_split))
        
        self._meta_split = dataset.meta_split

        self.dataset = dataset
        self.n_way = n_way
        self.target_transform = target_transform
        self.dataset_transform = dataset_transform
        self.seed()

    @property
    def meta_split(self):
        return self._meta_split

    def seed(self, seed = None):
        self.np_random = np.random.RandomState(seed = seed)
        # seed the dataset transform
        _seed_dataset_transform(self.dataset_transform, seed = seed)

    # all possible tasks containing `n_way` classes
    def __iter__(self):
        n_classes = len(self.dataset)
        for index in combinations(n_classes, self.n_way):
            yield self[index]

    # sample a task containing `n_way` classes from the dataset
    def sample_task(self):
        index = self.np_random.choice(
            len(self.dataset),
            size = self.n_way,
            replace = False
        ) # index (list: [class_id1, class_id2, ...])
        return self[tuple(index)]


    def __getitem__(self, index):
        '''
        get a task with selected (`n_way`) classes

        input param:
            index (tuple: (class_id1, class_id2, ...)): 
                (`n_way`) indexes of the classes to be sampled in the task
        
        return:
            a task with selected classes
        '''

        assert len(index) == self.n_way
        # selected classes
        datasets = [self.dataset[i] for i in index]

        # label -> idx
        if isinstance(self.target_transform, EncodeTarget):
            self.target_transform.reset()

        task = ConcatTask(
            datasets = datasets, 
            n_classes = self.n_way,
            # use deepcopy to avoid any side effect across tasks
            target_transform = deepcopy(self.target_transform)
        )

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task

    def __len__(self):
        n_classes, length = len(self.dataset), 1
        for i in range(1, self.n_way + 1):
            length *= (n_classes - i + 1) // i

        if length > sys.maxsize:
            warnings.warn(
                'The number of possible tasks in {0} is '
                'combinatorially large (equal to C({1}, {2})), and exceeds '
                'machine precision. Setting the length of the dataset to the '
                'maximum integer value, which undervalues the actual number of '
                'possible tasks in the dataset. Therefore the value returned by '
                '`len(dataset)` should not be trusted as being representative '
                'of the true number of tasks.'.format(self, len(self.dataset), self.n_way), 
                UserWarning, stacklevel = 2
            )
            length = sys.maxsize
        
        return length
    

def _seed_dataset_transform(transform, seed = None):
    if isinstance(transform, Compose):
        for subtransform in transform.transforms:
            _seed_dataset_transform(subtransform, seed = seed)
    elif hasattr(transform, 'seed'):
        transform.seed(seed = seed)