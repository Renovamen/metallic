import os
from collections import defaultdict
from typing import Callable, Optional, Dict, List
from PIL import ImageOps
import torch
from torch.utils.data import ConcatDataset
from torchvision.datasets.omniglot import Omniglot as TorchOmniglot

from .base import ClassDataset, MetaDataset
from .. import _utils

class OmniglotClassDataset(ClassDataset):
    """
    A dataset composed of classes from Omniglot.

    Args:
        root (str): Root directory of dataset
        meta_split (str, optional, default='train'): Name of the split to
            be used: 'train' / 'val' / 'test
        use_vinyals_split (bool, optional, default=True): If ``True``, use
            the splits defined in [2], or use ``images_background`` for train
            split and ``images_evaluation`` for test split.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version
        target_transform (callable, optional): A function/transform that
            takes in the target and transforms it
        augmentations (list of callable, optional):  A list of functions that
            augment the dataset with new classes.
        download (bool, optional, default=False): If true, downloads the dataset
            zip files from the internet and puts it in root directory. If the
            zip files are already downloaded, they are not downloaded again.
    """

    dataset_name = 'omniglot'

    def __init__(
        self,
        root: str,
        meta_split: str = 'train',
        use_vinyals_split: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augmentations: List[Callable] = None,
        download: bool = False
    ) -> None:
        super(OmniglotClassDataset, self).__init__(
            root = root,
            meta_split = meta_split,
            cache_path = self.dataset_name + '_cache.pth.tar',
            transform = transform,
            target_transform = target_transform,
            augmentations = augmentations
        )

        if self.meta_split == 'val' and (not use_vinyals_split):
            raise ValueError(
                'You must set `use_vinyals_split = True` to use the'
                'meta-validation split.'
            )
        if use_vinyals_split:
            self.meta_split = 'vinyals_{}'.format(meta_split)

        self.use_vinyals_split = use_vinyals_split

        self.omniglot = {}
        # background set
        self.omniglot['background'] = TorchOmniglot(
            root = self.root,
            background = True,
            download = download
        )
        # evaluation set, labels start after background set
        self.omniglot['evaluation'] = TorchOmniglot(
            root = self.root,
            background = False,
            download = download,
            target_transform = lambda x: x + len(self.omniglot['background']._characters)
        )
        # combine them
        self.dataset = ConcatDataset((self.omniglot['background'], self.omniglot['evaluation']))
        self.preprocess()

    def create_cache(self) -> None:
        self.labels = {}
        self.label_to_images = defaultdict(list)

        # create a map of target to samples
        for (image, label) in self.dataset:
            self.label_to_images[label].append(ImageOps.invert(image))

        # create a list of labels for each split
        # eval / background split
        get_name = {
            'train': 'background',
            'test': 'evaluation'
        }
        for name in ['train', 'test']:
            label_list = [label for (_, label) in self.omniglot[get_name[name]]]
            self.labels[name] = list(set(label_list))

        # Vinyals' split
        file_to_label = _file_to_label(self.omniglot)
        for name in ['train', 'val', 'test']:
            split_name = 'vinyals_{}'.format(name)
            split = _utils.load_splits(self.dataset_name, '{0}.json'.format(name))
            self.labels[split_name] = sorted([
                file_to_label['/'.join([name, alphabet, character])]
                for (name, alphabets) in split.items()
                for (alphabet, characters) in alphabets.items()
                for character in characters
            ])


class Omniglot(MetaDataset):
    """
    The Omniglot introduced in [1]. It contains 1623 character classes from
    50 different alphabets, each contains 20 samples. The original dataset
    is splited into background (train) and evaluation (test) sets.

    We also provide a choice to use the splits from [2].

    The dataset is downloaded from `here <https://github.com/brendenlake/omniglot>`_,
    and the splits are taken from `here <https://github.com/tristandeleu/pytorch-meta/tree/master/torchmeta/datasets/assets/omniglot>`_.

    Args:
        root (str): Root directory of dataset
        n_way (int): Number of the classes per tasks
        meta_split (str, optional, default='train'): Name of the split to
            be used: 'train' / 'val' / 'test
        use_vinyals_split (bool, optional, default=True): If ``True``, use
            the splits defined in [2], or use ``images_background`` for train
            split and ``images_evaluation`` for test split.
        k_shot_support (int, optional): Number of samples per class in support set
        k_shot_query (int, optional):  Number of samples per class in query set
        shuffle (bool, optional, default=True): If ``True``, samples in a class
            will be shuffled before been splited to support and query set
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version
        target_transform (callable, optional): A function/transform that
            takes in the target and transforms it
        augmentations (list of callable, optional):  A list of functions that
            augment the dataset with new classes.
        download (bool, optional, default=False): If true, downloads the dataset
            zip files from the internet and puts it in root directory. If the
            zip files are already downloaded, they are not downloaded again.

    NOTE:
        ``val`` split is not available when ``use_vinyals_split`` is set to
        ``False``.

    References
    ----------
    1. "`Human-level Concept Learning through Probabilistic Program Induction. <http://www.sciencemag.org/content/350/6266/1332.short>`_" *Brenden M. Lake, et al.* Science 2015.
    2. "`Matching Networks for One Shot Learning. <https://arxiv.org/abs/1606.04080>`_" Oriol Vinyals, et al. NIPS 2016.
    """

    def __init__(
        self,
        root: str,
        n_way: int,
        meta_split: str = 'train',
        use_vinyals_split: bool = True,
        k_shot_support: Optional[int] = None,
        k_shot_query: Optional[int] = None,
        shuffle: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augmentations: List[Callable] = None,
        download: bool = False
    ) -> None:
        dataset = OmniglotClassDataset(
            root = root,
            meta_split = meta_split,
            use_vinyals_split = use_vinyals_split,
            transform = transform,
            target_transform = target_transform,
            augmentations = augmentations,
            download = download
        )

        super(Omniglot, self).__init__(
            dataset = dataset,
            n_way = n_way,
            k_shot_support = k_shot_support,
            k_shot_query = k_shot_query,
            shuffle = shuffle
        )


def _file_to_label(data: dict) -> Dict[str, list]:
    file_to_label = {}
    start = {
        'background': 0,
        'evaluation': len(data['background']._characters)
    }
    for name in ['background', 'evaluation']:
        for (image, label) in data[name]:
            filename = '/'.join([name, data[name]._characters[label - start[name]]])
            file_to_label[filename] = label
    return file_to_label
