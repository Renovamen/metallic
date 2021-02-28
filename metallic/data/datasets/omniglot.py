import os
from collections import defaultdict
from typing import Callable, Optional, Dict
import torch
from torch.utils.data import ConcatDataset
from torchvision.datasets.omniglot import Omniglot as TorchOmniglot

from .base import ClassDataset, MetaDataset
from .. import _utils

class OmniglotClassDataset(ClassDataset):
    dataset_name = 'omniglot'
    cache_path = 'omniglot_cache.pth.tar'

    def __init__(
        self,
        root: str,
        meta_split: str = 'train',
        use_vinyals_split: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ) -> None:
        super(OmniglotClassDataset, self).__init__(
            root = root,
            meta_split = meta_split,
            cache_path = self.cache_path,
            transform = transform,
            target_transform = target_transform
        )

        if self.meta_split == 'val' and (not use_vinyals_split):
            raise ValueError(
                'You must set `use_vinyals_split = True` to use the'
                'meta-validation split.'
            )
        if use_vinyals_split:
            self.meta_split = "vinyals_{}".format(meta_split)

        self.use_vinyals_split = use_vinyals_split

        self.omniglot = {}
        # background set
        self.omniglot["background"] = TorchOmniglot(
            root = self.root,
            background = True,
            download = download
        )
        # evaluation set, labels start after background set
        self.omniglot["evaluation"] = TorchOmniglot(
            root = self.root,
            background = False,
            download = download,
            target_transform = lambda x: x + len(self.omniglot["background"]._characters)
        )
        # combine them
        self.dataset = ConcatDataset((self.omniglot["background"], self.omniglot["evaluation"]))
        self.preprocess()

    def create_labels(self) -> None:
        """
        Create a list of labels for each split.
        """
        self.labels = {}

        # eval / background split
        get_name = {
            "train": "background",
            "test": "evaluation"
        }
        for name in ["train", "test"]:
            label_list = [label for (_, label) in self.omniglot[get_name[name]]]
            self.labels[name] = list(set(label_list))

        # Vinyals' split
        file_to_label = _file_to_label(self.omniglot)
        for name in ["train", "val", "test"]:
            split_name = "vinyals_{}".format(name)
            split = _utils.splits.load_splits(self.dataset_name, '{0}.json'.format(name))
            self.labels[split_name] = sorted([
                file_to_label["/".join([name, alphabet, character])]
                for (name, alphabets) in split.items()
                for (alphabet, characters) in alphabets.items()
                for character in characters
            ])


class Omniglot(MetaDataset):
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
        download: bool = False
    ) -> None:
        dataset = OmniglotClassDataset(
            root = root,
            meta_split = meta_split,
            use_vinyals_split = use_vinyals_split,
            transform = transform,
            target_transform = target_transform,
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
        "background": 0,
        "evaluation": len(data["background"]._characters)
    }
    for name in ["background", "evaluation"]:
        for (image, label) in data[name]:
            filename = "/".join([name, data[name]._characters[label - start[name]]])
            file_to_label[filename] = label
    return file_to_label
