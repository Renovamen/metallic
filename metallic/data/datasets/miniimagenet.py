import os
from typing import Callable, Optional, Tuple, Any, List
from collections import defaultdict
import pickle
from PIL import Image
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Dataset as TorchDataset
from torchvision.datasets.utils import download_file_from_google_drive, \
    extract_archive, check_integrity

from .base import ClassDataset, MetaDataset

class MiniImageNetClassDataset(ClassDataset):
    """
    A dataset composed of classes from mini-ImageNet.

    Args:
        root (str): Root directory of dataset
        meta_split (str, optional, default='train'): Name of the split to
            be used: 'train' / 'val' / 'test
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

    dataset_name = 'mini-imagenet'
    google_drive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
    zip_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
    pkl_name = 'mini-imagenet-cache-{0}.pkl'

    def __init__(
        self,
        root: str,
        meta_split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augmentations: List[Callable] = None,
        download: bool = False
    ) -> None:
        super(MiniImageNetClassDataset, self).__init__(
            root = os.path.join(root, self.dataset_name),
            meta_split = meta_split,
            cache_path = self.dataset_name + '_cache.pth.tar',
            transform = transform,
            target_transform = target_transform,
            augmentations = augmentations
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.preprocess()

    def _check_integrity(self) -> bool:
        return check_integrity(os.path.join(self.root, self.dataset_name + '.tar.gz'), self.zip_md5)

    def download(self) -> None:
        """Download file from Google drive."""
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        filename = self.dataset_name + '.tar.gz'
        download_file_from_google_drive(
            file_id = self.google_drive_id,
            root = self.root,
            filename = filename,
            md5 = self.zip_md5
        )

        archive = os.path.join(self.root, filename)
        print("Extracting {} to {}".format(archive, self.root))
        extract_archive(archive, self.root)

    def create_cache(self) -> None:
        self.labels = {}
        self.label_to_images = defaultdict(list)
        cumulative_size = 0

        for split in ["train", "val", "test"]:
            pkl_path = os.path.join(self.root, self.pkl_name.format(split))
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                images, targets = data['image_data'], data['class_dict']
                n_classes = len(targets)

                categorical = (torch.randperm(len(targets)) + cumulative_size).tolist()
                to_categorical = dict(
                    (target, categorical[i])
                    for (i, target) in enumerate(list(targets.keys()))
                )
                self.labels[split] = categorical

                for label, indices in targets.items():
                    self.label_to_images[to_categorical[label]] = [
                        Image.fromarray(image)
                        for image in images[indices]
                    ]
            cumulative_size += n_classes


class MiniImageNet(MetaDataset):
    """
    The mini-ImageNet dataset introduced in [1]. It samples 100 classed from
    ImageNet (ILSVRC-2012), in which 64 for training, 16 for validation, and
    20 for testing. Each of the class contains 600 samples.

    The dataset is downloaded from `here <https://github.com/renmengye/few-shot-ssl-public/>`_.

    NOTE:
        [1] didn't released their splits at first, so [2] created their own
        splits. Here we use the splits from [2].

    Args:
        root (str): Root directory of dataset
        n_way (int): Number of the classes per tasks
        meta_split (str, optional, default='train'): Name of the split to
            be used: 'train' / 'val' / 'test
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

    References
    ----------
    1. "`Matching Networks for One Shot Learning. <https://arxiv.org/abs/1606.04080>`_" Oriol Vinyals, et al. NIPS 2016.
    2. "`Optimization as a Model for Few-Shot Learning. <https://openreview.net/pdf?id=rJY0-Kcll>`_" Sachin Ravi, et al. ICLR 2017.
    """

    def __init__(
        self,
        root: str,
        n_way: int,
        meta_split: str = 'train',
        k_shot_support: Optional[int] = None,
        k_shot_query: Optional[int] = None,
        shuffle: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augmentations: List[Callable] = None,
        download: bool = False
    ) -> None:
        dataset = MiniImageNetClassDataset(
            root = root,
            meta_split = meta_split,
            transform = transform,
            target_transform = target_transform,
            augmentations = augmentations,
            download = download
        )

        super(MiniImageNet, self).__init__(
            dataset = dataset,
            n_way = n_way,
            k_shot_support = k_shot_support,
            k_shot_query = k_shot_query,
            shuffle = shuffle
        )
