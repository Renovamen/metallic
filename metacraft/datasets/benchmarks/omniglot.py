import os
import json
import glob
import h5py
from PIL import Image, ImageOps
from torchvision.datasets.utils import list_dir, download_url

from metacraft.datasets.utils import load_splits
from metacraft.datasets.transforms import EncodeTarget
from metacraft.datasets.benchmarks.base import Dataset, ConcatTask, ClassDataset, MetaDataset


class Omniglot(MetaDataset):
    '''
    The Omniglot dataset. 

    attributes:
        root (string):
            Root directory where the dataset folder `omniglot` exists.
        
        n_way (int):
            Number of classes per tasks. This corresponds to "N" in "N-way" 
            classification.
        
        meta_split (str):
            'train', 'val', 'test', name of the split to use.
        
        use_vinyals_split (bool, default = True):
            If set to `True`, the dataset uses the splits defined in [3]. If `False`, 
            then the meta-train split corresponds to `images_background`, and the 
            meta-test split corresponds to `images_evaluation` (raises an error when 
            calling the meta-validation split).
        
        transform (callable, optional):
            A function/transform that takes a `PIL` image, and returns a transformed 
            version. See also `torchvision.transforms`.

        dataset_transform (callable, optional):
            A function/transform that takes a dataset (ie. a task), and returns a 
            transformed version of it. E.g. `MetaSplitter()`.
        
        class_augmentations (list of callable, optional):
            A list of functions that augment the dataset with new classes. These classes 
            are transformations of existing classes. See `datasets.transforms.augmentations`.
    '''
    def __init__(self, root, n_way, meta_split = None, use_vinyals_split = True, 
                 transform = None, dataset_transform = None, class_augmentations = None):
        
        dataset = OmniglotClassDataset(
            root = root, 
            meta_split = meta_split,
            use_vinyals_split = use_vinyals_split,
            transform = transform,
            class_augmentations = class_augmentations
        )

        target_transform = EncodeTarget(n_way)
        
        super(Omniglot, self).__init__(
            dataset = dataset,
            n_way = n_way,
            target_transform = target_transform,
            dataset_transform = dataset_transform
        )


class OmniglotClassDataset(ClassDataset):
    
    dataset_name = 'omniglot'

    output_name = 'data.hdf5'
    output_name_labels = '{0}{1}_labels.json'

    folder_split = [
        ('images_background', 'train'),
        ('images_evaluation', 'test')
    ]

    def __init__(self, root, meta_split = None, use_vinyals_split = True, 
                 transform = None, class_augmentations = None):
        
        super(OmniglotClassDataset, self).__init__(
            meta_split = meta_split,
            class_augmentations = class_augmentations
        )

        if self.meta_split == 'val' and (not use_vinyals_split):
            raise ValueError('You must set `use_vinyals_split = True` to use the meta-validation split.')

        self.root = os.path.expanduser(root)
        self.use_vinyals_split = use_vinyals_split
        self.transform = transform

        self.output_path = os.path.join(self.root, self.output_name)
        self.output_path_labels = os.path.join(
            self.root,
            self.output_name_labels.format(
                'vinyals_' if use_vinyals_split else '', 
                self.meta_split
            )
        )

        self._data = None
        self._labels = None

        # preprocess the images along with their labels and store them locally
        self.preprocess()

        if not self._check_integrity():
            raise RuntimeError('Omniglot integrity check failed')
        
        self._n_classes = len(self.labels)


    def __getitem__(self, index):
        class_name = '/'.join(self.labels[index % self.n_classes])  # images_?/alphabet?/character?
        data = self.data[class_name]  # all images under class `class_name`
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return OmniglotDataset(
            index, data, class_name,
            transform = transform, 
            target_transform = target_transform
        )


    # number of classes in this dataset
    @property
    def n_classes(self):
        return self._n_classes


    # load images from .hdf5
    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.output_path, 'r')
        return self._data


    # load labels from .json
    @property
    def labels(self):
        if self._labels is None:
            with open(self.output_path_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels


    # check if the preprocessed files have been created
    def _check_integrity(self):
        return (
            os.path.isfile(self.output_path)
            and os.path.isfile(self.output_path_labels)
        )


    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None


    # preprocess the images along with their labels and store them locally
    def preprocess(self):
        # check if the output files have already been created
        if self._check_integrity():
            return

        # save images to HDF5 file, and their labels to JSON files
        with h5py.File(self.output_path, 'w') as f:

            for folder_name, split in self.folder_split:
                group = f.create_group(folder_name)

                alphabets = list_dir(os.path.join(self.root, folder_name))
                characters = [
                    (folder_name, alphabet, character)
                    for alphabet in alphabets
                    for character in list_dir(os.path.join(self.root, folder_name, alphabet))
                ]

                # save labels to JSON files
                output_path_labels = os.path.join(self.root, self.output_name_labels.format('', split))
                with open(output_path_labels, 'w') as f_labels:
                    labels = sorted(characters)
                    json.dump(labels, f_labels)

                '''
                save images to HDF5 file

                ├── image_background/
                │   ├── alphabet1/character1 (class1)
                │   ├── alphabet1/character2 (class2)
                │   ...
                │   └── alphabetn/characterm (classz)
                └── image_evaluation/
                    ├── ...
                    ...
                '''
                for _, alphabet, character in characters:
                    image_paths = glob.glob(os.path.join(self.root, folder_name, alphabet, character, '*.png'))
                    # each character (independent of alphabet) is a class (dataset)
                    dataset = group.create_dataset(
                        '{0}/{1}'.format(alphabet, character), 
                        (len(image_paths), 105, 105), 
                        dtype = 'uint8'
                    )
                    for i, impath in enumerate(image_paths):
                        image = Image.open(impath, mode = 'r').convert('L')
                        dataset[i] = ImageOps.invert(image)

        # Vinyal's split
        for split in ['train', 'val', 'test']:
            output_path_labels = os.path.join(self.root, self.output_name_labels.format('vinyals_', split))
            data = load_splits(self.dataset_name, '{0}.json'.format(split))

            with open(output_path_labels, 'w') as f:
                labels = sorted([
                    ('images_{0}'.format(name), alphabet, character)
                    for (name, alphabets) in data.items()
                    for (alphabet, characters) in alphabets.items()
                    for character in characters
                ])
                json.dump(labels, f)


class OmniglotDataset(Dataset):
    def __init__(self, index, data, class_name, transform = None, target_transform = None):
        super(OmniglotDataset, self).__init__(
            index = index, 
            transform = transform,
            target_transform = target_transform
        )
        self.data = data
        self.class_name = class_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)