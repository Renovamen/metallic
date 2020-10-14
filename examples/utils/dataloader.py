from torchvision.transforms import Compose, Resize, ToTensor

from metacraft.datasets.benchmarks import Omniglot
from metacraft.datasets.transforms import Rotation
from metacraft.datasets.data import MetaSplitter, BatchMetaDataLoader


def load_data(config):
    
    dataset_transform = MetaSplitter(
        shuffle = True,
        k_shot_support = config.k_shot, 
        k_shot_query = config.k_shot
    )

    if config.dataset == 'omniglot':
        transform = Compose([Resize(28), ToTensor()])
        class_augmentations = [Rotation([90, 180, 270])]

        train_dataset = Omniglot(
            root = config.dataset_path,
            n_way = config.n_way,
            meta_split = 'train',
            transform = transform,
            dataset_transform = dataset_transform,
            class_augmentations = class_augmentations
        )

        val_dataset = Omniglot(
            root = config.dataset_path,
            n_way = config.n_way,
            meta_split = 'val',
            transform = transform,
            dataset_transform = dataset_transform,
            class_augmentations = class_augmentations
        )

        test_dataset = Omniglot(
            root = config.dataset_path,
            n_way = config.n_way,
            meta_split = 'test',
            transform = transform,
            dataset_transform = dataset_transform,
            class_augmentations = class_augmentations
        )

    else:
        raise NotImplementedError("Dataset not implemented.")
        
    train_loader = BatchMetaDataLoader(
        dataset = train_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        num_workers = config.num_workers
    )

    val_loader = BatchMetaDataLoader(
        dataset = val_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        num_workers = config.num_workers
    )

    test_loader = BatchMetaDataLoader(
        dataset = test_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        num_workers = config.num_workers
    )

    return train_loader, val_loader, test_loader