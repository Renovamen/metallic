from torchvision.transforms import Compose, Resize, ToTensor
from ..datasets import Omniglot
from ..transforms import Rotation

def omniglot_benchmark(
    root: str, n_way: int = 5, k_shot: int = 1, **kwargs
) -> Omniglot:
    transform = Compose([Resize(28), ToTensor()])
    augmentations = [Rotation(90), Rotation(180), Rotation(270)]

    train_dataset = Omniglot(
        root = root,
        n_way = n_way,
        meta_split = 'train',
        k_shot_support = k_shot,
        k_shot_query = k_shot,
        transform = transform,
        augmentations = augmentations,
        **kwargs
    )
    val_dataset = Omniglot(
        root = root,
        n_way = n_way,
        meta_split = 'val',
        k_shot_support = k_shot,
        k_shot_query = k_shot,
        transform = transform,
        augmentations = augmentations,
        **kwargs
    )
    test_dataset = Omniglot(
        root = root,
        n_way = n_way,
        meta_split = 'test',
        k_shot_support = k_shot,
        k_shot_query = k_shot,
        transform = transform,
        augmentations = augmentations,
        **kwargs
    )

    return train_dataset, val_dataset, test_dataset
