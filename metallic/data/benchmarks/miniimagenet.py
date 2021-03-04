from torchvision.transforms import Compose, Resize, ToTensor
from ..datasets import MiniImageNet
from ..transforms import Rotation

def miniimagenet_benchmark(
    root: str, n_way: int = 5, k_shot: int = 1, **kwargs
) -> MiniImageNet:
    transform = Compose([Resize(84), ToTensor()])

    train_dataset = MiniImageNet(
        root = root,
        n_way = n_way,
        meta_split = 'train',
        k_shot_support = k_shot,
        k_shot_query = k_shot,
        transform = transform,
        **kwargs
    )
    val_dataset = MiniImageNet(
        root = root,
        n_way = n_way,
        meta_split = 'val',
        k_shot_support = k_shot,
        k_shot_query = k_shot,
        transform = transform,
        **kwargs
    )
    test_dataset = MiniImageNet(
        root = root,
        n_way = n_way,
        meta_split = 'test',
        k_shot_support = k_shot,
        k_shot_query = k_shot,
        transform = transform,
        **kwargs
    )

    return train_dataset, val_dataset, test_dataset
