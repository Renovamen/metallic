from typing import Callable
from torchvision.transforms import Compose

def compose_transform(transform1: Callable, transform2: Callable) -> Callable:
    """
    Composed another two transform functions.
    """
    if transform2 is None:
        new_transform = transform1
    else:
        new_transform = Compose([transform1, transform2])
    return new_transform
