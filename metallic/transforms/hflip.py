import torch
import torchvision.transforms.functional as TF

class HorizontalFlip:
    """
    Horizontal flip the given PIL Image or torch Tensor.
    """
    def __call__(self, image: torch.Tensor):
        return TF.hflip(image)
