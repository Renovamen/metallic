import torch
import torchvision.transforms.functional as TF

class VerticalFlip:
    """
    Vertically flip the given PIL Image or torch Tensor.
    """
    def __call__(self, image: torch.Tensor):
        return TF.vflip(image)
