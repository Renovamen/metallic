from typing import Union
import torch
import torchvision.transforms.functional as TF

class Rotation:
    """
    Rotate the the given PIL Image or torch Tensor by the given angle.

    Args:
        angle (int): Rotation angle
        resample (int, optional, default=False): An optional resampling filter.
            See `filter <https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters>`_
            for more information. If omitted, or if the image has mode “1” or
            “P”, it is set to PIL.Image.NEAREST. If input is Tensor, only
            ``PIL.Image.NEAREST`` and ``PIL.Image.BILINEAR`` are supported.
        expand (bool, optional, default=False): Optional expansion flag. If
            true, expands the output to make it large enough to hold the
            entire rotated image. If false or omitted, make the output image
            the same size as the input image. Note that the expand flag assumes
            rotation around the center and no translation.
        center (list or tuple, optional, default=False): Optional center of
            rotation, (x, y). Origin is the upper left corner. Default is the
            center of the image.
    """

    def __init__(
        self,
        angle: int,
        resample: int = False,
        expand: bool = False,
        center: Union[list, tuple] = None
    ) -> None:
        super(Rotation, self).__init__()
        self.angle = angle % 360
        self.resample = resample
        self.expand = expand
        self.center = center

    def __call__(self, image: torch.Tensor):
        return TF.rotate(
            image,
            angle = self.angle,
            resample = self.resample,
            expand = self.expand,
            center = self.center
        )
