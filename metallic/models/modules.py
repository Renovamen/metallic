from typing import Union, Tuple, Optional
from collections import OrderedDict
import torch
from torch import nn

class Flatten(nn.Module):
    """
    Flatten input tensor to ``(batch_size, -1)`` shape.
    """
    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), -1)


class ConvBlock(nn.Module):
    """
    A base convolutional block.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image

    out_channels : int
        Number of channels produced by the convolution

    kernel_size : int or tuple, optional, default=3
        Size of the convolving kernel

    stride : int or tuple, optional, default=1
        Stride of the convolution

    pool_kernel_size : int or tuple, optional, default=2)
        ``kernel_size`` of the max pooling layer

    dropout : float, optional
        Dropout, ``None`` if no dropout layer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        pool_kernel_size: Union[int, Tuple[int]] = 2,
        dropout: Optional[float] = None
    ):
        super(ConvBlock, self).__init__()

        module_list = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = 1,
                bias = True
            ),  # (batch_size, out_channels, img_size, img_size)
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(pool_kernel_size),  # (batch_size, out_channels, img_size / 2, img_size / 2)
            nn.ReLU()
        ]

        if dropout:
            module_list.append(nn.Dropout(dropout))

        self.core = nn.Sequential(*module_list)
        self.init_weights()

    def init_weights(self):
        # conv layer
        nn.init.xavier_uniform_(self.core[0].weight.data, gain = 1.0)
        nn.init.constant_(self.core[0].bias.data, 0.0)
        # batch normalization layer
        nn.init.uniform_(self.core[1].weight)

    def forward(self, x: torch.Tensor):
        output = self.core(x)
        return output


class ConvGroup(nn.Module):
    """
    A base convolutional group.

    Parameters
    ----------
    in_channels : int, optional, default=1
        Number of channels in the input image

    hidden_size : int, optional, default=64
        Dimensionality of the hidden representation

    kernel_size : int or tuple, optional, default=3
        Size of the convolving kernel

    stride : int or tuple, optional, default=1
        Stride of the convolution

    pool_kernel_size : int or tuple, optional, default=2)
        ``kernel_size`` of the max pooling layer

    dropout : float, optional
        dropout, ``None`` if no dropout layer

    layers : int, optional, default=4
        Number of convolutional layers

    Notes
    -----
    - Omniglot: hidden_size=64, in_channels=1, pool=False
    - MiniImagenet: hidden_size=32, in_channels=3, pool=True
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_size: int = 64,
        kernel_size: Union[int, Tuple[int]] = 3,
        stride: Union[int, Tuple[int]] = 1,
        pool_kernel_size: Union[int, Tuple[int]] = 2,
        dropout: Optional[float] = None,
        layers: int = 4
    ):
        super(ConvGroup, self).__init__()

        module_list = [
            ConvBlock(
                in_channels = in_channels,
                out_channels = hidden_size,
                kernel_size = kernel_size,
                stride = stride,
                pool_kernel_size = pool_kernel_size,
                dropout = dropout
            )
        ]

        for _ in range(layers - 1):
            module_list.append(
                ConvBlock(
                    in_channels = hidden_size,
                    out_channels = hidden_size,
                    kernel_size = kernel_size,
                    stride = stride,
                    pool_kernel_size = pool_kernel_size,
                    dropout = dropout
                )
            )

        self.core = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor):
        output = self.core(x)
        return output


class LinearBlock(nn.Module):
    """
    A base linear block.

    Parameters
    ----------
    input_size : int
        Size of each input sample

    output_size : int
        Size of each output sample
    """

    def __init__(self, in_features: int, out_features: int):
        super(LinearBlock, self).__init__()
        self.core = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
        self.init_weights()

    def init_weights(self):
        # linear layer
        nn.init.xavier_uniform_(self.core[0].weight)
        nn.init.constant_(self.core[0].bias.data, 0.0)

    def forward(self, x: torch.Tensor):
        output = self.core(x)
        return output
