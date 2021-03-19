from typing import Optional
import torch
from torch import nn
from .modules import ConvGroup, Flatten, LinearBlock

class OmniglotCNN(nn.Module):
    """
    The convolutional network used for experiments on Omniglot, firstly
    introduced by [1].

    It has 4 modules with a 3 × 3 convolutions and 64 filters, followed by
    batch normalization, a ReLU nonlinearity, and 2 × 2 max-pooling.

    This network assumes the images are downsampled to 28 × 28 and have 1
    channel. Namely, the shapes of inputs are (1, 28, 28).

    Args:
        n_classes (int): Size of the network's output. This corresponds to
            ``N`` in ``N-way`` classification. ``None`` if the linear classifier
            is not needed.

    .. admonition:: References

        1. "`Matching Networks for One Shot Learning. <https://arxiv.org/abs/1606.04080>`_" \
            Oriol Vinyals, et al. NIPS 2016.
    """

    def __init__(self, n_classes: Optional[int] = None):
        super(OmniglotCNN, self).__init__()

        self.hidden_size = 64
        self.encoder = ConvGroup(
            in_channels = 1,
            hidden_size = self.hidden_size,
            layers = 4
        )
        self.flatten = Flatten()
        self.n_classes = n_classes

        if n_classes:
            self.classifier = nn.Linear(self.hidden_size, n_classes)
            self.init_weights()

    def init_weights(self):
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.mul_(0.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input data (batch_size, in_channels = 1,
                img_size = 28, img_size = 28)

        Returns:
            - ``n_classes`` is not None: Class scores (batch_size, n_classes)
            - ``n_classes`` is None: Features (batch_size, 64)
        """
        features = self.encoder(x)  # (batch_size, 64, 28 / 16 = 1, 28 / 16 = 1)
        features = self.flatten(features)  # (batch_size, 64)

        if self.n_classes:
            output = self.classifier(features)  # (batch_size, n_classes)

        return output


class MiniImagenetCNN(nn.Module):
    """
    The convolutional network used for experiments on MiniImagenet, firstly
    introduced by [1].

    It has 4 modules with a 3 × 3 convolutions and 32 filters, followed by
    batch normalization, a ReLU nonlinearity, and 2 × 2 max-pooling.

    This network assumes the images are downsampled to 84 × 84 and have 3
    channel. Namely, the shapes of inputs are (3, 84, 84).

    Args:
        n_classes (int): Size of the network's output. This corresponds to
            ``N`` in ``N-way`` classification. ``None`` if the linear classifier
            is not needed.

    .. admonition:: References

        1. "`Optimization as a Model for Few-Shot Learning. \
            <https://openreview.net/pdf?id=rJY0-Kcll>`_" Sachin Ravi, et al. ICLR 2017.
    """

    def __init__(self, n_classes: Optional[int] = None):
        super(OmniglotCNN, self).__init__()

        self.hidden_size = 32
        self.encoder = ConvGroup(
            in_channels = 3,
            hidden_size = self.hidden_size,
            layers = 4
        )
        self.flatten = Flatten()
        self.n_classes = n_classes

        if n_classes:
            self.classifier = nn.Linear(5 * 5 * self.hidden_size, n_classes)
            self.init_weights()

    def init_weights(self):
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.mul_(0.0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input data (batch_size, in_channels = 3,
                img_size = 84, img_size = 84)

        Returns:
            - ``n_classes`` is not None: Class scores (batch_size, n_classes)
            - ``n_classes`` is None: Features (batch_size, 800)
        """
        features = self.encoder(x)  # (batch_size, 32, 84 / 16 = 5, 84 / 16 = 5)
        features = self.flatten(features)  # (batch_size, 32 × 5 × 5 = 800)

        if self.n_classes:
            output = self.classifier(features)  # (batch_size, n_classes)

        return output


class OmniglotMLP(nn.Module):
    """
    The fully-connected network used for experiments on Omniglot, firstly
    introduced by [1].

    It has 4 hidden layers with sizes 256, 128, 64, 64, each including batch
    normalization and ReLU nonlinearities, followed by a linear layer and
    softmax.

    Args:
        input_size (int): Size of the network's input
        n_classes (int): Size of the network's output. This corresponds to
            ``N`` in ``N-way`` classification.

    .. admonition:: References

        1. "`Meta-Learning with Memory-Augmented Neural Networks. \
            <http://proceedings.mlr.press/v48/santoro16.pdf>`_" Adam Santoro, et al. ICML 2016.
    """

    def __init__(self, input_size: int, n_classes: int):
        super(OmniglotMLP, self).__init__()

        linear_sizes = [input_size, 256, 128, 64, 64]

        self.encoder = nn.Sequential(
            LinearBlock(in_size, out_size)
            for in_size, out_size in zip(linear_sizes[:-1], linear_sizes[1:])
        )
        self.classifier = nn.Linear(linear_sizes[-1], n_classes)
        self.flatten = Flatten()
        self.init_weights()

    def init_weights(self):
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.mul_(0.0)

    def forward(self, x: torch.Tensor):
        x = self.flatten(x)
        features = self.encoder(x)
        output = self.classifier(features)
        return output
