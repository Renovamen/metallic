import torch
from torch import nn
from metallic.models.base import LinearBlock

class MLP(nn.Module):
    """
    The fully-connected network used for experiments on Omniglot, firstly
    introduced by [1], also used by [2].

    It has 4 hidden layers with sizes 256, 128, 64, 64, each including batch
    normalization and ReLU nonlinearities, followed by a linear layer and
    softmax.

    Args:
        input_size (int): Size of the network's input
        n_classes (int): Size of the network's output. This corresponds to
            ``N`` in ``N-way`` classification.

    References
    ----------
    1. "Meta-Learning with Memory-Augmented Neural Networks." Adam Santoro, et al. ICML 2016.
    2. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." Chelsea Finn, et al. ICML 2017.
    """

    def __init__(self, input_size: int, n_classes: int):
        super(MLP, self).__init__()

        linear_sizes = [input_size, 256, 128, 64, 64]

        self.features = nn.Sequential(
            LinearBlock(in_size, out_size)
            for in_size, out_size in zip(linear_sizes[:-1], linear_sizes[1:])
        )
        self.classifier = nn.Linear(linear_sizes[-1], n_classes)
        self.init_weights()

    def init_weights(self):
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.mul_(0.0)

    def forward(self, x: torch.tensor):
        x = x.view(x.size(0), -1)
        features = self.features(x)
        scores = self.classifier(features)
        return scores
