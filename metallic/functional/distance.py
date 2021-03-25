"""
A collection of some distance computing functions for calculating similarity
between two tensors. They are useful in metric-based meta-learning algorithms.
"""

import torch
import torch.nn.functional as f
from typing import Callable

def euclidean_distance(
    x: torch.FloatTensor, y: torch.FloatTensor
) -> torch.FloatTensor:
    """
    Compute the pairwise squared Euclidean distance between two tensors:

    .. math ::
        \\text{distance} = \| x - y \|^2

    Args:
        x (torch.Tensor): A tensor with shape ``(N, embed_dim)``
        y (torch.Tensor): A tensor with shape ``(M, embed_dim)``
        eps (float, optional, default=1e-10): A small value to avoid division
            by zero.

    Returns:
        distance (torch.Tensor): Euclidean distance between tensor ``x`` and \
            tensor ``y``, with shape ``(M, N)``

    .. admonition:: References

        1. "`Prototypical Networks for Few-shot Learning. \
            <https://arxiv.org/abs/1703.05175>`_" Jake Snell, et al. NIPS 2017.
    """
    n = x.size(0)
    m = y.size(0)
    x = x.unsqueeze(0).expand(m, n, -1)
    y = y.unsqueeze(1).expand(m, n, -1)
    distance = ((x - y) ** 2).sum(dim=-1)
    return distance

def cosine_distance(
    x: torch.FloatTensor, y: torch.FloatTensor, eps: float = 1e-10
) -> torch.FloatTensor:
    """
    Compute the pairwise cosine distance between two tensors:

    .. math ::
        \\text{distance} = \\frac{x \cdot y}{\| x \|_2 \cdot \| x_2 \|_2}

    Args:
        x (torch.Tensor): A tensor with shape ``(N, embed_dim)``
        y (torch.Tensor): A tensor with shape ``(M, embed_dim)``

    Returns:
        distance (torch.Tensor): cosine distance between tensor ``x`` and \
            tensor ``y``, with shape ``(M, N)``

    .. admonition:: References

        1. "`Matching Networks for One Shot Learning. \
            <https://arxiv.org/abs/1606.04080>`_" Oriol Vinyals, et al. NIPS 2016.
    """

    x_norm = f.normalize(x, dim=1, eps=eps)
    y_norm = f.normalize(y, dim=1, eps=eps)
    return (x_norm @ y_norm.t()).t()


_DISTANCE = {
    'euclidean': euclidean_distance,
    'cosine': cosine_distance
}

def get_distance_function(distance: str) -> Callable:
    return _DISTANCE[distance]
