"""
A collection of some distance computing functions for calculating similarity
between two tensors. They are useful in metric-based meta-learning algorithms.
"""

import torch

def normalize(x: torch.Tensor, eps: float = 1e-10):
    """
    Normalize a tensor:

    .. math ::
        \\text{Normalized}(x) = \\frac{x}{\| x \|_2}

    Args:
        x (torch.Tensor): A tensor with shape ``(N, embed_dim)``
        eps (float, optional, default=1e-10): A small value to avoid division
            by zero.

    Returns:
        result (torch.Tensor): Normalize version of tensor ``x``, with shape
            ``(N, embed_dim)``
    """
    x = x / torch.clamp(x.norm(p=2, dim=1, keepdim=True), min=eps)
    return x

def euclidean_distance(
    x: torch.FloatTensor, y: torch.FloatTensor
) -> torch.torch.FloatTensor:
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
        distance (torch.Tensor): Euclidean distance between tensor ``x`` and
            tensor ``y``, with shape ``(M, N)``

    .. admonition:: References

        1. "`Prototypical Networks for Few-shot Learning. \
            <https://arxiv.org/abs/1703.05175>`_" Jake Snell, et al. NIPS 2017.
    """
    n = x.size(0)
    m = y.size(0)
    x = x.unsqueeze(0).expand(m, n, -1)
    y = y.unsqueeze(1).expand(m, n, -1)
    distance = ((x - y) ** 2).sum(dim=2)
    return distance

def cosine_distance(
    x: torch.FloatTensor, y: torch.FloatTensor, eps: float = 1e-10
) -> torch.torch.FloatTensor:
    """
    Compute the pairwise cosine distance between two tensors:

    .. math ::
        \\text{distance} = \\frac{x \cdot y}{\| x \|_2 \cdot \| x_2 \|_2}

    Args:
        x (torch.Tensor): A tensor with shape ``(N, embed_dim)``
        y (torch.Tensor): A tensor with shape ``(M, embed_dim)``

    Returns:
        distance (torch.Tensor): cosine distance between tensor ``x`` and
            tensor ``y``, with shape ``(M, N)``

    .. admonition:: References

        1. "`Matching Networks for One Shot Learning. \
            <https://arxiv.org/abs/1606.04080>`_" Oriol Vinyals, et al. NIPS 2016.
    """

    x_norm = normalize(x, eps)
    y_norm = normalize(y, eps)
    return torch.mm(x_norm, y_norm.t())

_DISTANCE = {
    'euclidean': euclidean_distance,
    'cosine': cosine_distance
}

def get_distance_function(distance: str):
    return _DISTANCE[distance]
