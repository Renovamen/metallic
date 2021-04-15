import torch
from torch import nn
from typing import Optional, Sequence, List

def apply_grads(model: nn.Module, grads: Sequence[torch.Tensor]) -> None:
    """
    Map a list of gradients to a model.

    Parameters
    ----------
    grads : Sequence[torch.Tensor]
        List of gradient for each model parameter
    """
    if not len(grads) == len(list(model.parameters())):
        msg = 'WARNING: Parameters and gradients have different length. ('
        msg += str(len(list(model.parameters()))) + ' vs ' + str(len(grads)) + ')'
        print(msg)

    for param, grad in zip(model.parameters(), grads):
        if grad is not None:
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad += grad.clone()

def accum_grads(grads: Sequence[Sequence[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Compute accumulated gradients

    Parameters
    ----------
    grads : Sequence[Sequence[torch.Tensor]]
        List of gradients on all previous tasks

    Returns
    -------
    sum_grads : List[torch.Tensor]
       Accumulated gradients
    """
    sum_grads = [torch.stack(g).sum(dim=0) for g in zip(*grads)]
    return sum_grads
