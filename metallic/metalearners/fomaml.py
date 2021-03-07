from typing import Callable, Optional
from torch import nn, optim
from .maml import MAML

class FOMAML(MAML):
    """
    Implementation of Fisrt-Order Model-Agnostic Meta-Learning (FOMAML)
    algorithm proposed in [1]. In FOMAML, the second derivatives in outer loop
    are omitted, which means the gradients are directly computed on the fast
    parameters.

    `Here <https://github.com/cbfinn/maml>`_ is the official implementation
    of MAML based on Tensorflow.

    Args:
        model (torch.nn.Module): Model to be wrapped
        in_optim (torch.optim.Optimizer): Optimizer for the inner loop
        out_optim (torch.optim.Optimizer): Optimizer for the outer loop
        root (str): Root directory to save checkpoints
        save_basename (str, optional): Base name of the saved checkpoints
        lr_scheduler (callable, optional): Learning rate scheduler
        loss_function (callable, optional): Loss function
        inner_steps (int, optional, defaut=1): Number of gradient descent
            updates in inner loop
        device (optional): Device on which the model is defined

    References
    ----------
    1. "`Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. \
        <https://arxiv.org/abs/1703.03400>`_" Chelsea Finn, et al. ICML 2017.
    """

    alg_name = 'FOMAML'

    def __init__(
        self,
        model: nn.Module,
        in_optim: optim.Optimizer,
        out_optim: optim.Optimizer,
        root: str,
        save_basename: Optional[str] = None,
        lr_scheduler: Optional[Callable] = None,
        loss_function: Optional[Callable] = None,
        inner_steps: int = 1,
        device: Optional = None
    ) -> None:
        super(FOMAML, self).__init__(
            model = model,
            in_optim = in_optim,
            out_optim = out_optim,
            root = root,
            save_basename = save_basename,
            lr_scheduler = lr_scheduler,
            loss_function = loss_function,
            inner_steps = inner_steps,
            first_order = True,
            device = device
        )
