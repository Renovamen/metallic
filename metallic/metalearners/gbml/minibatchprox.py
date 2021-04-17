from typing import Callable, Optional
import torch
from torch import nn, optim

from .reptile import Reptile
from ...functional import ProximalRegLoss

class MinibatchProx(Reptile):
    """
    Implementation of MinibatchProx algorithm proposed in [1].

    `Here <https://panzhous.github.io/assets/code/MetaMinibatchProx.zip>`_ is
    the official implementation of MinibatchProx based on Tensorflow.

    Parameters
    ----------
    model : torch.nn.Module
        Model to be wrapped

    in_optim : torch.optim.Optimizer
        Optimizer for the inner loop

    out_optim : torch.optim.Optimizer
        Optimizer for the outer loop

    root : str
        Root directory to save checkpoints

    save_basename : str, optional
        Base name of the saved checkpoints

    lr_scheduler : callable, optional
        Learning rate scheduler

    loss_function : callable, optional
        Loss function to be wrapped in :func:`metallic.functional.ProximalRegLoss()`

    inner_steps : int, optional, defaut=1
        Number of gradient descent updates in inner loop

    device : optional
        Device on which the model is defined


    .. admonition:: References

        1. "`Efficient Meta Learning via Minibatch Proximal Update. \
            <https://panzhous.github.io/assets/pdf/2019-NIPS-metaleanring.pdf>`_" \
            Pan Zhou, et al. NIPS 2019.
    """

    alg_name = 'MinibatchProx'

    def __init__(
        self,
        model: nn.Module,
        in_optim: optim.Optimizer,
        out_optim: optim.Optimizer,
        root: str,
        save_basename: Optional[str] = None,
        lr_scheduler: Optional[Callable] = None,
        loss_function: Optional[Callable] = None,
        inner_steps: int = 5,
        lamb: float = 0.1,
        device: Optional = None
    ) -> None:
        if save_basename is None:
            save_basename = self.alg_name

        for g in out_optim.param_groups:
            g['lr'] = g['lr'] * lamb

        super(MinibatchProx, self).__init__(
            model = model,
            in_optim = in_optim,
            out_optim = out_optim,
            root = root,
            save_basename = save_basename,
            lr_scheduler = lr_scheduler,
            loss_function = loss_function,
            inner_steps = inner_steps,
            device = device
        )

        self.reg_loss_function = ProximalRegLoss(self.loss_function, lamb)

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target) -> None:
        """Inner loop update."""
        # record meta-parameters
        init_params = [
            p.detach().clone().requires_grad_(True)
            for p in fmodel.parameters()
        ]

        for step in range(self.inner_steps):
            train_output = fmodel(train_input)
            params = list(fmodel.parameters())  # model-parameters
            support_loss = self.reg_loss_function(
                train_output, train_target, init_params, params
            )
            diffopt.step(support_loss)
