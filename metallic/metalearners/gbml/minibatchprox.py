from typing import Callable, Optional, Tuple
import copy
import higher
import torch
from torch import nn, optim

from .base import GBML
from .utils import apply_grads
from ...utils import accuracy, ProximalRegLoss

class MinibatchProx(GBML):
    """
    Implementation of MinibatchProx algorithm proposed in [1].

    `Here <https://panzhous.github.io/assets/code/MetaMinibatchProx.zip>`_ is
    the official implementation of MinibatchProx based on Tensorflow.

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
        lamb (float, optional, float=0.1): Regularization strength of the inner
            level proximal regularization
        device (optional): Device on which the model is defined

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
    def inner_loop(
        self, fmodel, diffopt, train_input, train_target, init_params
    ) -> None:
        """Inner loop update."""
        for step in range(self.inner_steps):
            train_output = fmodel(train_input)
            params = list(fmodel.parameters())  # model-parameters
            support_loss = self.reg_loss_function(
                train_output, train_target, init_params, params
            )
            diffopt.step(support_loss)

    def outer_loop(self, batch: dict, meta_train: bool = True) -> Tuple[float]:
        """Outer loop update."""

        self.out_optim.zero_grad()

        task_batch, n_tasks = self.get_tasks(batch)

        outer_loss, outer_accuracy = 0., 0.
        grad_list = []

        for task_data in task_batch:
            support_input, support_target, query_input, query_target = task_data

            with higher.innerloop_ctx(
                self.model, self.in_optim, track_higher_grads=False
            ) as (fmodel, diffopt):
                # record meta-parameters
                init_params = [
                    p.detach().clone().requires_grad_(True)
                    for p in fmodel.parameters()
                ]

                # inner loop (adapt)
                self.inner_loop(fmodel, diffopt, support_input, support_target, init_params)

                # evaluate on the query set
                with torch.set_grad_enabled(meta_train):
                    query_output = fmodel(query_input)
                    query_loss = self.loss_function(query_output, query_target)
                    query_loss /= len(query_input)

                # find accuracy on query set
                query_accuracy = accuracy(query_output, query_target)

                outer_loss += query_loss.detach().item()
                outer_accuracy += query_accuracy.item()

                # compute gradients when in the meta-training stage
                if meta_train == True:
                    outer_grad = []
                    for p, fast_p in zip(self.model.parameters(), fmodel.parameters()):
                        outer_grad.append((p.data - fast_p.data) / n_tasks)
                    grad_list.append(outer_grad)

        if meta_train == True:
            # apply gradients to the original model parameters
            apply_grads(self.model, grad_list[-1])
            # outer loop update
            self.out_optim.step()

        outer_loss /= n_tasks
        outer_accuracy /= n_tasks

        return outer_loss, outer_accuracy
