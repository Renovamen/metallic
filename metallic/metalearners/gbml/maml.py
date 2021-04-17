from typing import Callable, Optional, Tuple
import higher
import torch
from torch import nn, optim

from .base import GBML
from ...functional import apply_grads, accum_grads

class MAML(GBML):
    """
    Implementation of Model-Agnostic Meta-Learning (MAML) algorithm proposed
    in [1].

    `Here <https://github.com/cbfinn/maml>`_ is the official implementation
    of MAML based on Tensorflow.


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
        Loss function

    inner_steps : int, optional, defaut=1
        Number of gradient descent updates in inner loop

    first_order : bool, optional, default=False
        Use the first-order approximation of MAML (FOMAML) or not

    device : optional
        Device on which the model is defined


    .. admonition:: References

        1. "`Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. \
            <https://arxiv.org/abs/1703.03400>`_" Chelsea Finn, et al. ICML 2017.
    """

    alg_name = 'MAML'

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
        first_order: bool = False,
        device: Optional = None
    ) -> None:
        if save_basename is None:
            save_basename = self.alg_name

        super(MAML, self).__init__(
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

        self.first_order = first_order
        self.grad_list = []

    def clear_before_outer_loop(self):
        """Initialization before each outer loop if needed."""
        self.grad_list = []

    def outer_loop_update(self):
        """Update the model's meta-parameters to optimize the query loss."""
        apply_grads(self.model, accum_grads(self.grad_list))  # apply accumulated gradients to the original model parameters
        self.out_optim.step()  # outer loop update

    def compute_outer_grads(
        self, task: Tuple[torch.Tensor], n_tasks: int, meta_train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients on query set."""

        support_input, support_target, query_input, query_target = task

        # Use higher to make the model stateless and use differentiable
        # optimizer. So that the model's parameters can be automatically
        # kept copies of as they are being updated.
        with higher.innerloop_ctx(
            self.model, self.in_optim, track_higher_grads=(meta_train and (not self.first_order))
        ) as (fmodel, diffopt):
            # fmodel: stateless version of the model
            # diffopt: differentiable version of the optimizer

            # inner loop (adapt)
            self.inner_loop(fmodel, diffopt, support_input, support_target)

            # evaluate on the query set
            with torch.set_grad_enabled(meta_train):
                query_output = fmodel(query_input)
                query_loss = self.loss_function(query_output, query_target)
                query_loss /= len(query_input)

            # compute gradients when in the meta-training stage
            if meta_train == True:
                # (query_loss / n_tasks).backward()
                outer_grad = torch.autograd.grad(query_loss / n_tasks, fmodel.parameters(time=0))
                self.grad_list.append(outer_grad)

        return query_output, query_loss
