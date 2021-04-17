from typing import Callable, Optional, Tuple
import higher
import torch
from torch import nn, optim

from .base import GBML
from ...functional import apply_grads, accum_grads

class Reptile(GBML):
    """
    Implementation of Reptile algorithm proposed in [1].

    `Here <https://github.com/openai/supervised-reptile>`_ is the official
    implementation of Reptile based on Tensorflow.

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

    device : optional
        Device on which the model is defined


    .. admonition:: References

        1. "`On First-Order Meta-Learning Algorithms. <https://arxiv.org/abs/1803.02999>`_" \
            Alex Nichol, et al. arxiv 2018.
    """

    alg_name = 'Reptile'

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
        device: Optional = None
    ) -> None:
        if save_basename is None:
            save_basename = self.alg_name

        super(Reptile, self).__init__(
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

        with higher.innerloop_ctx(
            self.model, self.in_optim, track_higher_grads=False
        ) as (fmodel, diffopt):
            # inner loop (adapt)
            self.inner_loop(fmodel, diffopt, support_input, support_target)

            # evaluate on the query set
            with torch.set_grad_enabled(meta_train):
                query_output = fmodel(query_input)
                query_loss = self.loss_function(query_output, query_target)
                query_loss /= len(query_input)

            # compute gradients when in the meta-training stage
            if meta_train == True:
                outer_grad = []
                for p, fast_p in zip(self.model.parameters(), fmodel.parameters()):
                    outer_grad.append((p.data - fast_p.data) / n_tasks)
                self.grad_list.append(outer_grad)

        return query_output, query_loss
