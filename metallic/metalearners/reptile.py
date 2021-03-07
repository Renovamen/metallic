from typing import Callable, Optional, Tuple
import copy
import higher
import torch
from torch import nn, optim

from .gbml import GBML
from ._utils import apply_grads
from ..utils import accuracy

class Reptile(GBML):
    """
    Implementation of Reptile algorithm proposed in [1].

    `Here <https://github.com/openai/supervised-reptile>`_ is the official
    implementation of Reptile based on Tensorflow.

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
            inner_steps = inner_steps,
            device = device
        )

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target) -> None:
        """Inner loop update."""
        for step in range(self.inner_steps):
            train_output = fmodel(train_input)
            support_loss = self.loss_function(train_output, train_target)
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
                # inner loop (adapt)
                self.inner_loop(fmodel, diffopt, support_input, support_target)

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
