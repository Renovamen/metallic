from typing import Callable, Optional, Tuple
import higher
import torch
from torch import nn, optim

from .base import GBML
from .utils import apply_grads
from ...utils import accuracy

class MAML(GBML):
    """
    Implementation of Model-Agnostic Meta-Learning (MAML) algorithm proposed
    in [1].

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
        first_order (bool, optional, default=False): Use the first-order
            approximation of MAML (FOMAML) or not
        device (optional): Device on which the model is defined

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

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target) -> None:
        """Inner loop update."""
        for step in range(self.inner_steps):
            # compute loss on the support set
            train_output = fmodel(train_input)
            support_loss = self.loss_function(train_output, train_target)
            # update parameters
            diffopt.step(support_loss)

    def outer_loop(self, batch: dict, meta_train: bool = True) -> Tuple[float]:
        """Outer loop update."""

        # clear gradient of last batch
        self.out_optim.zero_grad()

        # get task batch
        task_batch, n_tasks = self.get_tasks(batch)

        # loss and accuracy on query set (outer loop)
        outer_loss, outer_accuracy = 0., 0.

        # record gradients
        grad_list = []

        for task_data in task_batch:
            # input: (n_way × k_shot, channels, img_size, img_size)
            # target: (n_way × k_shot)
            support_input, support_target, query_input, query_target = task_data

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

                # find accuracy on query set
                query_accuracy = accuracy(query_output, query_target)

                outer_loss += query_loss.detach().item()
                outer_accuracy += query_accuracy.item()

                # compute gradients when in the meta-training stage
                if meta_train == True:
                    # query_loss.backward()
                    outer_grad = torch.autograd.grad(query_loss / n_tasks, fmodel.parameters(time=0))
                    grad_list.append(outer_grad)

        # When in the meta-training stage, update the model's meta-parameters to
        # optimize the query losses across all of the tasks sampled in this batch.
        if meta_train == True:
            # apply gradients to the original model parameters
            apply_grads(self.model, grad_list[-1])
            # outer loop update
            self.out_optim.step()

        # average the losses and accuracies
        outer_loss /= n_tasks
        outer_accuracy /= n_tasks

        return outer_loss, outer_accuracy