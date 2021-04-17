from typing import Callable, Optional, Tuple
import higher
import torch
from torch import nn, optim

from .base import GBML
from ...utils import get_accuracy
from ...functional import apply_grads, accum_grads

class ANIL(GBML):
    """
    Implementation of Almost No Inner Loop (ANIL) algorithm proposed in [1],
    which only update the head of the neural netowork in inner loop.

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

        1. "`Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness \
            of MAML. <https://arxiv.org/abs/1909.09157>`_" Aniruddh Raghu, et al. ICLR 2020.
    """

    alg_name = 'ANIL'

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
        if save_basename is None:
            save_basename = self.alg_name

        super(ANIL, self).__init__(
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

        self.encoder = model.encoder
        self.head = model.classifier

    def step(self, batch: dict, meta_train: bool = True) -> Tuple[float]:
        """Outer loop update."""
        self.out_optim.zero_grad()

        task_batch, n_tasks = self.get_tasks(batch)

        outer_loss, outer_accuracy = 0., 0.
        encoder_grad_list, head_grad_list = [], []

        for task_data in task_batch:
            support_input, support_target, query_input, query_target = task_data

            with higher.innerloop_ctx(
                self.head, self.in_optim, copy_initial_weights=False, track_higher_grads=meta_train
            ) as (fhead, diffopt):
                with torch.no_grad():
                    support_feature = self.encoder(support_input)
                # inner loop (adapt)
                self.inner_loop(fhead, diffopt, support_feature, support_target)

                # evaluate on the query set
                with torch.set_grad_enabled(meta_train):
                    quert_feature = self.encoder(query_input)
                    query_output = fhead(quert_feature)
                    query_loss = self.loss_function(query_output, query_target)
                    query_loss /= len(query_input)

                # find accuracy on query set
                query_accuracy = get_accuracy(query_output, query_target)

                outer_loss += query_loss.detach().item()
                outer_accuracy += query_accuracy.item()

                # compute gradients when in the meta-training stage
                if meta_train == True:
                    (query_loss / n_tasks).backward()

        if meta_train == True:
            # outer loop update
            self.out_optim.step()

        # average the losses and accuracies
        outer_loss /= n_tasks
        outer_accuracy /= n_tasks

        return outer_loss, outer_accuracy
