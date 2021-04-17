import os
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import torch
from torch import nn, optim

from ..base import MetaLearner
from ...utils import get_accuracy

class GBML(MetaLearner, ABC):
    """
    A base class for gradient-based meta-learning algorithms.

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
        Device on which the model is defined. If `None`, device will be
        detected automatically.
    """

    def __init__(
        self,
        model: nn.Module,
        in_optim: optim.Optimizer,
        out_optim: optim.Optimizer,
        root: Optional[str] = None,
        save_basename: Optional[str] = None,
        lr_scheduler: Optional[Callable] = None,
        loss_function: Optional[Callable] = None,
        inner_steps: int = 1,
        device: Optional = None
    ) -> None:
        super(GBML, self).__init__(
            model = model,
            root = root,
            save_basename = save_basename,
            lr_scheduler = lr_scheduler,
            loss_function = loss_function,
            device = device
        )

        self.model.train()

        self.in_optim = in_optim
        self.out_optim = out_optim
        self.inner_steps = inner_steps

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target) -> None:
        """Inner loop update."""
        for step in range(self.inner_steps):
            # compute loss on the support set
            train_output = fmodel(train_input)
            support_loss = self.loss_function(train_output, train_target)
            # update parameters
            diffopt.step(support_loss)

    @classmethod
    def load(cls, model_path: str, **kwargs):
        """Load a trained model."""
        state = torch.load(model_path)

        # load model and optimizers
        kwargs['model'] = state['model']
        kwargs['in_optim'] = state['in_optim']
        kwargs['out_optim'] = state['out_optim']

        # model name and save path
        if 'root' not in kwargs:
            kwargs['root'] = os.path.dirname(model_path)
        if 'save_basename' not in kwargs:
            kwargs['save_basename'] = os.path.basename(model_path)

        return cls(**kwargs)

    def save(self, prefix: Optional[str] = None) -> str:
        """Save the trained model."""
        if self.root is None or self.save_basename is None:
            raise RuntimeError('The root directory or save basename of the'
                'checkpoints is not defined.')

        state = {
            'model': self.model,
            'in_optim': self.in_optim,
            'out_optim': self.out_optim
        }

        name = self.save_basename
        if prefix is not None:
            name = prefix + name + '.pth.tar'

        path = os.path.join(self.root, name)
        torch.save(state, os.path.join(self.root, name))
        return path

    @abstractmethod
    def compute_outer_grads(
        self, batch: Tuple[torch.Tensor], meta_train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gradients on query set."""
        pass

    def clear_before_outer_loop(self):
        """Initialization before each outer loop if needed."""
        pass

    def outer_loop_update(self):
        """Update the model's meta-parameters to optimize the query loss."""
        self.out_optim.step()

    def step(self, batch: dict, meta_train: bool = True) -> Tuple[float]:
        """Outer loop"""

        # clear gradient of last batch
        self.out_optim.zero_grad()

        # loss and accuracy on query set (outer loop)
        outer_loss, outer_accuracy = 0., 0.

        self.clear_before_outer_loop()

        # get task batch
        task_batch, n_tasks = self.get_tasks(batch)

        for task in task_batch:
            # compute outer loop gradients
            query_output, query_loss = self.compute_outer_grads(task, n_tasks, meta_train)

            # find accuracy on query set
            _, _, _, query_target = task
            query_accuracy = get_accuracy(query_output, query_target)

            # record losses and accuracy
            outer_loss += query_loss.detach().item()
            outer_accuracy += query_accuracy.item()

        # When in the meta-training stage, update the model's meta-parameters to
        # optimize the query loss across all of the tasks sampled in this batch.
        if meta_train == True:
            self.outer_loop_update()

        # average loss and acccuracy
        outer_loss /= n_tasks
        outer_accuracy /= n_tasks

        return outer_loss, outer_accuracy
