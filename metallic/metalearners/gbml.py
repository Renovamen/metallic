import os
from typing import Callable, Optional
import torch
from torch import nn, optim

class GBML:
    """
    A base class for gradient-based meta-learning algorithms.

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
    """
    def __init__(
        self,
        model: nn.Module,
        in_optim: optim.Optimizer,
        out_optim: optim.Optimizer,
        root: Optional[str] = None,
        save_basename: Optional[str] = None,
        lr_scheduler: Optional[Callable] = None,
        loss_function: Optional[Callable] = nn.CrossEntropyLoss(),
        inner_steps: int = 1,
        device: Optional = None
    ) -> None:
        self.device = device

        self.model = model.to(device)
        self.model.train()

        self.root = os.path.expanduser(root)
        self.save_basename = save_basename

        self.in_optim = in_optim
        self.out_optim = out_optim
        self.lr_scheduler = lr_scheduler
        self.inner_steps = inner_steps

        self.loss_function = loss_function

    def get_tasks(self, batch: dict) -> tuple:
        # support set
        support_inputs, support_targets = batch['support']
        support_inputs = support_inputs.to(self.device)
        support_targets = support_targets.to(self.device)

        # query set
        query_inputs, query_targets = batch['query']
        query_inputs = query_inputs.to(self.device)
        query_targets = query_targets.to(self.device)

        # number of tasks
        n_tasks = query_targets.size(0)

        task_batch = zip(
            support_inputs, support_targets, query_inputs, query_targets
        )
        return task_batch, n_tasks

    def inner_loop(self):
        """Inner loop update."""
        raise NotImplementedError

    def outer_loop(self):
        """Outer loop update."""
        raise NotImplementedError

    def lr_schedule(self) -> None:
        """Schedule learning rate."""
        self.lr_scheduler.step()

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
        if 'save_basename' not in  kwargs:
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
