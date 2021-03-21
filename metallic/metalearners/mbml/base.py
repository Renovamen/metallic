import os
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple
import torch
from torch import nn, optim

from ..base import MetaLearner

class MBML(MetaLearner, ABC):
    """
    A base class for metric-based meta-learning algorithms.

    Args:
        model (torch.nn.Module): Model to be wrapped
        optim (torch.optim.Optimizer): Optimizer
        root (str): Root directory to save checkpoints
        save_basename (str, optional): Base name of the saved checkpoints
        lr_scheduler (callable, optional): Learning rate scheduler
        loss_function (callable, optional): Loss function
        device (optional): Device on which the model is defined. If `None`,
            device will be detected automatically.
    """

    def __init__(
        self,
        model: nn.Module,
        optim: optim.Optimizer,
        root: Optional[str] = None,
        save_basename: Optional[str] = None,
        lr_scheduler: Optional[Callable] = None,
        loss_function: Optional[Callable] = None,
        device: Optional = None
    ) -> None:
        super(MBML, self).__init__(
            model = model,
            root = root,
            save_basename = save_basename,
            lr_scheduler = lr_scheduler,
            loss_function = loss_function,
            device = device
        )

        self.optim = optim

    @classmethod
    def load(cls, model_path: str, **kwargs):
        """Load a trained model."""
        state = torch.load(model_path)

        # load model and optimizers
        kwargs['model'] = state['model']
        kwargs['optim'] = state['optim']

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
            'optim': self.optim
        }

        name = self.save_basename
        if prefix is not None:
            name = prefix + name + '.pth.tar'

        path = os.path.join(self.root, name)
        torch.save(state, os.path.join(self.root, name))
        return path

    @abstractmethod
    def step(self, batch: dict, meta_train: bool = True) -> Tuple[float]:
        pass
