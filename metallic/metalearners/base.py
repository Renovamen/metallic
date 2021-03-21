import os
from abc import ABC, abstractmethod
from typing import Callable, Optional
import torch
from torch import nn, optim

class MetaLearner(ABC):
    """
    A base class for all meta-learning algorithms.

    Args:
        model (torch.nn.Module): Model to be wrapped
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
        root: Optional[str] = None,
        save_basename: Optional[str] = None,
        lr_scheduler: Optional[Callable] = None,
        loss_function: Optional[Callable] = None,
        device: Optional = None
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = model.to(device)

        self.root = os.path.expanduser(root)
        self.save_basename = save_basename

        self.lr_scheduler = lr_scheduler

        if loss_function is None:
            loss_function = nn.CrossEntropyLoss()
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

    def lr_schedule(self) -> None:
        """Schedule learning rate."""
        self.lr_scheduler.step()

    @classmethod
    @abstractmethod
    def load(cls, model_path: str, **kwargs):
        """Load a trained model."""
        pass

    @abstractmethod
    def save(self, prefix: Optional[str] = None) -> str:
        """Save the trained model."""
        pass
