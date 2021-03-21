from typing import Callable, Optional, Tuple
import higher
import torch
from torch import nn, optim

from .base import MBML
from ...utils import get_accuracy
from ...functional import get_prototypes, get_distance_function

class ProtoNet(MBML):
    """
    Implementation of Prototypical Networks proposed in [1].

    `Here <https://github.com/jakesnell/prototypical-networks>`_ is the official
    implementation of Prototypical Networks based on PyTorch.

    Args:
        model (torch.nn.Module): Model to be wrapped
        optim (torch.optim.Optimizer): Optimizer
        root (str): Root directory to save checkpoints
        save_basename (str, optional): Base name of the saved checkpoints
        lr_scheduler (callable, optional): Learning rate scheduler
        loss_function (callable, optional): Loss function
        device (optional): Device on which the model is defined. If `None`,
            device will be detected automatically.

    .. admonition:: References

        1. "`Prototypical Networks for Few-shot Learning. \
            <https://arxiv.org/abs/1703.05175>`_" Jake Snell, et al. NIPS 2017.
    """

    alg_name = 'ProtoNet'

    def __init__(
        self,
        model: nn.Module,
        optim: optim.Optimizer,
        root: Optional[str] = None,
        save_basename: Optional[str] = None,
        lr_scheduler: Optional[Callable] = None,
        loss_function: Optional[Callable] = None,
        distance: str = 'euclidean',
        device: Optional = None
    ) -> None:
        if save_basename is None:
            save_basename = self.alg_name

        super(ProtoNet, self).__init__(
            model = model,
            optim = optim,
            root = root,
            save_basename = save_basename,
            lr_scheduler = lr_scheduler,
            loss_function = loss_function,
            device = device
        )

        self.get_distance = get_distance_function(distance)

    def step(self, batch: dict, meta_train: bool = True) -> Tuple[float]:
        if meta_train:
            self.model.train()
        else:
            self.model.eval()

        task_batch, n_tasks = self.get_tasks(batch)
        losses, accuracies = 0., 0.

        self.optim.zero_grad()

        for task_data in task_batch:
            # input: (n_way × k_shot, channels, img_size, img_size)
            # target: (n_way × k_shot)
            support_input, support_target, query_input, query_target = task_data

            with torch.set_grad_enabled(meta_train):
                support_embeddings = self.model(support_input)
                query_embeddings = self.model(query_input)

                prototypes = get_prototypes(support_embeddings, support_target)
                distance = self.get_distance(prototypes, query_embeddings)  # (n_samples, n_way)
                loss = self.loss_function(-distance, query_target)

            with torch.no_grad():
                accuracy = get_accuracy(-distance, query_target)

            losses += loss.detach().item()
            accuracies += accuracy.item()

            if meta_train == True:
                (loss / n_tasks).backward()
                self.optim.step()

        # average the losses and accuracies
        losses /= n_tasks
        accuracies /= n_tasks

        return losses, accuracies
