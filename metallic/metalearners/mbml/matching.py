from typing import Callable, Optional, Tuple
import higher
import torch
from torch import nn, optim
import torch.nn.functional as f

from .base import MBML
from ...utils import get_accuracy
from ...functional import get_distance_function

class MatchNet(MBML):
    """
    Implementation of Matching Networks proposed in [1].

    Parameters
    ----------
    model : torch.nn.Module
        Model to be wrapped

    optim : torch.optim.Optimizer
        Optimizer

    root : str
        Root directory to save checkpoints

    save_basename : str, optional
        Base name of the saved checkpoints

    lr_scheduler : callable, optional
        Learning rate scheduler

    loss_function : callable, optional
        Loss function

    distance : str, optional, default='cosine'
        Type of distance function to be used for computing similarity

    device : optional
        Device on which the model is defined. If `None`, device will be
        detected automatically.


    .. admonition:: References

        1. "`Matching Networks for One Shot Learning. \
            <https://arxiv.org/abs/1606.04080>`_" Oriol Vinyals, et al. NIPS 2016.
    """

    alg_name = 'MatchNet'

    def __init__(
        self,
        model: nn.Module,
        optim: optim.Optimizer,
        root: Optional[str] = None,
        save_basename: Optional[str] = None,
        lr_scheduler: Optional[Callable] = None,
        loss_function: Optional[Callable] = None,
        distance: str = 'cosine',
        device: Optional = None
    ) -> None:
        if save_basename is None:
            save_basename = self.alg_name

        super(MatchNet, self).__init__(
            model = model,
            optim = optim,
            root = root,
            save_basename = save_basename,
            lr_scheduler = lr_scheduler,
            loss_function = loss_function,
            device = device
        )

        self.get_distance = get_distance_function(distance)

    def attention(
        self, distance: torch.FloatTensor, targets: torch.LongTensor
    ) -> torch.FloatTensor:
        """
        An attention kernel which is served as a classifier. It defines a
        probability distribution over output labels given a query example.

        The classifier output is defined as a weighted sum of labels of support
        points, and the weights should be proportional to the similarity between
        support and query embeddings.

        Parameters
        ----------
        distance : torch.FloatTensor
            Similarity between support points embeddings and query points
            embeddings, with shape ``(n_samples_query, n_samples_support)``

        targets : torch.LongTensor
            Targets of the support points, with shape ``(n_samples_support)``

        Returns
        -------
        pred_pdf : torch.FloatTensor
            Probability distribution over output labels, with shape ``(n_samples_query, n_way)``
        """
        n_samples_support = targets.size(-1)  # numper of samples in support set
        n_way = torch.unique(targets).size(0)  # number of samples per class

        softmax_pdf = f.softmax(distance, dim=1)  # (n_samples_query, n_samples_support)

        # one-hot encode labels
        targets_one_hot = softmax_pdf.new_zeros(n_samples_support, n_way)
        targets_one_hot.scatter_(1, targets.unsqueeze(-1), 1)  # (n_samples_support, n_way)

        pred_pdf = torch.mm(softmax_pdf, targets_one_hot)  # (n_samples_query, n_way)

        return pred_pdf

    def single_task(
        self, task: Tuple[torch.Tensor], meta_train: bool = True
    ) -> Tuple[float]:
        support_input, support_target, query_input, query_target = task

        with torch.set_grad_enabled(meta_train):
            support_embeddings = self.model(support_input)
            query_embeddings = self.model(query_input)

            distance = self.get_distance(support_embeddings, query_embeddings)  # (n_samples_query, n_samples_support)
            preds = self.attention(distance, support_target)
            loss = self.loss_function(preds, query_target)

        with torch.no_grad():
            accuracy = get_accuracy(preds, query_target)

        return loss, accuracy
