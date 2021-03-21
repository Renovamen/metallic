from numbers import Number
from collections import OrderedDict
import numpy as np
import torch

def get_accuracy(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy using predicted scores and targets."""
    _, predictions = scores.max(dim = 1)  # (n_samples)
    correct_predictions = torch.eq(predictions, targets).sum().float()
    accuracy = correct_predictions / targets.size(0)
    return accuracy

class MetricTracker:
    """Keep track of metrics."""

    def __init__(self, *names) -> None:
        self._metrics = OrderedDict()
        for name in names:
            self.add(name)

    def add(self, name: str) -> None:
        """Add a new metric."""
        if name in self._metrics:
            warnings.warn('The metric `{}` already exists in the tracking'
                'list. To avoid duplication, this metric is ignored'.format(name),
                UserWarning, stacklevel=2
            )
        else:
            self._metrics[name] = np.array([])

    def reset(self) -> None:
        """Clear all of the recorded metrics."""
        for name in self.metrics.keys():
            self._metrics[name] = np.array([])

    def update(self, name: str, value: Number) -> None:
        """Update a new value to a specified metric."""
        self._metrics[name] = np.append(self._metrics[name], value)

    @property
    def metrics(self) -> dict:
        """Return a ``dict`` containing all metrics."""
        return self._metrics
