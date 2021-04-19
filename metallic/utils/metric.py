from numbers import Number
from typing import Union, Dict
from collections import OrderedDict
import numpy as np
import torch

def get_accuracy(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy using predicted scores and targets."""
    _, predictions = scores.max(dim = 1)  # (n_samples)
    correct_predictions = torch.eq(predictions, targets).sum().float()
    accuracy = correct_predictions / targets.size(0)
    return accuracy


class Metric:
    """Keep track of a single metric."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._data = np.array([])

    def reset(self) -> None:
        """Clear all of the recorded data."""
        self._data = np.array([])

    def update(self, name: str, value: Union[Number, np.number]) -> None:
        """Record a new value."""
        self._data = np.append(self._data, value)

    @property
    def mean(self) -> np.number:
        """Return the average value of the collected data."""
        return self._data.mean()

    @property
    def std(self) -> np.number:
        """Return the std value of the collected data."""
        return self._data.std()

    @property
    def max(self) -> np.number:
        """Return the maxinum value of the collected data."""
        if self._data.shape[0] == 0:
            return -np.inf
        else:
            return self._data.max()

    @property
    def min(self) -> np.number:
        """Return the minimum value of the collected data."""
        if self._data.shape[0] == 0:
            return np.inf
        else:
            return self._data.min()

    @property
    def recent(self) -> np.number:
        """Return the recent recorded data."""
        return self._data[-1]


class MetricTracker:
    """Keep track of metrics."""

    def __init__(self, *names) -> None:
        self._metrics = OrderedDict()
        for name in names:
            self.add(name)

    def add(self, name: str) -> None:
        """Add a new metric."""
        if name in self._metrics:
            warnings.warn(
                'The metric `{}` already exists in the tracking list. To avoid '
                'duplication, this metric is ignored'.format(name),
                UserWarning, stacklevel=2
            )
        else:
            self._metrics[name] = Metric(name)

    def reset(self) -> None:
        """Clear all of the recorded metrics."""
        for name in self.metrics.keys():
            self._metrics[name].reset()

    def update(self, name: str, value: Union[Number, np.number]) -> None:
        """Update a new value to a specified metric."""
        self._metrics[name].update(name, value)

    @property
    def metrics(self) -> Dict[str, Metric]:
        """Return a ``dict`` containing all metrics."""
        return self._metrics

    def __getitem__(self, index: str) -> Metric:
        return self._metrics[index]
