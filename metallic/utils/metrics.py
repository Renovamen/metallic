import torch

def accuracy(scores: torch.Tensor, targets: torch.Tensor) -> float:
    _, predictions = scores.max(dim = 1)  # (n_samples)
    correct_predictions = torch.eq(predictions, targets).sum().float()
    accuracy = correct_predictions / targets.size(0)
    return accuracy

class TrackMetric:
    """ Keep track of the most recent, average, sum, and count of a metric."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
