import torch
from collections import defaultdict

class Categorical:
    """
    Given k classes, map the their labels to ``[0, k)``.
    """
    def __init__(self, n_classes: int = None):
        super(Categorical, self).__init__()
        self.n_classes = n_classes
        self.labels = torch.randperm(self.n_classes).tolist()
        self.classes = defaultdict(None)

    def __call__(self, target: int):
        if (self.n_classes is not None) and (len(self.classes) > self.n_classes):
            raise ValueError('The number of labels ({0}) is greater than '
                '`n_classes` ({1}).'.format(len(self.classes), self.n_classes))

        if target not in self.classes:
            self.classes[target] = len(self.classes) if self.n_classes is None else self.labels[len(self.classes)]
        return self.classes[target]
