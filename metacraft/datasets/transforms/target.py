import torch
from collections import defaultdict

class EncodeTarget(object):
    '''
    Given samples from `n_classes` classes, maps the labels to `[0, n_classes)`.

    input params:
        n_classes (int): Number of classes.
    '''
    
    def __init__(self, n_classes):
        super(EncodeTarget, self).__init__()
        self.n_classes = n_classes
        self._classes = None
        self._labels = None

    def reset(self):
        self._classes = None
        self._labels = None

    @property
    def classes(self):
        if self._classes is None:
            self._classes = defaultdict(None)
            self._classes.default_factory = lambda: self.labels[len(self._classes)]
        return self._classes

    @property
    def labels(self):
        if (self._labels is None) and (self.n_classes is not None):
            # a random permutation of integers from 0 to n_classes - 1
            self._labels = torch.randperm(self.n_classes).tolist()
        return self._labels

    def __call__(self, target):
        return self.classes[target]

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, self.n_classes or '')


class FixTargetTransform(object):
    '''
    Record the target with its corresponding transform function.
    Only make sense for segmention tasks.

    input params:
        transform (callable):
            A function/transform that takes a `PIL` image, and returns a 
            transformed version. See also `torchvision.transforms`.
    '''

    def __init__(self, transform = None):
        self.transform = transform

    def __call__(self, index):
        '''
        input param:
            index (str): A target.

        return:
            (Tuple): (target, its corresponding transform)
        '''
        return (index, self.transform)

    def __repr__(self):
        return ('{0}({1})'.format(self.__class__.__name__, self.transform))