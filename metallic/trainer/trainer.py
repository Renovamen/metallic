import time
from tqdm import tqdm
from typing import Optional
from torch import nn

from ..data import MetaDataLoader
from ..metalearners import MetaLearner
from ..utils import MetricTracker, Logger

class Trainer:
    """
    A wrap of training procedure for meta-learning algorithms.

    Args:
        metalearner (MetaLearner): An instance of :class:`~metallic.metalearners.MetaLearner`
            class
        train_loader (MetaDataLoader): Train data loader, an instance of
            :class:`~metallic.data.dataloader.MetaDataLoader` class
        val_loader (MetaDataLoader, optional): Validation data loader, an
            instance of :class:`~metallic.data.dataloader.MetaDataLoader` class
        n_epoches (int, optional, default=100): Number of epoches
        n_iters_per_epoch (int, optional, default=500): Number of the iterations
            per epoch
        n_iters_test (int, optional, default=600): Number of the iterations
            during meta-test stage
        logger (Logger, optional): An instance of
            :class:`~metallic.utils.logger.Logger` class
    """

    def __init__(
        self,
        metalearner: MetaLearner,
        train_loader: MetaDataLoader,
        val_loader: Optional[MetaDataLoader] = None,
        n_epoches: int = 100,
        n_iters_per_epoch: int = 500,
        n_iters_test: int = 600,
        logger: Optional[Logger] = None
    ):
        self.metalearner = metalearner
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epoches = n_epoches
        self.n_iters_per_epoch = n_iters_per_epoch
        self.n_iters_test = n_iters_test
        self.logger = logger

        self.n_way = self.train_loader.dataset.n_way
        self.k_shot = self.train_loader.dataset.task_splits['support']

    def save(self, is_best: bool = False):
        """Save checkpoints."""
        if self.metalearner.root and self.metalearner.save_basename:
            self.metalearner.save('{0}shot_{1}way_'.format(self.n_way, self.k_shot))
            # If this checkpoint is the best so far, store a copy so it
            # doesn't get overwritten by a worse checkpoint.
            if is_best:
                best_path = self.metalearner.save(
                    'best_{0}shot_{1}way_'.format(self.n_way, self.k_shot)
                )
                print('Saved the current best checkpoint to: {0}.'.format(best_path))

    def lr_schedule(self):
        """Schedule learning rate."""
        if self.metalearner.lr_scheduler:
            self.metalearner.lr_schedule()

    def run_epoch(self, epoch: int, train: bool = True):
        """Train or evaluate an epoch."""
        tracker = MetricTracker('batch_time', 'data_time', 'loss', 'accuracy')

        # reset the start time
        start = time.time()

        loader = self.train_loader if train else self.val_loader
        n_iters = self.n_iters_per_epoch if train else self.n_iters_test
        stage = 'meta-train' if train else 'meta-test'

        for i_iter, batch in tqdm(
            enumerate(loader), total=n_iters, desc=f"Epoch [{epoch}] ({stage})"
        ):
            # data loading time per batch
            tracker.update('data_time', time.time() - start)

            # get loss and accuracy
            loss, accuracy = self.metalearner.step(batch, meta_train=train)

            # track average loss and accuracy
            tracker.update('loss', loss)
            tracker.update('accuracy', accuracy)

            # track average forward prop. + back prop. time per batch
            tracker.update('batch_time', time.time() - start)

            # reset the start time
            start = time.time()

            # log training status
            if self.logger is not None:
                self.logger.log(tracker.metrics, epoch, i_iter + 1, stage)

            if (i_iter + 1) >= n_iters:
                break

        return tracker.metrics['accuracy'].mean()

    def run_train(self):
        """Run training procedure."""
        best_acc = 0.

        for epoch in range(1, self.n_epoches + 1):
            # meta-train an epoch
            recent_acc = self.run_epoch(epoch, train=True)

            # meta-test an epoch, get the average accuracy over all batches
            if self.val_loader is not None:
                recent_acc = self.run_epoch(epoch, train=False)

            # if the current model achieves the best accuracy
            is_best = recent_acc > best_acc
            best_acc = max(recent_acc, best_acc)

            # save checkpoint
            self.save(is_best)

            # schedule learning rate
            self.lr_schedule()
