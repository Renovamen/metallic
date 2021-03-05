import time
from typing import Optional
from torch import nn

from ..data import MetaDataLoader
from ..metalearners import GBML
from ..utils import MetricTracker, Logger

class GBMLTrainer:
    """
    A wrap of training procedure for gradient-based meta-learning algorithms.

    Args:
        metalearner (GBML): An instance of :class:`~metallic.metalearners.GBML`
            class
        train_loader (MetaDataLoader): Train data loader, an instance of
            :class:`~metallic.data.dataloader.MetaDataLoader` class
        val_loader (MetaDataLoader, optional): Validation data loader, an
            instance of :class:`~metallic.data.dataloader.MetaDataLoader` class
        n_epoches (int, optional, default=100): Number of epoches
        n_batches (int, optional, default=500): Number of the batches in an epoch
        logger (Logger, optional): An instance of
            :class:`~metallic.utils.logger.Logger` class
    """

    def __init__(
        self,
        metalearner: GBML,
        train_loader: MetaDataLoader,
        val_loader: Optional[MetaDataLoader] = None,
        n_epoches: int = 100,
        n_batches: int = 500,
        logger: Optional[Logger] = None
    ):
        self.metalearner = metalearner
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epoches = n_epoches
        self.n_batches = n_batches
        self.logger = logger

        self.n_way = self.train_loader.dataset.n_way
        self.k_shot = self.train_loader.dataset.task_splits['support']

    def train(self, epoch: int):
        """
        Meta-train for an epoch.
        """

        tracker = MetricTracker('batch_time', 'data_time', 'loss', 'accuracy')

        # reset the start time
        start = time.time()

        # training loop
        for batch_id, batch in enumerate(self.train_loader):
            # data loading time per batch
            tracker.update('data_time', time.time() - start)

            # perform outer loop and get loss and accuracy
            outer_loss, outer_accuracy = self.metalearner.outer_loop(batch)

            # track average outer loss and accuracy
            tracker.update('loss', outer_loss)
            tracker.update('accuracy', outer_accuracy)

            # track average forward prop. + back prop. time per batch
            tracker.update('batch_time', time.time() - start)

            # reset the start time
            start = time.time()

            # log training status
            if self.logger is not None:
                self.logger.log(tracker.metrics, epoch + 1, batch_id + 1)

            if (batch_id + 1) >= self.n_batches:
                break

        return tracker('accuracy').mean()


    def test(self, epoch: int):
        """
        Meta-test/validate for an epoch.
        """

        tracker = MetricTracker('batch_time', 'data_time', 'loss', 'accuracy')

        # reset the start time
        start = time.time()

        # validation loop
        for batch_id, batch in enumerate(self.val_loader):
            # data loading time per batch
            tracker.update('data_time', time.time() - start)

            # perform outer loop and get loss and accuracy
            outer_loss, outer_accuracy = self.metalearner.outer_loop(batch, meta_train=False)

            # track average outer loss and accuracy
            tracker.update('loss', outer_loss)
            tracker.update('accuracy', outer_accuracy)

            # track average forward prop. time per batch
            tracker.update('batch_time', time.time() - start)

            # reset the start time
            start = time.time()

            # print test status
            if self.logger is not None:
                self.logger.log(tracker.metrics, epoch + 1, batch_id + 1)

            if (batch_id + 1) >= self.n_batches:
                break

        return tracker('accuracy').mean()

    def run_train(self):
        """
        Run training procedure.
        """
        best_acc = 0.

        for epoch in range(self.n_epoches):
            # meta-train an epoch
            recent_acc = self.train(epoch)

            # meta-validate an epoch, get the average accuracy over all batches
            if self.val_loader is not None:
                recent_acc = self.test(epoch)

            # if the current model achieves the best accuracy
            is_best = recent_acc > best_acc
            best_acc = max(recent_acc, best_acc)

            # save checkpoint
            if self.metalearner.root and self.metalearner.save_basename:
                self.metalearner.save('{0}shot_{1}way_'.format(self.n_way, self.k_shot))
                # If this checkpoint is the best so far, store a copy so it
                # doesn't get overwritten by a worse checkpoint.
                if is_best:
                    best_path = self.metalearner.save(
                        'best_{0}shot_{1}way_'.format(self.n_way, self.k_shot)
                    )
                    print('Saved the current best checkpoint to: {0}.'.format(best_path))
