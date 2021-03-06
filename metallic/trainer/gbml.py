import time
from tqdm import tqdm
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
        n_iters_per_epoch (int, optional, default=500): Number of the iterations
            per epoch
        n_iters_test (int, optional, default=600): Number of the iterations
            during meta-test stage
        logger (Logger, optional): An instance of
            :class:`~metallic.utils.logger.Logger` class
    """

    def __init__(
        self,
        metalearner: GBML,
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

    def train(self, epoch: int):
        """
        Meta-train an epoch.
        """
        tracker = MetricTracker('batch_time', 'data_time', 'loss', 'accuracy')

        # reset the start time
        start = time.time()

        # meta-training loop
        for i_iter, batch in tqdm(
            enumerate(self.train_loader), total=self.n_iters_per_epoch, desc=f"Epoch [{epoch}] (meta-train)"
        ):
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
                self.logger.log(tracker.metrics, epoch, i_iter + 1)

            if (i_iter + 1) >= self.n_iters_per_epoch:
                break

        return tracker.metrics['accuracy'].mean()


    def test(self, epoch: int):
        """
        Meta-test an epoch.
        """
        tracker = MetricTracker('batch_time', 'data_time', 'loss', 'accuracy')

        # reset the start time
        start = time.time()

        # meta-test loop
        for i_iter, batch in tqdm(
            enumerate(self.val_loader), total=self.n_iters_test, desc=f"Epoch [{epoch}] (meta-test)"
        ):
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
                self.logger.log(tracker.metrics, epoch, i_iter + 1)

            if (i_iter + 1) >= self.n_iters_test:
                break

        return tracker.metrics['accuracy'].mean()

    def run_train(self):
        """
        Run training procedure.
        """
        best_acc = 0.

        for epoch in range(1, self.n_epoches + 1):
            # meta-train an epoch
            recent_acc = self.train(epoch)

            # meta-test an epoch, get the average accuracy over all batches
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
