import time
from typing import Optional
from torch import nn

from ..data import MetaDataLoader
from ..metalearners import GBML
from ..utils.metrics import TrackMetric

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
            to be trained on
    """

    def __init__(
        self,
        metalearner: GBML,
        train_loader: MetaDataLoader,
        val_loader: Optional[MetaDataLoader] = None,
        n_epoches: int = 100,
        n_batches: int = 500,
        print_freq: int = 100
    ):
        self.metalearner = metalearner
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epoches = n_epoches
        self.n_batches = n_batches
        self.print_freq = print_freq

        self.n_way = self.train_loader.dataset.n_way
        self.k_shot = self.train_loader.dataset.task_splits['support']

    def train(self, epoch: int):
        """
        Meta-train for an epoch.
        """
        batch_time = TrackMetric()  # forward prop. + back prop. time
        data_time = TrackMetric()  # data loading time per batch
        outer_losses = TrackMetric()  # losses
        outer_accuracies = TrackMetric()  # accuracies

        # reset the start time
        start = time.time()

        # training loop
        for batch_id, batch in enumerate(self.train_loader):
            # data loading time per batch
            data_time.update(time.time() - start)

            # perform outer loop and get loss and accuracy
            outer_loss, outer_accuracy = self.metalearner.outer_loop(batch)

            # track average outer loss and accuracy
            outer_losses.update(outer_loss)
            outer_accuracies.update(outer_accuracy)

            # track average forward prop. + back prop. time per batch
            batch_time.update(time.time() - start)

            # reset the start time
            start = time.time()

            # print training status
            if (batch_id + 1) % self.print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch + 1, batch_id + 1, self.n_batches,
                        batch_time = batch_time,
                        data_time = data_time,
                        loss = outer_losses,
                        acc = outer_accuracies
                    )
                )

            if (batch_id + 1) >= self.n_batches:
                break

        return outer_accuracies.avg


    def test(self, epoch: int):
        """
        Meta-test/validate for an epoch.
        """
        batch_time = TrackMetric()  # forward prop. + back prop. time
        data_time = TrackMetric()  # data loading time per batch
        outer_losses = TrackMetric()  # losses
        outer_accuracies = TrackMetric()  # accuracies

        # reset the start time
        start = time.time()

        # validation loop
        for batch_id, batch in enumerate(self.val_loader):
            # data loading time per batch
            data_time.update(time.time() - start)

            # perform outer loop and get loss and accuracy
            outer_loss, outer_accuracy = self.metalearner.outer_loop(batch, meta_train=False)

            # track average outer loss and accuracy
            outer_losses.update(outer_loss)
            outer_accuracies.update(outer_accuracy)

            # track average forward prop. time per batch
            batch_time.update(time.time() - start)

            # reset the start time
            start = time.time()

            # print training status
            if (batch_id + 1) % self.print_freq == 0:
                print(
                    'Validation: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch + 1, batch_id + 1, self.n_batches,
                        batch_time = batch_time,
                        loss = outer_losses,
                        acc = outer_accuracies
                    )
                )

            if (batch_id + 1) >= self.n_batches:
                break

        return outer_accuracies.avg

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
