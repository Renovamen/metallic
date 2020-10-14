import time
from torch import nn
from metacraft.utils import TrackMetric


class BaseMetaLearner(object):

    def __init__(self, model, device = None):

        super(BaseMetaLearner, self).__init__()

        self.device = device
        self.module = model.to(device)
        self.loss_function = nn.CrossEntropyLoss().to(device)
        

    def outer_loop(self):
        raise NotImplementedError()


    def train(self, epoch, train_loader, num_batches = 500, print_freq = 1):
        '''
        Meta-train an epoch.
        '''

        # self.module.train()

        batch_time = TrackMetric()  # forward prop. + back prop. time
        data_time = TrackMetric()  # data loading time per batch
        outer_losses = TrackMetric()  # losses
        outer_accuracies = TrackMetric()  # accuracies

        # set start time
        start = time.time()

        # training loop
        for batch_id, batch in enumerate(train_loader):
            # data loading time per batch
            data_time.update(time.time() - start)

            # perform outer loop and get loss and accuracy
            outer_loss, outer_accuracy = self.outer_loop(batch)

            # track average outer loss and accuracy
            outer_losses.update(outer_loss)
            outer_accuracies.update(outer_accuracy)

            # track average forward prop. + back prop. time per batch
            batch_time.update(time.time() - start)

            # reset the start time
            start = time.time()

            # print training status
            if (batch_id + 1) % print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch + 1, batch_id + 1, num_batches,
                                                                    batch_time = batch_time,
                                                                    data_time = data_time,
                                                                    loss = outer_losses,
                                                                    acc = outer_accuracies)
                )

            if (batch_id + 1) >= num_batches:
                break


    def validate(self, epoch, val_loader, num_batches = 500, print_freq = 1):
        '''
        Meta-validate an epoch.
        '''

        # self.module.eval()

        batch_time = TrackMetric()  # forward prop. time
        outer_losses = TrackMetric()  # losses
        outer_accuracies = TrackMetric()  # accuracies

        # set start time
        start = time.time()

        # validation loop
        for batch_id, batch in enumerate(val_loader):
            # perform outer loop and get loss and accuracy
            outer_loss, outer_accuracy = self.outer_loop(batch, meta_train = False)

            # track average outer loss and accuracy
            outer_losses.update(outer_loss)
            outer_accuracies.update(outer_accuracy)

            # track average forward prop. time per batch
            batch_time.update(time.time() - start)

            # reset the start time
            start = time.time()

            # print training status
            if (batch_id + 1) % print_freq == 0:
                print(
                    'Validation: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch + 1, batch_id + 1, num_batches,
                                                                    batch_time = batch_time,
                                                                    loss = outer_losses,
                                                                    acc = outer_accuracies)
                )

            if (batch_id + 1) >= num_batches:
                break
        
        return outer_accuracies.avg