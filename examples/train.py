import sys

sys.path.insert(0, '/Users/zou/Desktop/metacraft')

import os
import torch
from torch import nn
import torch.backends.cudnn as cudnn

from metacraft.models import OmniglotCNN, MLP
from metacraft.metalearners import MAML

from utils.metalearner import set_metalearner
from utils.dataloader import load_data
from utils.opts import parse_opt


cudnn.benchmark = True  # set to true only if inputs to model are fixed size;
                        # otherwise lot of computational overhead

def train(config):

    train_loader, val_loader, _ = load_data(config)

    # set model
    model = OmniglotCNN(config.n_way)

    # optimizer of the outer loop
    outer_optimizer = torch.optim.Adam(
        model.parameters(),
        lr = config.outer_lr
    )

    # loss function
    loss_function = nn.CrossEntropyLoss()

    # get meta-learner
    metalearner = set_metalearner(config, model, outer_optimizer, loss_function)

    best_acc = 0.

    for epoch in range(config.num_epoches):
        # meta-train an epoch
        metalearner.train(epoch, train_loader, config.num_batches, config.print_freq)

        # meta-validate an epoch, and get the average accuracy over all batches
        recent_acc = metalearner.validate(epoch, val_loader, config.num_batches, config.print_freq)

        # if the current model achieves the best accuracy
        is_best = recent_acc > best_acc
        best_acc = max(recent_acc, best_acc)

        # save checkpoint
        if config.checkpoint_path is not None:
            checkpoint_name = config.checkpoint_basename + '_{0}shot_{1}way.pth.tar'.format(config.n_way, config.k_shot)
            checkpoint_path = os.path.join(config.checkpoint_path, checkpoint_name)
            state = {
                'model': model,
                'outer_optimizer': outer_optimizer,
                'metalearner': 'maml',
                'dataset': config.dataset,
            }
            torch.save(state, checkpoint_path)

            # If this checkpoint is the best so far, store a copy so it doesn't
            # get overwritten by a worse checkpoint.
            if is_best:
                best_checkpoint_name = 'best_' + config.checkpoint_basename + '_{0}shot_{1}way.pth.tar'.format(config.n_way, config.k_shot)
                best_checkpoint_path = os.path.join(config.checkpoint_path, best_checkpoint_name)
                print('Saving the current best checkpoint to: {0}...'.format(best_checkpoint_path))
                torch.save(state, best_checkpoint_path)


if __name__ == '__main__':
    config = parse_opt()
    train(config)