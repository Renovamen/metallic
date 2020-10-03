import os
import time
import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor
import torch.backends.cudnn as cudnn

from metacraft.models import OmniglotCNN, MLP
from metacraft.metalearners import MAML

from dataloader import load_data
from opts import parse_opt


cudnn.benchmark = True  # set to true only if inputs to model are fixed size;
                        # otherwise lot of computational overhead
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config):

    train_loader, val_loader, test_loader = load_data(config)

    # set model
    model = OmniglotCNN(config.n_way)

    # optimizer of the outer loop
    outer_optimizer = torch.optim.Adam(
        model.parameters(),
        lr = config.outer_lr
    )
    # loss function
    loss_function = nn.CrossEntropyLoss()

    # move to device
    model = model.to(device)
    loss_function = loss_function.to(device)

    # set meta-learner
    maml = MAML(
        model = model,
        outer_optimizer = outer_optimizer,
        loss_function = loss_function,
        inner_lr = config.inner_lr,
        inner_steps = config.inner_steps,
        first_order = False,
        device = device
    )

    for epoch in range(config.num_epoches):
        # meta-train an epoch
        maml.train(epoch, train_loader, config.num_batches, config.print_freq)
        # meta-validate an epoch
        maml.validate(epoch, val_loader, config.num_batches, config.print_freq)

        # save checkpoint
        if config.checkpoint_path is not None:
            checkpoint_name = config.checkpoint_basename + '_{0}shot_{1}way.pth.tar'.format(config.n_way, config.k_shot)
            state = {
                'model': model,
                'outer_optimizer': outer_optimizer,
                'metalearner': 'maml',
                'dataset': config.dataset,
            }
            torch.save(state, os.path.join(config.checkpoint_path, checkpoint_name))


if __name__ == '__main__':
    config = parse_opt()
    train(config)