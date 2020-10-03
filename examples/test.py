import os
import torch
from torch import nn
from utils.metalearner import set_metalearner
from utils.dataloader import load_data
from utils.opts import parse_opt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(config):

    _, _, test_loader = load_data(config)

    checkpoint_name = 'best_' + config.checkpoint_basename + '_{0}shot_{1}way.pth.tar'.format(config.n_way, config.k_shot)
    checkpoint_path = os.path.join(config.checkpoint_path, checkpoint_name)

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location = str(device))
    
    # get model
    model = checkpoint['model']
    
    # get outer loop optimizer
    outer_optimizer = checkpoint['outer_optimizer']

    # loss function
    loss_function = nn.CrossEntropyLoss()

    # get meta-learner
    metalearner = set_metalearner(config, model, outer_optimizer, loss_function)

    avg_acc = metalearner.validate(0, test_loader, config.num_batches, config.print_freq)

    print('\n * TEST ACCURACY - %.1f percent\n' % (avg_acc * 100))


if __name__ == '__main__':
    config = parse_opt()
    test(config)