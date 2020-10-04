import os
import torch
from torch import nn
from utils import load_data, set_metalearner, parse_opt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(config):

    _, _, test_loader = load_data(config)

    checkpoint_name = 'best_' + config.checkpoint_basename + '_{0}shot_{1}way.pth.tar'.format(config.n_way, config.k_shot)
    checkpoint_path = os.path.join(config.checkpoint_path, checkpoint_name)

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location = str(device))
    # get model
    model = checkpoint['model']
    # get inner loop optimizer
    inner_optimizer = checkpoint['inner_optimizer']
    # get outer loop optimizer
    outer_optimizer = checkpoint['outer_optimizer']

    # loss function
    loss_function = nn.CrossEntropyLoss()

    # get meta-learner
    metalearner = set_metalearner(config, model, loss_function,
                                  inner_optimizer, outer_optimizer)

    avg_acc = metalearner.validate(0, test_loader, config.num_batches, config.print_freq)

    print('\n * TEST ACCURACY - %.1f%%\n' % (avg_acc * 100))


if __name__ == '__main__':
    config = parse_opt()
    test(config)