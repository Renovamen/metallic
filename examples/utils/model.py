from torch import optim, nn
from metacraft.models import OmniglotCNN, MLP

def set_model(config):

    if config.model == 'omniglotcnn':
        model = OmniglotCNN(config.n_way)
    
    # loss function
    loss_function = nn.CrossEntropyLoss()

    # optimizer
    if config.metalearner == 'maml':
        inner_optimizer = None
        # optimizer of the outer loop
        outer_optimizer = optim.Adam(model.parameters(), lr = config.outer_lr)
    elif config.metalearner == 'fomaml':
        inner_optimizer = None
        # optimizer of the outer loop
        outer_optimizer = optim.Adam(model.parameters(), lr = config.outer_lr)
    elif config.metalearner == 'reptile':
        # optimizer of the inner loop
        inner_optimizer = optim.Adam(model.parameters(), lr = config.inner_lr)
        # optimizer of the outer loop
        outer_optimizer = optim.SGD(model.parameters(), lr = config.outer_lr)

    return model, loss_function, inner_optimizer, outer_optimizer