import torch
from torch import nn
from metacraft.metalearners import MAML, FOMAML, Reptile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_metalearner(config, model, loss_function, inner_optimizer, outer_optimizer):

    # loss function
    loss_function = nn.CrossEntropyLoss()

    # move to device
    model = model.to(device)
    loss_function = loss_function.to(device)

    # set meta-learner
    if config.metalearner == 'maml':
        metalearner = MAML(
            model = model,
            outer_optimizer = outer_optimizer,
            loss_function = loss_function,
            inner_lr = config.inner_lr,
            inner_steps = config.inner_steps,
            first_order = False,
            device = device
        )
    elif config.metalearner == 'fomaml':
        metalearner = FOMAML(
            model = model,
            outer_optimizer = outer_optimizer,
            loss_function = loss_function,
            inner_lr = config.inner_lr,
            inner_steps = config.inner_steps,
            device = device
        )
    elif config.metalearner == 'reptile':
        metalearner = Reptile(
            model = model,
            inner_optimizer = inner_optimizer,
            outer_optimizer = outer_optimizer,
            loss_function = loss_function,
            inner_lr = config.inner_lr,
            inner_steps = config.inner_steps,
            device = device
        )
    
    return metalearner