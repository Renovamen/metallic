import torch
from torch import nn
from metacraft.metalearners import MAML, FOMAML, Reptile, MinibatchProx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_metalearner(config, model, inner_optimizer, outer_optimizer):

    # set meta-learner
    if config.metalearner == 'maml':
        metalearner = MAML(
            model = model,
            outer_optimizer = outer_optimizer,
            inner_lr = config.inner_lr,
            inner_steps = config.inner_steps,
            first_order = False,
            device = device
        )
    elif config.metalearner == 'fomaml':
        metalearner = FOMAML(
            model = model,
            outer_optimizer = outer_optimizer,
            inner_lr = config.inner_lr,
            inner_steps = config.inner_steps,
            device = device
        )
    elif config.metalearner == 'reptile':
        metalearner = Reptile(
            model = model,
            inner_optimizer = inner_optimizer,
            outer_optimizer = outer_optimizer,
            inner_lr = config.inner_lr,
            inner_steps = config.inner_steps,
            device = device
        )
    elif config.metalearner == 'minibatchprox':
        metalearner = MinibatchProx(
            model = model,
            inner_optimizer = inner_optimizer,
            outer_optimizer = outer_optimizer,
            inner_lr = config.inner_lr,
            inner_steps = config.inner_steps,
            reg_lambda = config.reg_lambda,
            device = device
        )
    else:
        raise NotImplementedError("Meta-learner not implemented.")
    
    return metalearner