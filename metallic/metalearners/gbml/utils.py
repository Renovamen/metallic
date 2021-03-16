from torch import nn

def apply_grads(model: nn.Module, grads: list) -> None:
    """Map a list of gradients to a model."""
    if not len(grads) == len(list(model.parameters())):
        msg = 'WARNING: Parameters and gradients have different length. ('
        msg += str(len(list(model.parameters()))) + ' vs ' + str(len(grads)) + ')'
        print(msg)

    for param, grad in zip(model.parameters(), grads):
        if grad is not None:
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad += grad.clone()
