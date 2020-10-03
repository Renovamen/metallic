'''
These functions are borrowed from:
https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py
'''
import copy
import torch

'''
Update the parameters of a module in-place, in a way that preserves
differentiability:
    [ param \gets param + update ] for param in param_list

input params:
    module (nn.Module): The module to update.
    update_list (list): A list of gradients for each parameter of the model.
                        If None, will use the tensors in .update attributes.
'''
def update_module(module, update_list = None, memo = None):

    # some submodules might share parameters, so we should avoid duplicate updates
    # see: https://github.com/learnables/learn2learn/issues/174
    if memo is None:
        memo = {}

    if update_list is not None:
        param_list = list(module.parameters())
        if not len(update_list) == len(list(param_list)):
            msg = 'WARNING: update_module(): Parameters and updates have different length. ('
            msg += str(len(param_list)) + ' vs ' + str(len(update_list)) + ')'
            print(msg)
        for param, update in zip(param_list, update_list):
            param.update = update

    # first, update the params: \theta = \theta + update
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p is not None and hasattr(p, 'update') and p.update is not None:
            if p in memo:
                module._parameters[param_key] = memo[p]
            else:
                updated = p + p.update
                memo[p] = updated
                module._parameters[param_key] = updated

    # second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff is not None and hasattr(buff, 'update') and buff.update is not None:
            if buff in memo:
                module._buffers[buffer_key] = memo[buff]
            else:
                updated = buff + buff.update
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            update_list = None,
            memo = memo
        )

    # finally, rebuild the flattened parameters for RNNs
    # see this issue for more details: https://github.com/learnables/learn2learn/issues/139
    module._apply(lambda x: x)
    return module


'''
Creates a copy of a list of parameters using torch.clone().

input params:
    param_list (list): Parameters to be cloned.

return:
    the cloned parameters (list)
'''
def clone_parameters(param_list):
    return [p.clone() for p in param_list]


'''
Creates a copy of a module, whose parameters/buffers/submodules are
created using PyTorch's torch.clone().

This implies that the computational graph is kept, and you can compute
the derivatives of the new modules' parameters w.r.t the original
parameters.

input params:
    module (nn.Module): Module to be cloned.

return:
    the cloned module (nn.Module)
'''
def clone_module(module, memo = None):

    # some submodules might share parameters, so we should avoid duplicate updates
    # see: https://github.com/learnables/learn2learn/issues/174
    if memo is None:
        memo = {}

    # first, create a copy of the module, adapted from:
    # https://github.com/pytorch/pytorch/blob/65bad41cbec096aa767b3752843eddebf845726f/torch/nn/modules/module.py#L1171
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[param_ptr] = cloned

    # then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo = memo
            )

    # finally, rebuild the flattened parameters for RNNs
    # see this issue for more details:
    # https://github.com/learnables/learn2learn/issues/139
    clone = clone._apply(lambda x: x)
    return clone
