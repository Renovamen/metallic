'''
MAML (Model-Agnostic Meta-Learning):

Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. Chelsea Finn, et al. ICML 2017.
Paper: https://arxiv.org/pdf/1703.03400.pdf
Official implementation: https://github.com/cbfinn/maml
'''

import traceback
from torch import nn
from torch.autograd import grad
from metacraft.utils.module import update_module, clone_module


'''
An inner loop update step in MAML.

input params:
    model (nn.Module): The model to update.
    inner_lr (float): Inner loop learning rate.
    grad_list (list): A list of gradients for each parameter of the model.
'''
def maml_inner_loop_update(model, inner_lr, grad_list):
    param_list = list(model.parameters())
    
    # sanity check
    if not len(grad_list) == len(list(param_list)):
        msg = 'WARNING: inner_loop_update(): Parameters and gradients have different length. ('
        msg += str(len(param_list)) + ' vs ' + str(len(grad_list)) + ')'
        print(msg)
    
    # update = - \alpha \nabla_{\theta} L_i(\theta)
    for param, grad in zip(param_list, grad_list):
        if grad is not None:
            param.update = - inner_lr * grad
    
    return update_module(model)


class MAML(nn.Module):
    '''
    attributes:
        model (nn.Module): Module to be wrapped.
        lr (float): Fast adaptation learning rate.
        first_order (bool): FOMAML?
        allow_unused (bool): Whether to allow differentiation of unused parameters.
        allow_nograd (bool): Whether to allow adaptation with parameters that have `requires_grad = False`.
    '''
    def __init__(self, model, lr, first_order = False, 
                 allow_unused = None, allow_nograd = False):
        
        super(MAML, self).__init__()

        self.module = model
        self.lr = lr

        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused


    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


    '''
    Takes a gradient step on the loss and updates the cloned parameters in place.

    input params:
        loss (Tensor): Loss to minimize upon update.
        first_order: (bool): Whether to use first- or second-order updates, 
                             defaults to self.first_order.
        allow_unused (bool): Whether to allow differentiation of unused 
                             parameters, defaults to self.allow_unused.
        allow_nograd (bool): Whether to allow adaptation with parameters that 
                             have `requires_grad = False`. Defaults to 
                             self.allow_nograd.
    '''
    def inner_loop_step(self, loss, first_order = None, allow_unused = None, allow_nograd = None):
        
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        if allow_nograd:
            # compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(
                loss, diff_params,
                retain_graph = second_order,
                create_graph = second_order,
                allow_unused = allow_unused
            )

            gradients = []
            grad_counter = 0
            # handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(
                    loss, self.module.parameters(),
                    retain_graph = second_order,
                    create_graph = second_order,
                    allow_unused = allow_unused
                )
            except RuntimeError:
                traceback.print_exc()
                print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')

        # update the module
        self.module = inner_loop_update(self.module, self.lr, gradients)


    def train_epoch(self, dataloader, num_batches = 500, **kwargs):
        with tqdm(total = num_batches, **kwargs) as pbar:
            for results in self.train_iter(dataloader, num_batches = num_batches):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    postfix['accuracy'] = '{0:.4f}'.format(np.mean(results['accuracies_after']))
                pbar.set_postfix(**postfix)


    '''
    Returns a `MAML`-wrapped copy of the module whose parameters and buffers
    are `torch.clone`d from the original module.

    This implies that back-propagating losses on the cloned module will
    populate the buffers of the original module.
    For more information, refer to learn2learn.clone_module().

    input param:
        first_order (bool): Whether the clone uses first- or second-order 
                            updates. Defaults to self.first_order.
        allow_unused (bool): Whether to allow differentiation of unused 
                             parameters. Defaults to self.allow_unused.
        allow_nograd (bool): Whether to allow adaptation with parameters that 
                             have `requires_grad = False`. Defaults to 
                             self.allow_nograd.

    '''
    def clone(self, first_order = None, allow_unused = None, allow_nograd = None):

        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(
            clone_module(self.module),
            lr = self.lr,
            first_order = first_order,
            allow_unused = allow_unused,
            allow_nograd = allow_nograd
        )