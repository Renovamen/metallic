import time
import traceback
import torch
from torch import nn
from torch.autograd import grad
from metacraft.utils.module import update_module, clone_module


'''
An inner loop update step in MAML.

input params:
    model (nn.Module):
        The model to update.
    
    inner_lr (float):
        Inner loop learning rate.
    
    grad_list (list):
        A list of gradients for each parameter of the model.
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


def get_accuracy(scores, labels):
    _, predictions = scores.max(dim = 1)  # (n_samples)
    correct_predictions = torch.eq(predictions, labels).sum().float()
    accuracy = correct_predictions / labels.size(0)
    return accuracy


class MAML(nn.Module):
    '''
    An implementation of MAML (Model-Agnostic Meta-Learning):
    
    Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. Chelsea Finn, et al. ICML 2017.
    Paper: https://arxiv.org/pdf/1703.03400.pdf
    Official implementation: https://github.com/cbfinn/maml

    For FOMAML (First-Order Model-Agnostic Meta-Learning) mentioned in the
    above mentioned paper, just set `first_order = True`.

    attributes:
        model (nn.Module):
            Module to be wrapped.

        outer_optimizer (torch.optim.Optimizer):
            Optimizer for the outer loop.
        
        loss_function (callable):
            The loss function for both the inner and outer loop.
        
        inner_lr (float):
            Fast adaptation learning rate.
        
        inner_steps (int, optional, default = 1):
            The number of gradient descent updates in the inner loop.

        first_order (bool, optional, default = False):
            FOMAML (First-Order Model-Agnostic Meta-Learning)?
        
        device (torch.device`, optional, default = None):
            The device on which the model is defined.
    '''
    def __init__(self, model, outer_optimizer, loss_function, inner_lr,
                 inner_steps = 1, first_order = False, device = None):
        
        super(MAML, self).__init__()

        self.module = model
        self.outer_optimizer = outer_optimizer
        self.loss_function = loss_function
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.device = device

    
    def inner_loop(self, cloned_module, input_data):

        support_input, support_target, query_input, query_target = input_data
        # input: (n_way × k_shot, channels, img_size, img_size)
        # target: (n_way × k_shot)

        # inner loop: takes `inner_steps` gradient step
        for step in range(self.inner_steps):
            support_output = cloned_module(support_input)
            support_loss = self.loss_function(support_output, support_target)
            support_loss /= len(support_input)
            cloned_module = self.inner_loop_step(cloned_module, support_loss)

        # evaluate on the query set
        query_output = cloned_module(query_input)
        query_loss = self.loss_function(query_output, query_target)
        query_loss /= len(query_input)

        # find accuracy on query set
        query_accuracy = get_accuracy(query_output, query_target)
        
        return query_loss, query_accuracy


    '''
    Meta-train an epoch.    
    '''
    def train(self, epoch, train_loader, num_batches, print_freq = 1):

        self.module.train()

        # set start time
        start = time.time()

        # training loop
        for batch_id, batch in enumerate(train_loader):
            # data loading time per batch
            data_time = time.time() - start

            # clear gradient of last batch
            self.outer_optimizer.zero_grad()

            # support set
            support_inputs, support_targets = batch['support']
            support_inputs = support_inputs.to(self.device)
            support_targets = support_targets.to(self.device)

            # query set
            query_inputs, query_targets = batch['query']
            query_inputs = query_inputs.to(self.device)
            query_targets = query_targets.to(self.device)

            # number of tasks
            num_tasks = query_targets.size(0)

            task_batch = zip(support_inputs, support_targets, query_inputs, query_targets)

            # loss of the outer loop
            outer_loss = torch.tensor(0.).to(self.device)
            # accuracy
            outer_accuracy = torch.tensor(0.).to(self.device)
            
            for task_id, task_data in enumerate(task_batch):
                cloned_module = self.clone()  # get a cloned module
                query_loss, query_accuracy = self.inner_loop(
                    cloned_module = cloned_module,
                    input_data = task_data
                )
                query_loss.backward()
                outer_loss += query_loss.item()
                outer_accuracy += query_accuracy.item()
            
            # average the accumulated gradients
            for param in self.module.parameters():
                param.grad.data.div_(num_tasks)
            
            # outer loop update
            self.outer_optimizer.step()

            # average the losses and accuracies
            outer_loss.div_(num_tasks)
            outer_accuracy.div_(num_tasks)
            # forward prop. + back prop. time per batch
            batch_time = time.time() - start

            # reset the start time
            start = time.time()
            
            # print training status
            if batch_id % print_freq == 0:
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time:.3f}\t'
                    'Data Load Time {data_time:.3f}\t'
                    'Loss {loss:.4f}\t'
                    'Accuracy {acc:.3f}'.format(epoch, batch_id, num_batches,
                                                batch_time = batch_time,
                                                data_time = data_time, 
                                                loss = outer_loss,
                                                acc = outer_accuracy)
                )

            if batch_id >= num_batches:
                break
    
    
    '''
    Meta-validate an epoch.
    '''
    def validate(self, epoch, val_loader, num_batches, print_freq = 1):

        # set start time
        start = time.time()

        # validation loop
        for batch_id, batch in enumerate(val_loader):
            # data loading time per batch
            data_time = time.time() - start

            # support set
            support_inputs, support_targets = batch['support']
            support_inputs = support_inputs.to(self.device)
            support_targets = support_targets.to(self.device)

            # query set
            query_inputs, query_targets = batch['query']
            query_inputs = query_inputs.to(self.device)
            query_targets = query_targets.to(self.device)

            # number of tasks
            num_tasks = query_targets.size(0)

            task_batch = zip(support_inputs, support_targets, query_inputs, query_targets)

            # loss of the outer loop
            outer_loss = torch.tensor(0.).to(self.device)
            # accuracy
            outer_accuracy = torch.tensor(0.).to(self.device)
            
            for task_id, task_data in enumerate(task_batch):
                cloned_module = self.clone()  # get a cloned module
                query_loss, query_accuracy = self.inner_loop(
                    cloned_module = cloned_module,
                    input_data = task_data
                )
                outer_loss += query_loss.item()
                outer_accuracy += query_accuracy.item()

            # average the losses and accuracies
            outer_loss.div_(num_tasks)
            outer_accuracy.div_(num_tasks)
            # forward prop. + back prop. time per batch
            batch_time = time.time() - start

            # reset the start time
            start = time.time()
            
            # print training status
            if batch_id % print_freq == 0:
                print(
                    'Validation: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time:.3f}\t'
                    'Data Load Time {data_time:.3f}\t'
                    'Loss {loss:.4f}\t'
                    'Accuracy {acc:.3f}'.format(epoch, batch_id, num_batches,
                                                batch_time = batch_time,
                                                data_time = data_time, 
                                                loss = outer_loss,
                                                acc = outer_accuracy)
                )

            if batch_id >= num_batches:
                break


    '''
    Takes a gradient step on the loss and updates the cloned parameters in place.

    input params:
        loss (Tensor):
            Loss to minimize upon update.
        
        first_order: (bool, optional, default = None):
            Whether to use first- or second-order updates, defaults 
            to self.first_order.
    '''
    def inner_loop_step(self, cloned_module, loss, first_order = None):
        
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order

        gradients = grad(
            loss, cloned_module.parameters(),
            retain_graph = second_order,
            create_graph = second_order
        )
        
        # update the module
        cloned_module = maml_inner_loop_update(cloned_module, self.inner_lr, gradients)
        return cloned_module


    '''
    Returns a copy of the module whose parameters and buffers are
    `torch.clone`d from the original module.

    This implies that back-propagating losses on the cloned module will
    populate the buffers of the original module.

    For more information, refer to utils.module.clone_module().
    '''
    def clone(self):
        return clone_module(self.module)