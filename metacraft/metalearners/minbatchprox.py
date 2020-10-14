import copy
import torch
from metacraft.metalearners.base import BaseMetaLearner
from metacraft.utils import get_accuracy

class MinibatchProx(BaseMetaLearner):
    '''
    An implementation of MinibatchProx:

    Efficient Meta Learning via Minibatch Proximal Update. Pan Zhou, et al. NIPS 2019.
    Paper: https://panzhous.github.io/assets/pdf/2019-NIPS-metaleanring.pdf
    Official implementation: https://panzhous.github.io/assets/code/MetaMinibatchProx.zip

    input params:
        model (nn.Module):
            Module to be wrapped.

        outer_optimizer (torch.optim.Optimizer):
            Optimizer for the inner loop.

        outer_optimizer (torch.optim.Optimizer):
            Optimizer for the outer loop.

        inner_lr (float):
            Fast adaptation learning rate.

        inner_steps (int, optional, default = 1):
            The number of gradient descent updates in the inner loop.

        device (torch.device, optional, default = None):
            The device on which the model is defined.
    '''

    def __init__(self, model, inner_optimizer, outer_optimizer, inner_lr,
                 inner_steps = 1, reg_lambda = 0.1, device = None):

        super(MinibatchProx, self).__init__(model, device)

        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.outer_optimizer = outer_optimizer

        self.inner_optimizer = inner_optimizer
        self.inner_optimizer_type = inner_optimizer.__class__
        self.inner_optimizer_state = self.inner_optimizer.state_dict()

        self.reg_lambda = reg_lambda


    '''
    Compute regularized cross-entropy loss using params before and after
    inner loop update: L(params) + \labmbda/2 * ||params - init_params||^2
    
    input params:
        output (torch.Tensor): Output results of the model.
        target (torch.Tensor): Ground truth.
        init_params (list): List of meta-initial parameters (before inner update).
        params (list): List of task-specific parameters (after inner update).
    '''
    def reg_loss_function(self, output, target, init_params, params):
        sq_diff = sum([((init_param - param) ** 2).sum() for init_param, param in zip(init_params, params)])
        reg_term = 0.5 * self.reg_lambda * sq_diff
        return self.loss_function(output, target) + reg_term


    def inner_loop(self, copied_module, support_input, support_target, init_params):
        '''
        Update the (copied) parameters in the inner loop (adapt stage).
        '''

        # inner loop: takes `inner_steps` gradient step
        for step in range(self.inner_steps):
            # clear gradients
            self.inner_optimizer.zero_grad()
            
            support_output = copied_module(support_input)
            
            # get task specific parameters
            params = list(copied_module.parameters())

            # compute regularized loss on support set
            support_loss = self.reg_loss_function(support_output, support_target,
                                                  init_params, params)
            
            # back propagation and update (copied) parameters
            support_loss.backward()
            self.inner_optimizer.step()

        return copied_module


    def outer_loop(self, batch, meta_train = True):
        '''
        input params:
            batch (OrderedDict):
                Input data of the current batch. See `datasets.data.splitter.MetaSplitter`
                for  details.

            meta_train (bool, optional, default = True):
                Whether we are performing meta-training?
                It should be noted that, we only update the parameters in outer loop
                during the meta-training stage (`meta_train = True`). During meta-test
                stage (`meta_train = False`), we just compute the losses and accuracies.
        '''

        # clear gradients of the last batch
        self.outer_optimizer.zero_grad()
        for param in self.module.parameters():
            param.grad = torch.zeros_like(param.data)

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

        # loss and accuracy on query set (outer loop)
        outer_loss, outer_accuracy = 0., 0.

        for task_id, task_data in enumerate(task_batch):
            # input: (n_way × k_shot, channels, img_size, img_size)
            # target: (n_way × k_shot)
            support_input, support_target, query_input, query_target = task_data
            
            # deepcopy the module
            copied_module = copy.deepcopy(self.module)
            
            # We do this because in the inner loop, we compute gradients on the
            # INITIAL parameters for all task in the same batch.
            self.inner_optimizer = self.inner_optimizer_type(
                copied_module.parameters(),
                lr = self.inner_lr
            )
            self.inner_optimizer.load_state_dict(self.inner_optimizer_state)

            # a backup copy of meta-init parameters
            init_params = [p.detach().clone().requires_grad_(True) for p in copied_module.parameters()]

            # inner loop (adapt)
            copied_module = self.inner_loop(
                copied_module = copied_module,
                support_input = support_input,
                support_target = support_target,
                init_params = init_params
            )

            # evaluate on the query set
            with torch.set_grad_enabled(meta_train):
                query_output = copied_module(query_input)
                # compute loss
                query_loss = self.loss_function(query_output, query_target)
                query_loss /= len(query_input)
                # compute accuracy
                query_accuracy = get_accuracy(query_output, query_target)

            self.inner_optimizer_state = self.inner_optimizer.state_dict()
            
            # \theta_i - \theta
            for param, new_param in zip(self.module.parameters(), copied_module.parameters()):
                param.grad.data.add_(param.data - new_param.data)

            outer_loss += query_loss.item()
            outer_accuracy += query_accuracy.item()

        # outer loop update when meta-training
        if meta_train == True:
            # (1\n) * \sum_i (\theta_i - \theta)
            for param in self.module.parameters():
                param.grad.data.div_(num_tasks)
            # Reptile: \theta ← \theta + outer_lr * \lambda * (1\n) * \sum_i (\theta_i - \theta)
            self.outer_optimizer.step()

        # average the losses and accuracies
        outer_loss /= num_tasks
        outer_accuracy /= num_tasks

        return outer_loss, outer_accuracy