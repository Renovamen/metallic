import os
import time
import torch
from torch import nn
from torchvision.transforms import Compose, Resize, ToTensor
import torch.backends.cudnn as cudnn

from metacraft.models import OmniglotCNN, MLP
from metacraft.metalearners import MAML

from dataloader import load_data
from opts import parse_opt


cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_accuracy(scores, labels):
    _, predictions = scores.max(dim = 1)  # (n_samples)
    correct_predictions = torch.eq(predictions, labels).sum().float()
    accuracy = correct_predictions / labels.size(0)
    return accuracy


def inner_loop(inner_steps, learner, loss_function, input_data):

    support_input, support_target, query_input, query_target = input_data

    # move to device
    support_input = support_input.to(device)  # (n_way × k_shot, channels, img_size, img_size)
    support_target = support_target.to(device)  # (n_way × k_shot)
    query_input = query_input.to(device)  # (n_way × k_shot, channels, img_size, img_size)
    query_target = query_target.to(device)  # (n_way × k_shot)

    # inner loop: takes `inner_steps` gradient step
    for step in range(inner_steps):
        support_output = learner(support_input)
        support_loss = loss_function(support_output, support_target)
        support_loss /= len(support_input)
        learner.inner_loop_step(support_loss)

    # evaluate on the query set
    query_output = learner(query_input)
    query_loss = loss_function(query_output, query_target)
    query_loss /= len(query_input)

    # find accuracy on query set
    query_accuracy = get_accuracy(query_output, query_target)
    
    return query_loss, query_accuracy


def train(config):

    train_loader, val_loader, test_loader = load_data(config)

    # set model
    model = OmniglotCNN(config.n_way)
    model.train()

    # optimizer of the outer loop
    outer_optimizer = torch.optim.Adam(model.parameters(), lr = config.outer_lr)
    # loss function
    loss_function = nn.CrossEntropyLoss()

    # move to device
    model = model.to(device)
    loss_function = loss_function.to(device)

    # set meta-learner
    maml = MAML(model, lr = config.inner_lr, first_order = False)

    start = time.time()

    # training loop
    for batch_id, batch in enumerate(train_loader):
        # data loading time per batch
        data_time = time.time() - start

        # clear gradient of last batch
        outer_optimizer.zero_grad()

        # support set
        support_inputs, support_targets = batch['support']
        support_inputs = support_inputs.to(device)
        support_targets = support_targets.to(device)

        # query set
        query_inputs, query_targets = batch['query']
        query_inputs = query_inputs.to(device)
        query_targets = query_targets.to(device)

        task_batch = zip(support_inputs, support_targets, query_inputs, query_targets)

        # loss of the outer loop
        outer_train_loss = torch.tensor(0.).to(device)
        # accuracy
        outer_train_accuracy = torch.tensor(0.).to(device)
        
        for task_id, task_data in enumerate(task_batch):
            # meta-train
            learner = maml.clone()
            query_loss, query_accuracy = inner_loop(
                inner_steps = config.inner_steps,
                learner = learner,
                loss_function = loss_function,
                input_data = task_data
            )
            query_loss.backward()
            outer_train_loss += query_loss.item()
            outer_train_accuracy += query_accuracy.item()
        
        # average the accumulated gradients
        for param in maml.parameters():
            param.grad.data.div_(config.batch_size)
        
        # outer loop update
        outer_optimizer.step()

        # average the losses and accuracies
        outer_train_loss.div_(config.batch_size)
        outer_train_accuracy.div_(config.batch_size)
        # forward prop. + back prop. time per batch
        batch_time = time.time() - start

        # reset the start time
        start = time.time()
        
        # print training status
        if batch_id % config.print_freq == 0:
            print(
                'Iter: [{0}/{1}]\t'
                'Batch Time {batch_time:.3f}\t'
                'Data Load Time {data_time:.3f}\t'
                'Loss {loss:.4f}\t'
                'Accuracy {acc:.3f}'.format(batch_id, config.num_batches,
                                            batch_time = batch_time,
                                            data_time = data_time, 
                                            loss = outer_train_loss,
                                            acc = outer_train_accuracy)
            )

        if batch_id >= config.num_batches:
            break

    # save checkpoint
    if config.checkpoint_path is not None:
        checkpoint_name = config.checkpoint_basename + '_{0}shot_{1}way.pth.tar'.format(config.n_way, config.k_shot)
        state = {
            'model': model,
            'outer_optimizer': outer_optimizer,
            'metalearner': 'maml',
            'dataset': 'omniglot',
        }
        torch.save(state, os.path.join(config.checkpoint_path, checkpoint_name))


if __name__ == '__main__':
    config = parse_opt()
    train(config)