import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from typing import Any, Callable
from AbstractRNNCell import AbstractRNNCell
from SineDataset import SineDataset
from CharSequenceDataset import CharSequenceDataset
from memory_task import memory_task_loss, memory_task_acc

import matplotlib.pyplot as plt

def train_loop(num_epochs, model : AbstractRNNCell, 
               init_optimizer : Callable[[Any], Optimizer], loss_func : Callable, 
               batch_size : int, dataset : Dataset) -> AbstractRNNCell:
    """
    Trains an RNN using the specified loss function and optimizer.
    Args:
        -num_epochs: the number of epochs to train for
        -model: the RNN cell
        -init_optimizer: a function accepting as arguments the model
        parameters and returning an optimizer
        -loss_func: the loss function, called on the model output and
        ground truth
        -batch_size: the batch size to be used
        -dataset: the dataset from which samples are drawn.
    Returns:
        The trained model.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    #initialize data loader and optimizer
    train_loader = DataLoader(dataset, batch_size, shuffle=True)
    optimizer = init_optimizer((model.parameters()))
    for e in range(num_epochs):

        avg_loss = 0
        num_batches = 0
        for (batch_idx, batch) in enumerate(train_loader):
            #####Start Subtask 2d#####
            #necessary to erase any leftover gradient state
            optimizer.zero_grad()
            outputs = model.forward(batch[0].to(device), reset_hidden_state=True)
            batch_loss = loss_func(outputs, batch[1].to(device))
            batch_loss.backward()
            avg_loss += batch_loss
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            #####End Subtask 2d#####
            num_batches += 1
        avg_loss /= num_batches 
        print("Epoch {0}, loss {1}".format(e, avg_loss))

    return model

def test_loop_sines(model : AbstractRNNCell, dataset : SineDataset, 
                    num_bootstrap_samples : int):
    """
    Test a trained model on sine continuation.
    Args:
        -model: the trained RNN
        -dataset: the dataset from which sine curves are drawn
        -num_bootstrap_samples: the number of initial samples provided as
        "context"/initialization to the RNN for every sequence
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #for this task we just stack everything into a single batch
    test_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    with torch.no_grad(): #run inference, hence no_grad
        for (batch_idx, batch) in enumerate(test_loader):

            input = batch[0].to(device)
            #give the RNN some "context", always maintain previous state
            for s in range(num_bootstrap_samples):
                o = model.forward(input[:, s, ...], reset_hidden_state=(s == 0))
            outputs = torch.zeros((batch[0].shape[0], 
                input.shape[1] - num_bootstrap_samples))
            #run the prediction step, each time feeding in the RNN's previous
            #output
            for s in range(input.shape[1] - num_bootstrap_samples):
                o = model.forward(o, reset_hidden_state=False)
                outputs[:, s] = o
        for seq_ind in range(batch[0].shape[0]):
            gt = batch[1][seq_ind, num_bootstrap_samples:].cpu().numpy()
            #plot the ground truth vs the prediction for all samples
            #(except bootstrapping ones)
            plt.figure()
            plt.plot(dataset.sin_arg_points[batch_idx, 1 + num_bootstrap_samples:], 
                gt, color='red', linestyle='none', marker='.', label='Ground truth')
            plt.plot(dataset.sin_arg_points[batch_idx, 1 + num_bootstrap_samples:], 
                outputs[seq_ind, :], color='blue', linestyle='none', 
                marker='.', label='Predicted')
            plt.legend()
            plt.savefig('Out_{0}_{1}.png'.format(model.get_rnn_type(), seq_ind))

def test_loop_mem_task(model : AbstractRNNCell, dataset : CharSequenceDataset,
    batch_size : int, to_remember_len : int):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(dataset, batch_size)
    avg_loss = 0
    avg_acc = 0
    num_batches = 0
    #####Start Subtask 2f#####
    with torch.no_grad(): #run inference, hence no_grad
        for (batch_idx, batch) in enumerate(test_loader):

            input = batch[0].to(device)    
            output = model.forward(input, reset_hidden_state=True)
            gt = batch[1].to(device)

            avg_loss += memory_task_loss(output, gt)
            avg_acc += memory_task_acc(output, gt, to_remember_len)
            num_batches += 1
        avg_acc /= num_batches
        avg_loss /= num_batches
        print("Average test loss across batches: {0}".format(avg_loss))
        print("Average test accuracy across batches: {0}".format(avg_acc))
    #####End Subtask 2f#####
