import numpy as np
import torch
from torch.autograd import Variable


class Solver(object):
    default_sgd_args = {"lr": 1e-4,
                         "momentum": 0.99,
                         "weight_decay": 5e-4}
    
    def __init__(self, optim=torch.optim.SGD, optim_args={},
                 loss_func=torch.nn.MSELoss(),size_average=False):
        optim_args_merged = self.default_sgd_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        

    def train(self, model, train_data, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_data: train data
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim()
        self._reset_histories()
        iter_per_epoch = len(train_data)

        if torch.cuda.is_available():
            model.cuda()
            
        print('START TRAIN.')
        for epoch in range(num_epochs):
            
            # TRAINING
            for i, (inputs, targets) in enumerate(train_data):
                inputs, targets = Variable(inputs), Variable(targets)
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs = model(inputs)               
                loss = self.loss_func(outputs, targets)
                
                loss.backward()
                optim.step()

                self.train_loss_history.append(loss.data.cpu().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f' % \
                        (i + epoch * iter_per_epoch,
                         iter_per_epoch * num_epochs,
                         train_loss))
                
            atrain_loss = np.mean(np.mean(self.train_loss_history)
            if log_nth:
                print('[Epoch %d/%d] TRAIN loss: %.3f' % (epoch + 1,num_epochs,atrain_loss))
                   
        print('FINISH.')
