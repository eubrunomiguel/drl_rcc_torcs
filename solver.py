import numpy as np
import torch
from torch.autograd import Variable


class Solver(object):
    
    def __init__(self, loss_func=torch.nn.MSELoss(),size_average=False):
        self.loss_func = loss_func
        

    def train(self, model, train_data, learning_rate=1e-2, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_data: train data
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = torch.optim.Adam(model.parameters(), lr = learning_rate)
        self.train_loss_history = []
        self.train_acc_history = []
        iter_per_epoch = len(train_data)

        if torch.cuda.is_available():
            model.cuda()
            
        print('START TRAIN.')
        for epoch in range(num_epochs):
            
            # TRAINING
            for i, (inputs, targets) in enumerate(train_data):
                inputs, targets = Variable(inputs.float()), Variable(targets.float())
                
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
                    
                end, begin = 1.1 * outputs, 0.9*outputs
                mask1, mask2 = targets >= begin, targets <=end
                
                train_acc = np.mean((mask1*mask2).data.cpu().numpy())
                self.train_acc_history.append(train_acc)
                
            atrain_acc, atrain_loss = np.mean(self.train_acc_history), np.mean(self.train_loss_history)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   atrain_acc,
                                                                   atrain_loss))
                
        print('FINISH.')
