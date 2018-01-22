import numpy as np
import torch
from torch.autograd import Variable


class Solver(object):
    
    def __init__(self, loss_func=torch.nn.MSELoss()):
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

        # freezes pre-trained network's gradient

        optim = torch.optim.Adam(filter(lambda x: x.requires_grad,model.parameters()), lr = learning_rate)

        self.train_loss_history = []
        self.train_acc_history = []

        iter_per_epoch = len(train_data)

        if torch.cuda.is_available():
            print("Cuda available")
            model.cuda()
            
        print('START TRAIN.')
        for epoch in range(num_epochs):
          
            acc_history = []
            loss_history = []
            # TRAINING
            for i, (inputs, targets) in enumerate(train_data):
                inputs, targets = Variable(inputs.float()), Variable(targets.float())
                
                if model.is_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                optim.zero_grad()
                outputs = model(inputs) 
                outputs = outputs[:,0,0,0] # simply transformation of [x, 1, 1, 1] to [x, 1]
                loss = self.loss_func(outputs, targets)
                loss.backward()
                optim.step()

                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)
                    print('[Iteration %d/%d] TRAIN loss: %.3f' % \
                        (i + epoch * iter_per_epoch,
                         iter_per_epoch * num_epochs,
                         train_loss))

                loss_history.append(loss.data.cpu().numpy())
                acc_history += getAccuracy(targets, outputs, 10)
                
            train_acc, train_loss = np.mean(acc_history), np.mean(loss_history)
            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)

            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/loss: %.3f/%.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_loss))
                
        print('FINISH.')
        
def getAccuracy(label, output, range_percentage):
    num_examples = label.data.shape[0]
    end = label * (1 + range_percentage/100.0)
    begin = label * (1 - range_percentage/100.0)
    accuracy = [0 for x in range(num_examples)]
    for i in range(num_examples):
        if label[i].data[0] > 0 and begin[i].data[0] <= output[i].data[0] <= end[i].data[0]:
            accuracy[i] = 1
        if label[i].data[0] < 0 and end[i].data[0] <= output[i].data[0] <= begin[i].data[0]:
            accuracy[i] = 1
    return accuracy