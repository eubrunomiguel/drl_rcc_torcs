import torch
import torch.nn as nn
import torch.nn.functional as F

class Actnn(nn.Module):
    
    def __init__(self):
        super(Actnn, self).__init__()
        
        self.conv1 = nn.Conv2d(3,24,5,stride = 2)
        self.conv2 = nn.Conv2d(24,36,5,stride = 2)
        self.conv3 = nn.Conv2d(36,48,5,stride = 2)
        self.conv4 = nn.Conv2d(48,64,3)
        self.conv5 = nn.Conv2d(64,64,3)
        self.fc1 = nn.Linear(64, 1)
        '''
        self.fc2 = nn.Linear(32, 10)
        self.fc2 = nn.Linear(10, 1)
        '''
        
    def forward(self, x):
 
        out = F.relu(self.conv2(F.relu(self.conv1(x))));
        out = F.relu(self.conv4(F.relu(self.conv3(out))));
        out = F.relu(self.conv5(out));
        out = out.view(-1, self.num_flat_features(out))
        out = F.tanh(self.fc1(out))
        '''
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        '''
        
        return out
    
    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
