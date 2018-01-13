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
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
        self.fc2 = nn.Linear(10, 1)
        
    def forward(self, x):
     
        out = F.relu(self.conv2(F.relu(self.conv1(x))));
        out = F.relu(self.conv4(F.relu(self.conv3(x))));
        out = F.relu(self.conv5(out));
        out = out.view(-1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        
        return out


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
