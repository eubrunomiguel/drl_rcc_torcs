import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


weightinit = False
grayscale = False
augmentation = True
preprocess = True
lossfunc = torch.nn.MSELoss(size_average=True)

from network import *
import torchvision.models as models

vgg16 = models.vgg16(pretrained=True)
vgg16 = torch.nn.Sequential(*list(vgg16.features.children()))
network = DrivingNN(pretrained_model=vgg16, grayscale=grayscale, weight_init=weightinit)

from data_feeder import *
train_data, val_data = getDrivingData(speed=0, track=0, preprocess=preprocess, greyscale=grayscale, augmentation=augmentation)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True, num_workers=4)

from solver import *

try:
    solver = Solver(loss_func = lossfunc)
    accuracy_history, loss_history = solver.train(network, train_loader, num_epochs=100, learning_rate=1e-3)
except KeyboardInterrupt:
#     model.save("models/segmentation_nn.model")
    pass

# Plot the loss function and train / validation accuracies
plt.subplots(nrows=2, ncols=1)

plt.subplot(2, 1, 1)
plt.plot(loss_history)
plt.title('Loss history')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(accuracy_history, label='train')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.tight_layout()
plt.show()