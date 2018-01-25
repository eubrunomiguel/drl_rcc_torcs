import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable




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


    solver = Solver(loss_func = lossfunc)
    accuracy_history, loss_history = solver.train(network, train_loader, num_epochs=100, learning_rate=1e-3)


