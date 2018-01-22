from network import *
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle as plk
import torch.nn.functional as F

def getDrivingData(file_name, num_training_percentage=80, num_validation_percentage=20, dtype=np.float32):
	"""
	Load and preprocess the training dataset.
	Transpose image data from H, W, C to C, H, W and group as N, H, W, C.
	Rescale the features and subtract the mean.
	Return a tuble of Dataset objects, in respect to <training:validation>.
	"""
	with open(file_name, 'rb') as file:
		racedata = plk.load(file)
		X, Y = zip(*racedata)
		X = np.array(X)
		Y = np.array(Y)

	return X, Y

X, Y = getDrivingData("race1516207298.txt")

num_examples = X.shape[0]

X = np.array(X)
Y = np.array(Y)

print(X.shape)

img = X[0,:,:,:]
plt.imshow(img, origin='lower')
plt.draw()
plt.pause(1)
