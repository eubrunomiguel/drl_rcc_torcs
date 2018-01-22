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

def rgb2gray(rgb):
    """Convert RGB image to grayscale

      Parameters:
        rgb : RGB image

      Returns:
        gray : grayscale image

    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

for i in range(num_examples):
	X[i, :, :, 0] = rgb2gray(X[i])

for i in range(num_examples):
	img = X[i,:,:,:]
	plt.imshow(img, origin='lower')
	plt.draw()
	plt.pause(1)
	break
