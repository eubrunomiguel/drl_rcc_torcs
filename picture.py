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

X, Y = getDrivingData("#track=1#speed=30#dim=2.txt")

num_examples = X.shape[0]

X = np.array(X)
Y = np.array(Y)

# Y = np.sort(Y)
# Histogram
# heights,bins = np.histogram(Y,bins=50)
# # Normalize
# heights = heights/float(sum(heights))
# binMids=bins[:-1]+np.diff(bins)/2.
# plt.plot(binMids,heights)
# plt.show()

img = X[0,:,:,:]
plt.imshow(img, origin='lower')
plt.draw()
plt.pause(1)
