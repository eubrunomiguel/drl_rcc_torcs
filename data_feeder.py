from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle as plk


class DrivingData(Dataset):
	def __init__(self, x, y):
		self.X = x
		self.Y = y

	def __getitem__(self, idx):
		img = self.X[idx]
		label = self.Y[idx]

		img = torch.from_numpy(img)

		# if self.transform:
		#	sample = self.transform(sample)

		return img, label

	def __len__(self):
		return len(self.Y)


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
		X = X.transpose(0, 3, 1, 2)
        

	# preprocess
	X /= 255.0
	X -= np.mean(X, axis=0)

	# subsample
	totalSamples = X.shape[0]
	num_train = int(totalSamples * (num_training_percentage / 100))
	num_validation = int(totalSamples * (num_validation_percentage / 100))

	mask = range(num_train)
	X_train = X[mask]
	y_train = Y[mask]

	mask = range(num_train, num_train + num_validation)
	X_val = X[mask]
	y_val = Y[mask]

	return DrivingData(X_train, y_train), DrivingData(X_val, y_val)


#getDrivingData("race1515861815.769681.txt")

# USAGE

# train_data, val_data = getDrivingData('race1515861815.769681.txt')
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True, num_workers=0)
#
# for i, (data, target) in enumerate(train_loader):
# 	print(target[0])