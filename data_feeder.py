from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle as plk
import matplotlib.pyplot as plt


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


def getDrivingData(speed=0, track=0, num_training_percentage=80, num_validation_percentage=20, dtype=np.float32, augumentation=True):
	"""
	Load and preprocess the training dataset.
	Transpose image data from H, W, C to C, H, W and group as N, H, W, C.
	Rescale the features and subtract the mean.
	Return a tuble of Dataset objects, in respect to <training:validation>.
	If speed is set and track not set, then return all tracks with that speed
	If track is set and speed not set, then return all data from the track
	If speed and track are not set, return all data
	If speed and track are set, return the specific file
	"""

	tracks = [0, 1, 2, 3, 4]
	speeds = [0, 30, 40, 50, 60, 70, 80, 90]
	filenames = []

	if speed not in speeds or track not in tracks:
		print("Data could not be found for track %d and speed %d" % (track, speed))
		exit()

	# If speed and track are set, return the specific file
	if speed is not 0 and track is not 0:
		filenames.append("racingdata/#track=%d#speed=%d.txt" % (track, speed))

	# If speed is set and track not set, then return all tracks with that speed
	if speed is not 0 and track is 0:
		for t in tracks:
			if t == 0:
				continue
			filenames.append("racingdata/#track=%d#speed=%d.txt" % (t, speed))

	# If track is set and speed not set, then return all data from the track
	if speed is 0 and track is not 0:
		for s in speeds:
			if s == 0:
				continue
			filenames.append("racingdata/#track=%d#speed=%d.txt" % (track, s))

	# If speed and track are not set, return all data
	if speed is 0 and track is 0:
		for t in tracks:
			for s in speeds:
				if t == 0 or s == 0:
					continue
				filenames.append("racingdata/#track=%d#speed=%d.txt" % (t, s))


	X = []
	Y = []

	for filename in filenames:
		with open(filename, 'rb') as file:
			racedata = plk.load(file)
			x, y = zip(*racedata)
			X += x
			Y += y

	X = np.array(X)
	Y = np.array(Y)

	# subsample masks
	totalSamples = X.shape[0]
	num_train = int(totalSamples * (num_training_percentage / 100))
	num_validation = int(totalSamples * (num_validation_percentage / 100))

	# training load
	mask = range(num_train)
	X_train = X[mask]
	y_train = Y[mask]

	if augumentation:
		X_train_flipped = np.flip(X_train, 2)
		X_train = np.concatenate((X_train, X_train_flipped), 0)
		y_train = np.concatenate((y_train, y_train), 0)


	# validation load
	mask = range(num_train, num_train + num_validation)
	X_val = X[mask]
	y_val = Y[mask]

	# preprocess
	X /= 255.0
	X -= np.mean(X, axis=0)

	# Transpose so that channels come first
	X = X.transpose(0, 3, 1, 2)

	print("Number of training examples %d" % (X_train.shape[0]))

	return DrivingData(X_train, y_train), DrivingData(X_val, y_val)


#getDrivingData("race1515861815.769681.txt")

# USAGE

# train_data, val_data = getDrivingData('race1515861815.769681.txt')
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True, num_workers=0)
#
# for i, (data, target) in enumerate(train_loader):
# 	print(target[0])
