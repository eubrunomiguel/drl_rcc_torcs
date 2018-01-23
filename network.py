import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class DrivingNN(nn.Module):

	def __init__(self, num_classes=1, pretrained_model=None, grayscale=False, weight_init=True):
		super(DrivingNN, self).__init__()

		self.grayscale = grayscale

		self.expander = nn.Conv2d(1,3,3, padding=1);

		if pretrained_model is not None:
			self.pretrained_model = pretrained_model
			for param in self.pretrained_model.parameters():
				param.requires_grad = False

		self.classifier = nn.Sequential(
			nn.Linear(512*2*2, 1024),
			# nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			# nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512, 128),
			# nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Linear(128, num_classes),
			# nn.Tanh()
		)

		if weight_init:
			print("Initializing weights")
			for i in range(len(self.classifier)):
				if not str(self.classifier[i]) == "ReLU()" and not str(self.classifier[i]) == "Tanh()" and "BatchNorm1d" not in str(self.classifier[i]):
					self._bias_init(self.classifier[i])
					self._weight_init(self.classifier[i])

	def forward(self, x):
		if self.grayscale:
			x = self.expander(x)

		if self.pretrained_model is not None:
			x = self.pretrained_model(x)
		x = x.view(-1, self.num_flat_features(x))
		x = self.classifier(x)
		return x

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

	def _bias_init(self, b):
		b.bias = torch.nn.Parameter(torch.zeros(len(b.bias)))

	def _weight_init(self, m):
		size = m.weight.size()
		fan_out = size[0]  # number of rows
		fan_in = size[1]  # number of columns
		variance = np.sqrt(2.0 / (fan_in + fan_out))
		m.weight.data.normal_(0.0, variance)

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
