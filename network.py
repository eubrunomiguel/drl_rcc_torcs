import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DrivingNN(nn.Module):

	def __init__(self, num_classes=1, pretrained_model=None):
		super(DrivingNN, self).__init__()

		print("ok")

		if pretrained_model is not None:
			self.pretrained_model = pretrained_model
			for param in self.pretrained_model.parameters():
				param.requires_grad = False

		self.classifier = nn.Sequential(
			nn.Conv2d(3, 24, 5, stride=2),
			nn.ReLU(),
			nn.Conv2d(24, 36, 5, stride=2),
			nn.ReLU(),
			nn.Conv2d(36, 48, 5, stride=2),
			nn.ReLU(),
			nn.Conv2d(48, 64, 3),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3),
			nn.ReLU(),
			nn.Conv2d(64, num_classes, 1),
			# nn.Tanh()
		)

		for i in range(len(self.classifier)):
			if not str(self.classifier[i]) == "ReLU()" and not str(self.classifier[i]) == "Tanh()":
				self._bias_init(self.classifier[i])
				self._weight_init(self.classifier[i])

	def forward(self, x):
		if self.pretrained_model is not None:
			x = self.pretrained_model(x)

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
