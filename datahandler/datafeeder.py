from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle as plk

class DrivingData(Dataset):
	def __init__(self, file_name):
		with open(file_name, 'rb') as file:
			self.racedata = plk.load(file)

	def __getitem__(self, idx):
		IMAGE = 0
		STEER = IMAGE + 1
		frame = self.racedata[idx]
		input_data = frame[IMAGE]
		label_data = float(frame[STEER])

		#if self.transform:
	    #	sample = self.transform(sample)

		return (input_data, label_data)

	def __len__(self):
		return len(self.racedata)


#train_data = DrivingData('race1515861815.769681.txt')
#train_loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0)

#for i, (data, target) in enumerate(train_loader):
# 	print(target[0])