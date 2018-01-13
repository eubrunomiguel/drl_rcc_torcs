from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def load_state():
	IMAGE = 0
	STEER = 1

	with open('file.txt', 'rb') as file:
		racedata = plk.load(file)

	for frame in racedata:
		input = frame[IMAGE]
		label = frame[STEER]

