from data_feeder import *
from network import *
import numpy as np
import torch
from torch.autograd import Variable
# import matplotlib.pyplot as plt


train_data, val_data = getDrivingData("racingdata/#track=1#speed=30.txt")
train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True, num_workers=0)

network = DrivingNN()
objectivefunction = torch.nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters())

# plt.axis([0, 10, 0, 1])
# plt.ion()

def getAccuracy(label, output, range_percentage):
	num_examples = label.shape[0]
	end = label * (1 + range_percentage/100.0)
	begin = label * (1 - range_percentage/100.0)
	accuracy = [0 for x in range(num_examples)]
	for i in range(num_examples):
		if label[i].data[0] > 0 and begin[i].data[0] <= output[i].data[0] <= end[i].data[0]:
			accuracy[i] = 1
		if label[i].data[0] < 0 and end[i].data[0] <= output[i].data[0] <= begin[i].data[0]:
			accuracy[i] = 1
	return accuracy

accuracy_history = []
loss_history = []

for epoch in range(1):
	loss_history = []
	accuracy = []
	for i, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		inputs, targets = Variable(data.float()), Variable(target.float())
		output_value = network(inputs)
		output_value = output_value[:,0,0,0]
		cost = objectivefunction(output_value, targets)
		cost.backward()
		optimizer.step()
		loss_history.append(cost.data[0])
		accuracy += getAccuracy(targets, output_value, 10)
	print(np.mean(accuracy))



