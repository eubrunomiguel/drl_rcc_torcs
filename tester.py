from data_feeder import *
from network import *
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

#hello

train_data, val_data = getDrivingData(speed=30, track=0)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True, num_workers=0)

network = DrivingNN()
objectivefunction = torch.nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters())

num_epochs = 100

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

for epoch in range(num_epochs):
	loss = []
	accuracy = []
	for i, (data, target) in enumerate(train_loader):
		optimizer.zero_grad()
		inputs, targets = Variable(data.float()), Variable(target.float())
		output_value = network(inputs)
		output_value = output_value[:,0,0,0]
		cost = objectivefunction(output_value, targets)
		cost.backward()
		optimizer.step()
		loss.append(cost.data[0])
		accuracy += getAccuracy(targets, output_value, 10)
	print("EPOCH %d/%d" % (epoch, num_epochs))
	loss_history.append(np.mean(loss))
	accuracy_history.append(np.mean(accuracy))


# Plot the loss function and train / validation accuracies
plt.subplots(nrows=2, ncols=1)

plt.subplot(2, 1, 1)
plt.plot(loss_history)
plt.title('Loss history')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(accuracy_history, label='train')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.tight_layout()
plt.show()
