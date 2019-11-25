import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import os.path as IO
import numpy as np
import matplotlib.pyplot as plt

conv1 = []
conv2 = []
conv3 = []
conv4 = []
conv5 = []
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 3, 6)
        self.conv2 = nn.Conv2d(3, 5, 6)
        self.conv3 = nn.Conv2d(5, 9, 6)
        #self.conv4 = nn.Conv2d(20, 10, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(9 * 13 * 13, 1500)
        self.fc2 = nn.Linear(1500, 1500)
        self.fc3 = nn.Linear(1500, 1500)
        self.fc4 = nn.Linear(1500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #conv1.append(x)
        x = F.relu(self.conv2(x))
        #conv2.append(x)
        x = F.relu(self.conv3(x))
        #conv3.append(x)
        #x = F.relu(self.conv4(x))
        #conv4.append(x) 
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
batchSize = 50
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# print(IO.exists('../MNIST_DATA/train-labels-idx1-ubyte'))
train_set = dset.MNIST('../MNIST_DATA/', train=True, transform=trans, download=True)
test_set = dset.MNIST('../MNIST_DATA/', train=False, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                           batch_size=batchSize,   
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                          batch_size=batchSize, 
                                          shuffle=False)
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=1e-3)   
savePath = './TrueDataTrain/Param/param'
appendix = '.t7'
loss_bound = 0.01
def train(epoch): # epoch -- ensemble number
    for i in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
        #   target = target.float()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if loss.item() < loss_bound:
                print("Epoch {0}:  Batch idx: {1}  Loss: {2} -- IO".format(i,batch_idx,loss.item()))
                break
            if batch_idx % 500 == 0:
                print("Epoch {0}:  Batch idx: {1}  Loss: {2}".format(i,batch_idx,loss.item()))
        path = savePath + str(i) + appendix
        torch.save(net.state_dict(),path)

train(5)