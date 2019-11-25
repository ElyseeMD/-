import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import pickle

"""""""""""""""""""""""""""""
Standard steps for Cifar10 DATA (32 × 32) 10 classes
"""""""""""""""""""""""""""""
# CNN Class Build
# x -- [NSample*NChannel*Height*Width]
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.conv2 = nn.Conv2d(5, 10, 5)
        #self.conv3 = nn.Conv2d(20, 4, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(10 * 24 * 24, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #conv1.append(x)
        x = F.relu(self.conv2(x))
        #conv2.append(x)
        #x = F.relu(self.conv3(x))
        #conv3.append(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

# download and load the MNIST data
batchSize = 100
loss_bound = 0.01
# transforms.Compose    combining the transforms action IN SEQUENCE!
# Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, 
#           this transform will normalize each channel of the input torch.*Tensor i.e. input[channel] = (input[channel] - mean[channel]) / std[channel] 
# ToTensor  Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = dset.CIFAR10('../CIFAR10_DATA/', train=True, transform=trans, download=True)
test_set = dset.CIFAR10('../CIFAR10_DATA/', train=False, transform=trans, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                           batch_size=batchSize, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                          batch_size=batchSize, 
                                          shuffle=False)

# training parameters
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=1e-3)

# loops for all batches and trains the network parameters (Only 1 time)
def Train(epoch):
    stop = False
    for N in range(epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if loss.item() < loss_bound:
                stop = True
                break
            if batch_idx % 200 == 0:
                print("Epoch Batch idx: {0}  Loss: {1}".format(batch_idx,loss.item()))
        if stop:
            print('Loss decreased below the given loss bound')
            break

# assign the training epochs
Train(50)


net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Extract the networks parameters (assume the layer start from 0)
# Illustration of the structure of net.parameters()
#     The sequence of the param list is: [weight1,bias1,weight2,bias2,weight3,bias3,...]
#     weight(n) has size [C(n),C(n-1)] and bias(n) has size [C(n)] where C(n) is channel number of the layer n
#     bias(n) is NOT a column vector
params = list(net.parameters())

# transform torch float tensor to numpy
Param = [p.detach().numpy() for p in params]

# save the trained model params | binary format
# load the variable in the file one by one
savePath = './ProjectDMF/TrueDataTrain/TrueParam_cifar10'
f = open(savePath,'wb')
pickle.dump(Param,f)