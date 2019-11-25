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

conv0 = []  # network input
conv1 = []
conv2 = []
conv3 = []
linear1 = []
linear2 = []
linear3 = []
out = []
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
        conv0.append(x)
        x = F.relu(self.conv1(x))
        conv1.append(x)
        x = F.relu(self.conv2(x))
        conv2.append(x)
        x = F.relu(self.conv3(x))
        conv3.append(x)
        #x = F.relu(self.conv4(x))
        #conv4.append(x) 
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        linear1.append(x)
        x = F.relu(self.fc2(x))
        linear2.append(x)
        x = F.relu(self.fc3(x))
        linear3.append(x)
        x = self.fc4(x)
        out.append(x)
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
loadPath = './TrueDataTrain/Param/param'
appendix = '.t7'

n = 1   # w index
net = Net()
path = loadPath + str(n) + appendix
model_dict=net.load_state_dict(torch.load(path))
for images, labels in test_loader:
    outputs = net(images)
#print(len(conv2)) # 200 batches in test_loader for batchsize = 50, eg.

# pre-define
Ensemble_num = 30
h_act0 = torch.zeros(len(conv0)*Ensemble_num, 28*28)
h_act1 = torch.zeros(len(conv1)*Ensemble_num, 23*23)
h_act2 = torch.zeros(len(conv2)*Ensemble_num, 18*18)
h_act3 = torch.zeros(len(conv3)*Ensemble_num, 13*13)


print("Input dimensionality")
for i in range(len(conv0)): # batches mumber
        for j in range(Ensemble_num): # channel idx = 1
            h_act0[i*Ensemble_num+j,:] = conv0[i][j,0].reshape(1,28*28)
# h_act0[sample index, data dimension]
h_act0_T = torch.transpose(h_act0,0,1)
CMat0 = torch.mm(h_act0_T,h_act0)/h_act0.size()[0]
h_act0_ = torch.mean(h_act0,0)
h_act0_ = h_act0_.view(1,h_act0_.size(0))  # transform to row vector
CMat0_ = torch.mm(torch.transpose(h_act0_,0,1),h_act0_)
CovMatrix0 = CMat0 - CMat0_
D0 = np.square(torch.trace(CovMatrix0).detach().numpy())/torch.trace(torch.mm(CovMatrix0,CovMatrix0)).detach().numpy()
print(D0)

print("Conv Layer 1")
# calculate dimensionality per channel
D1 = np.zeros(conv1[0][0].size(0)) # conv1[i][j]:   selected at batch [i] sample [j], with shape [Channel size * image height * image width]
for cindex in range(conv1[0][0].size(0)):
    for i in range(len(conv1)): # batches mumber
        for j in range(Ensemble_num): # channel idx = 1
            h_act1[i*Ensemble_num+j,:] = conv1[i][j,cindex].reshape(1,23*23)
    h_act1_T = torch.transpose(h_act1,0,1)
    CMat1 = torch.mm(h_act1_T,h_act1)/h_act1.size()[0]
    h_act1_ = torch.mean(h_act1,0)
    h_act1_ = h_act1_.view(1,h_act1_.size(0))
    CMat1_ = torch.mm(torch.transpose(h_act1_,0,1),h_act1_)
    CovMatrix1 = CMat1 - CMat1_
    D1[cindex] = np.square(torch.trace(CovMatrix1).detach().numpy())/torch.trace(torch.mm(CovMatrix1,CovMatrix1)).detach().numpy()
print(D1)
# calculate dimensionality per layer, big matrix
h_act_big_1 = torch.zeros(len(conv1)*Ensemble_num, 23*23*conv1[0][0].size(0))   # [sample number, data subspace dimension(channel number*image size)]
for i in range(len(conv1)):
    for j in range(Ensemble_num):
        h_act_big_1[i*Ensemble_num+j, :] = conv1[i][j].reshape(1,-1)
h_act_big_1_T = torch.transpose(h_act_big_1,0,1)
Cbig1 = torch.mm(h_act_big_1_T,h_act_big_1)/h_act_big_1.size()[0]
h_act_big_1_ = torch.mean(h_act_big_1, 0)
h_act_big_1_ = h_act_big_1_.view(1,h_act_big_1_.size(0))
Cbig1_ = torch.mm(torch.transpose(h_act_big_1_,0,1),h_act_big_1_)
CovMatrixBig1 = Cbig1 - Cbig1_
Dim1 = np.square(torch.trace(CovMatrixBig1).detach().numpy())/torch.trace(torch.mm(CovMatrixBig1,CovMatrixBig1)).detach().numpy()
print(Dim1)


print("Conv Layer 2")
D2 = np.zeros(conv2[0][0].size(0))
for cindex in range(conv2[0][0].size(0)):
    for i in range(len(conv2)):
        for j in range(Ensemble_num): # channel idx = 1
            h_act2[i*Ensemble_num+j,:] = conv2[i][j,cindex].reshape(1,18*18)
    h_act2_T = torch.transpose(h_act2,0,1)
    CMat2 = torch.mm(h_act2_T,h_act2)/h_act2.size()[0]
    h_act2_ = torch.mean(h_act2,0)
    h_act2_ = h_act2_.view(1,h_act2_.size(0))
    CMat2_ = torch.mm(torch.transpose(h_act2_,0,1),h_act2_)
    CovMatrix2 = CMat2 - CMat2_
    D2[cindex] = np.square(torch.trace(CovMatrix2).detach().numpy())/torch.trace(torch.mm(CovMatrix2,CovMatrix2)).detach().numpy()      
print(D2)
# calculate dimensionalilty per layer, big matrix
h_act_big_2 = torch.zeros(len(conv2)*Ensemble_num, 18*18*conv2[0][0].size(0))   # [sample number, data subspace dimension(channel number*image size)]
for i in range(len(conv2)):
    for j in range(Ensemble_num):
        h_act_big_2[i*Ensemble_num+j, :] = conv2[i][j].reshape(1,-1)
h_act_big_2_T = torch.transpose(h_act_big_2,0,1)
Cbig2 = torch.mm(h_act_big_2_T,h_act_big_2)/h_act_big_2.size()[0]
h_act_big_2_ = torch.mean(h_act_big_2, 0)
h_act_big_2_ = h_act_big_2_.view(1,h_act_big_2_.size(0))
Cbig2_ = torch.mm(torch.transpose(h_act_big_2_,0,1),h_act_big_2_)
CovMatrixBig2 = Cbig2 - Cbig2_
Dim2 = np.square(torch.trace(CovMatrixBig2).detach().numpy())/torch.trace(torch.mm(CovMatrixBig2,CovMatrixBig2)).detach().numpy()
print(Dim2)


print("Conv Layer 3")
D3 = np.zeros(conv3[0][0].size(0))
for cindex in range(conv3[0][0].size(0)):
    for i in range(len(conv3)):
        for j in range(Ensemble_num): # channel idx = 1
            h_act3[i*Ensemble_num+j,:] = conv3[i][j,cindex].reshape(1,13*13)
    h_act3_T = torch.transpose(h_act3,0,1)
    CMat3 = torch.mm(h_act3_T,h_act3)/h_act3.size()[0]
    h_act3_ = torch.mean(h_act3,0)
    h_act3_ = h_act3_.view(1,h_act3_.size(0))
    CMat3_ = torch.mm(torch.transpose(h_act3_,0,1),h_act3_)
    CovMatrix3 = CMat3 - CMat3_
    D3[cindex] = np.square(torch.trace(CovMatrix3).detach().numpy())/torch.trace(torch.mm(CovMatrix3,CovMatrix3)).detach().numpy()       
print(D3)
# calculate dimensionalilty per layer, big matrix
h_act_big_3 = torch.zeros(len(conv3)*Ensemble_num, 13*13*conv3[0][0].size(0))   # [sample number, data subspace dimension(channel number*image size)]
for i in range(len(conv3)):
    for j in range(Ensemble_num):
        h_act_big_3[i*Ensemble_num+j, :] = conv3[i][j].reshape(1,-1)
h_act_big_3_T = torch.transpose(h_act_big_3,0,1)
Cbig3 = torch.mm(h_act_big_3_T,h_act_big_3)/h_act_big_3.size()[0]
h_act_big_3_ = torch.mean(h_act_big_3, 0)
h_act_big_3_ = h_act_big_3_.view(1,h_act_big_3_.size(0))
Cbig3_ = torch.mm(torch.transpose(h_act_big_3_,0,1),h_act_big_3_)
CovMatrixBig3 = Cbig3 - Cbig3_
Dim3 = np.square(torch.trace(CovMatrixBig3).detach().numpy())/torch.trace(torch.mm(CovMatrixBig3,CovMatrixBig3)).detach().numpy()
print(Dim3)
del conv0, conv1, conv2, conv3

# FC net part
linear_act1 = torch.zeros(len(linear1)*Ensemble_num, 1500)  # [sample number, data dimension]
linear_act2 = torch.zeros(len(linear2)*Ensemble_num, 1500)
linear_act3 = torch.zeros(len(linear3)*Ensemble_num, 1500)
out_act = torch.zeros(len(out)*Ensemble_num, 10)

for i in range(len(linear1)):
    for j in range(Ensemble_num):
        linear_act1[i*Ensemble_num+j, :] = linear1[i][j].reshape(1,-1)
for i in range(len(linear2)):
    for j in range(Ensemble_num):
        linear_act2[i*Ensemble_num+j, :] = linear2[i][j].reshape(1,-1)
for i in range(len(linear3)):
    for j in range(Ensemble_num):
        linear_act3[i*Ensemble_num+j, :] = linear3[i][j].reshape(1,-1)
for i in range(len(out)):
    for j in range(Ensemble_num):
        out_act[i*Ensemble_num+j, :] = out[i][j].reshape(1,-1)


print("FC Layer 1")
linear_act1_T = torch.transpose(linear_act1,0,1)
Clinear1 = torch.mm(linear_act1_T,linear_act1)/linear_act1.size()[0]
linear_act1_ = torch.mean(linear_act1, 0)
linear_act1_ = linear_act1_.view(1,linear_act1_.size(0))
Clinear1_ = torch.mm(torch.transpose(linear_act1_,0,1),linear_act1_)
CovMatrixLinear1 = Clinear1 - Clinear1_
Diml1 = np.square(torch.trace(CovMatrixLinear1).detach().numpy())/torch.trace(torch.mm(CovMatrixLinear1,CovMatrixLinear1)).detach().numpy()
print(Diml1)

print("FC Layer 2")
linear_act2_T = torch.transpose(linear_act2,0,1)
Clinear2 = torch.mm(linear_act2_T,linear_act2)/linear_act2.size()[0]
linear_act2_ = torch.mean(linear_act2, 0)
linear_act2_ = linear_act2_.view(1,linear_act2_.size(0))
Clinear2_ = torch.mm(torch.transpose(linear_act2_,0,1),linear_act2_)
CovMatrixLinear2 = Clinear2 - Clinear2_
Diml2 = np.square(torch.trace(CovMatrixLinear2).detach().numpy())/torch.trace(torch.mm(CovMatrixLinear2,CovMatrixLinear2)).detach().numpy()
print(Diml2)

print("FC Layer 3")
linear_act3_T = torch.transpose(linear_act3,0,1)
Clinear3 = torch.mm(linear_act3_T,linear_act3)/linear_act3.size()[0]
linear_act3_ = torch.mean(linear_act3, 0)
linear_act3_ = linear_act3_.view(1,linear_act3_.size(0))
Clinear3_ = torch.mm(torch.transpose(linear_act3_,0,1),linear_act3_)
CovMatrixLinear3 = Clinear3 - Clinear3_
Diml3 = np.square(torch.trace(CovMatrixLinear3).detach().numpy())/torch.trace(torch.mm(CovMatrixLinear3,CovMatrixLinear3)).detach().numpy()
print(Diml3)

# output layer
print("Output Layer")
out_act_T = torch.transpose(out_act,0,1)
Cout = torch.mm(out_act_T,out_act)/out_act.size()[0]
out_act_ = torch.mean(out_act, 0)
out_act_ = out_act_.view(1,out_act_.size(0))
Cout_ = torch.mm(torch.transpose(out_act_,0,1),out_act_)
CovMatrixOut = Cout - Cout_
DimOut = np.square(torch.trace(CovMatrixOut).detach().numpy())/torch.trace(torch.mm(CovMatrixOut,CovMatrixOut)).detach().numpy()
print(DimOut)

# plot dimension
xLayer = range(8)
yDim = [D0, Dim1, Dim2, Dim3, Diml1, Diml2, Diml3, DimOut]
plt.figure(num=1)
plt.plot(xLayer, yDim, color='red', linewidth=3.0)
dot_area = 70*np.ones([1, 8])
plt.scatter(xLayer, yDim, s=dot_area, alpha=0.8)
plt.xlabel('layer index', fontsize=22)
plt.ylabel('dimensionality', fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

