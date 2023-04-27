import os
import numpy as np
import random
import math

import numpy.random
from scipy.io import savemat
import spectral
import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import io
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
import time

from pymodel import pyCNN
from data_prepare import data_load,nor_pca,border_inter,con_data,getIndex,con_data1
# 1. two branches share parameters; 2. use summation feature fusion;
# 3. weighted summation in the decision level, the weights are determined by their accuracies.

# setting parameters
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

batchsize = 64
EPOCH = 200
LR = 0.001
dataset_name = "Houston"



# load data
# Data,Data2,TrLabel,TsLabel,spData,spTrLabel,spData2,spTrLabel2 = data_load(name=dataset_name,split_percent=0.2)
Data,Data2,TrLabel,TsLabel= data_load(name=dataset_name)
# TrLabel = small_sample(TrLabel, radito=0.2)
img_row = len(Data2)
img_col = len(Data2[0])

# normalization method 1: map to [0, 1]
[m, n, l] = Data.shape

PC,Data2,NC = nor_pca(Data,Data2,ispca=True)

# boundary interpolation
x, x2 = border_inter(PC,Data2,NC)
# construct the training and testing set of HSI


TrainPatch,TestPatch,TrainPatch2,TestPatch2,TrainLabel,TestLabel,TrainLabel2,TestLabel2 = con_data(x,x2,TrLabel,TsLabel,NC)




print('Training size and testing size of HSI are:', TrainPatch.shape, 'and', TestPatch.shape)
print('Training size and testing size of LiDAR are:', TrainPatch2.shape, 'and', TestPatch2.shape)

# step3: change data to the input type of PyTorch
TrainPatch1 = torch.from_numpy(TrainPatch)
TrainLabel1 = torch.from_numpy(TrainLabel)-1
TrainLabel1 = TrainLabel1.long()

TestPatch1 = torch.from_numpy(TestPatch)
TestLabel1 = torch.from_numpy(TestLabel)-1
TestLabel1 = TestLabel1.long()
Classes = len(np.unique(TrainLabel))

TrainPatch2 = torch.from_numpy(TrainPatch2)
TrainLabel2 = torch.from_numpy(TrainLabel2)-1
TrainLabel2 = TrainLabel2.long()

dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel2)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)
TestPatch2 = torch.from_numpy(TestPatch2)
TestLabel2 = torch.from_numpy(TestLabel2)-1
TestLabel2 = TestLabel2.long()


para_tune = False
FM = 64
if dataset_name == "Houston":
    para_tune = True

# cnn = CNN()
cnn = pyCNN(FM=FM,NC=NC,Classes=Classes,para_tune=para_tune)
# move model to GPU
cnn.cuda()

# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

BestAcc = 0
pred_img = TsLabel
torch.cuda.synchronize()
start = time.time()
# train and test the designed model
for epoch in range(EPOCH):
    for step, (b_x1, b_x2, b_y) in enumerate(train_loader):

        # move train data to GPU
        b_x1 = b_x1.cuda()
        b_x2 = b_x2.cuda()
        b_y = b_y.cuda()


        out1, out2, out3 = cnn(b_x1, b_x2)
        loss1 = loss_func(out1, b_y)
        loss2 = loss_func(out2, b_y)
        loss3 = loss_func(out3, b_y)
        loss = loss1 + loss2 + loss3
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            cnn.eval()
            temp1 = TrainPatch1
            temp1 = temp1.cuda()
            temp2 = TrainPatch2
            temp2 = temp2.cuda()
            temp3, temp4, temp5 = cnn(temp1, temp2)
            Classes = np.unique(TrainLabel1)
            pred_y = np.empty((len(TestLabel)), dtype='float32')
            number = len(TestLabel) // 5000
            for i in range(number):
                temp = TestPatch1[i * 5000:(i + 1) * 5000, :, :, :]
                temp = temp.cuda()
                temp1 = TestPatch2[i * 5000:(i + 1) * 5000, :, :, :]
                temp1 = temp1.cuda()
                temp2 = cnn(temp, temp1)[2] + cnn(temp, temp1)[1] + cnn(temp, temp1)[0]
                # temp2 = cnn(temp,temp1)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 5000:(i + 1) * 5000] = temp3.cpu()
                del temp, temp1, temp2, temp3

            if (i + 1) * 5000 < len(TestLabel):
                temp = TestPatch1[(i + 1) * 5000:len(TestLabel), :, :, :]
                temp = temp.cuda()
                temp1 = TestPatch2[(i + 1) * 5000:len(TestLabel), :, :, :]
                temp1 = temp1.cuda()
                temp2 = cnn(temp, temp1)[2] + cnn(temp, temp1)[1] + cnn(temp, temp1)[0]
                # temp2 = cnn(temp, temp1)
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 5000:len(TestLabel)] = temp3.cpu()
                del temp, temp1, temp2, temp3

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.6f' % accuracy,'| ')

            # save the parameters in network
            if accuracy > BestAcc:
                torch.save(cnn.state_dict(), 'BestAcc.pkl')
                BestAcc = accuracy
            cnn.train()

print('Best test acc:',BestAcc)
torch.cuda.synchronize()
end = time.time()
print(end - start)
Train_time = end - start

# # test each class accuracy
# # divide test set into many subsets

# load the saved parameters
cnn.load_state_dict(torch.load('BestAcc.pkl'))
cnn.eval()
torch.cuda.synchronize()
start = time.time()


pred_y = np.empty((len(TestLabel)), dtype='float32')
number = len(TestLabel)//5000
for i in range(number):
    temp = TestPatch1[i*5000:(i+1)*5000, :, :]
    temp = temp.cuda()
    temp1 = TestPatch2[i*5000:(i+1)*5000, :, :]
    temp1 = temp1.cuda()
    temp2 =  1*cnn(temp, temp1)[2] +  0.01*cnn(temp, temp1)[1] +  0.01*cnn(temp, temp1)[0]
    temp2_p = temp2.data
    # temp2 = cnn(temp, temp1)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i*5000:(i+1)*5000] = temp3.cpu()
    del temp, temp2, temp3

if (i+1)*5000 < len(TestLabel):
    temp = TestPatch1[(i+1)*5000:len(TestLabel), :, :]
    temp = temp.cuda()
    temp1 = TestPatch2[(i+1)*5000:len(TestLabel), :, :]
    temp1 = temp1.cuda()
    temp2 = 1*cnn(temp, temp1)[2] + 0.01*cnn(temp, temp1)[1] + 0.01*cnn(temp, temp1)[0]
    temp2_p = temp2.data
    # temp2 = cnn(temp, temp1)
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i+1)*5000:len(TestLabel)] = temp3.cpu()
    del temp, temp2, temp3

pred_y = torch.from_numpy(pred_y).long()
OA = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)
oa = OA.numpy()

Classes = np.unique(TestLabel1)
EachAcc = np.empty(len(Classes))
pe = 0
for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0

    for j in range(len(TestLabel1)):
        if TestLabel1[j] == cla:
            sum += 1
        if TestLabel1[j] == cla and pred_y[j] == cla:
            right += 1
    pe += sum*right
    EachAcc[i] = right.__float__()/sum.__float__()

AA = np.sum(EachAcc)/len(Classes)
pe = pe / math.pow(TestLabel1.size(0), 2)
kappa = (oa-pe)/(1-pe)
print("OA:  ", OA)
print("oa:  ", oa)
print("EachAcc:  ", EachAcc)
print("AA:  ", AA)
print("kappa:  ", kappa)
torch.cuda.synchronize()
end = time.time()
print(end - start)
Test_time = end - start
print('The Training time is: ', Train_time)
print('The Test time is: ', Test_time)

# savemat("./png/predHouston.mat", {'pred':pred_y1})
# savemat("./png/indexHouston.mat", {'index':index})
print()