import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import numpy as np

#from radam import RangerLars
#from torchsummary import summary

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):

        x = x *( torch.tanh(F.softplus(x)))

        return x

mish = Mish()

class Bottleneck(nn.Module):
    def __init__(self, i, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        mish = nn.ReLU(inplace = True)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        mish = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(mish(self.bn1(x)))
        out = self.conv2(mish(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(mish(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
    def __init__(self, growthRate, reduction, nClasses):
        super(DenseNet, self).__init__()

        nChannels = 2 * growthRate
        self.conv = nn.Conv2d(1, 64, kernel_size = 7, stride = 2,
                               padding = 3, bias = False)
        self.bn = nn.BatchNorm2d(nChannels)
        mish = nn.ReLU(inplace = True)
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.dense1 = self._make_dense(nChannels, growthRate, 6)
        nChannels = nChannels * 4
        nOutChannels = nChannels // 2
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, 12)
        nChannels = nChannels * 4
        nOutChannels = nChannels // 2
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, 24)
        nChannels = nChannels * 4
        nOutChannels = nChannels // 2
        self.trans3 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, 16)
        nChannels = nChannels * 4
        nChannels = nChannels // 2
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc1 = nn.Linear(nChannels, 11)
        self.fc2 = nn.Linear(nChannels, 168)
        self.fc3 = nn.Linear(nChannels, 7)


    def _make_dense(self, nChannels, growthRate, nDenseBlocks):
        layers = []
        for i in range(nDenseBlocks):
            layers.append(Bottleneck(i+1, nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = mish(self.bn(self.conv(x)))
        out = self.max_pool(out)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        out = self.trans3(out)
        out = self.dense4(out)
        out = self.bn1(out)
        out = mish(out)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = torch.flatten(out, 1)
        x1 = self.fc1(out)
        x2 = self.fc2(out)
        x3 = self.fc3(out)
        return x1, x2, x3

model = DenseNet(growthRate = 32, reduction = 0.2, nClasses = 2)
