#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


# A small multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


# A small cnn model for mnist dataset
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class AlexNet(nn.Module):
    def __init__(self,args):
        super(AlexNet, self).__init__()
        self.Conv = nn.Sequential(
            # IN : 3*32*32
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=5,stride=2,padding=2),      # 论文中kernel_size = 11,stride = 4,padding = 2
            nn.ReLU(),
            # IN : 96*16*16
            nn.MaxPool2d(kernel_size=2,stride=2),              # 论文中为kernel_size = 3,stride = 2
            # IN : 96*8*8
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # IN :256*8*8
            nn.MaxPool2d(kernel_size=2,stride=2),              # 论文中为kernel_size = 3,stride = 2
            # IN : 256*4*4
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.MaxPool2d(kernel_size=2, stride=2),              # 论文中为kernel_size = 3,stride = 2
            # OUT : 384*2*2
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=32 * 2 * 2, out_features=2500),
            nn.ReLU(),
            nn.Linear(in_features=2500, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=args.num_classes),
        )
    def forward(self,x):
            x = self.Conv(x)
            x = x.view(-1, 64 * 2 * 2)
            x = self.linear(x)
            return x

# a small cnn model for cifar10 dataset

class CNNCifar100(nn.Module):
    def __init__(self, args):
        super(CNNCifar100, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=2,padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 2 * 2, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))
        
        x = (F.relu(self.conv3(x)))

        x = (F.relu(self.conv4(x)))

        x = self.pool(F.relu(self.conv5(x)))
        
        x = x.view(-1, 256 * 2 * 2)
        
        x = F.relu(self.fc1(x))
        
        x1 = F.relu(self.fc2(x))

        x1 = F.dropout(x1, training=self.training)

        x2 = self.fc3(x1)
        
        return x2,x2


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 16 * 5 * 5)
        
        x = F.relu(self.fc1(x))
        
        x1 = F.relu(self.fc2(x))

        x2 = F.dropout(x1, training=self.training)

        x2 = self.fc3(x2)
        
        return x2

   

class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, in_planes, planes, stride=1):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    
class Bottleneck(nn.Module):


    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
  

# architecture for resnet  

class ResNet(nn.Module):

    """ 
    Class for ResNet architecture - Standard Architecture

    Resnet with 18, 34, 50, 101 and 152 layers can be declared. 

    We used ResNet 34 for our experiments. 

    """

    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, 5089)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, training=self.training)
        out = self.linear(out)        
        return out

    def ResNet18():
        return ResNet(BasicBlock, [2,2,2,2])

    def ResNet34(args):
        return ResNet(BasicBlock, [3,4,6,3])

    def ResNet50(args):
        return ResNet(Bottleneck, [3,4,6,3])

    def ResNet101(args):
        return ResNet(Bottleneck, [3,4,23,3])

    def ResNet152(args):
        return ResNet(Bottleneck, [3,8,36,3])