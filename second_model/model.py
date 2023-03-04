# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 22:03:45 2023

@author: xavid
"""

import torch
from torch import nn
from torchsummary import summary

class Net(nn.Module):
    def __init__(self, days, features):
        super().__init__()
        self.days = days
        self.features = features
        self.convs = []
        for i in range(features):
            self.convs.append(nn.Conv1d(1,1,2))     
        self.maxpool = nn.MaxPool1d(2)
    
    def forward(self, x):
        out = torch.zeros((x.shape[0], 1, self.features))
        for i in range(self.features):
            out[:,:,i] = self.convs[i](x[:,:,i].view(-1,1,2)).view(-1,1)
            #out[:,i] = self.maxpool(tmp.t()).t()
        return out
        
summary(Net(1, 10))

c = torch.Tensor([[[1,2,1,1],[2,2,5,2]], [[1,2,2,4],[2,2,2,4]]])
a = Net(2,4)
print(a(c))