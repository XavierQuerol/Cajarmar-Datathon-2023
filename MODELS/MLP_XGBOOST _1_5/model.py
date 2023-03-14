# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 23:23:24 2023

@author: xavid
"""

import torch
from torch import nn
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          #nn.Linear(10*(365-240) + 1306, 2000),
          nn.Linear(2855, 1500),
          nn.ReLU(),
          nn.Linear(1500, 800),       
          nn.ReLU(),
          nn.Linear(800, 100),
          nn.ReLU(),
          nn.Linear(100,  20),
          nn.ReLU())
        self.common_layer = nn.Linear(21,  1)
    
    def forward(self, x, x_gboost):
        x = x.to(torch.float32)
        x = x.view([-1,x.shape[1]])
        x_gboost = x_gboost.view([-1,1])
        x = self.layers(x)
        x = torch.cat((x, x_gboost), dim=1)
        x = self.common_layer(x)
        return x
        
#summary(Net())