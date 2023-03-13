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
          nn.Linear(2559, 1750),
          nn.ReLU(),
          nn.Linear(1750, 1000),
          nn.ReLU(),
          nn.Linear(1000, 100),
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
        x2 = self.common_layer(x)
        return x2, x
    
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          #nn.Linear(10*(365-240) + 1306, 2000),
          nn.Linear(2559+21+1+1, 1750),
          nn.ReLU(),
          nn.Linear(1750, 1000),
          nn.ReLU(),
          nn.Linear(1000, 100),
          nn.ReLU(),
          nn.Linear(100,  20),
          nn.ReLU(),
          nn.Linear(20,  1))
    
        self.layers2 = nn.Sequential(
            nn.Linear(22, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            )
    
    def forward(self, x, x_xgboost, x_prev, x_sup):
        x_xgboost = x_xgboost.view([-1,1])
        x_sup = x_sup.view([-1,1])
        #x = torch.cat((x, x_xgboost, x_sup, x_prev), dim=1)
        #x = x.view([-1,x.shape[1]])
        x = torch.cat((x_sup, x_prev), dim=1)
        x = self.layers2(x)
        return x
        
#summary(Net())