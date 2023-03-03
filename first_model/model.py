# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 01:12:28 2023

@author: xavid
"""

from torch import nn
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(1453, 2000),
          nn.ReLU(),
          nn.Linear(2000, 1000),
          nn.ReLU(),
          nn.Linear(1000, 1000),
          nn.ReLU(),
          nn.Linear(1000, 100),
          nn.ReLU(),
          nn.Linear(100,  1)
        )
    
    def forward(self, x):
        return self.layers(x)
        
        