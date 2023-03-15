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
        
        self.convs_1 = nn.ModuleList([nn.Conv1d(1,3,5, padding=2) for i in range(10)])
        self.convs_1_c = 3
        self.convs_2 = nn.ModuleList([nn.Conv1d(3,5,5, padding=2) for i in range(10)])
        self.convs_2_c = 5
        self.convs_3 = nn.ModuleList([nn.Conv1d(5,7,5, padding=2) for i in range(10)])
        self.convs_3_c = 7
        self.convs_4 = nn.ModuleList([nn.Conv1d(7,7,5, padding=2) for i in range(10)])
        self.convs_4_c = 7
        self.convs_5 = nn.ModuleList([nn.Conv1d(7,9,5, padding=2) for i in range(10)])
        self.convs_5_c = 9
        self.convs_6 = nn.ModuleList([nn.Conv1d(9,11,5, padding=2) for i in range(10)])
        self.convs_6_c = 11
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d((5,1), stride=(2,1))
        
        self.layers = nn.Sequential(
          #nn.Linear(10*(365-240) + 1306, 2000),
          
          nn.Linear(3065, 1500),
          nn.ReLU(),
          nn.Linear(1500, 800),       
          nn.ReLU(),
          nn.Linear(800, 100),
          nn.ReLU(),
          nn.Linear(100,  20),
          nn.ReLU())
        self.common_layer = nn.Linear(21,  1)
    
    def convolute(self, x, convs, channels):
        
        out = torch.zeros([x.shape[0],channels,x.shape[2],x.shape[3]])
        
        for i, conv in enumerate(convs):
             out[:,:,:,i] = conv(x[:,:,:,i])   

        return out
    
    def forward(self, x_train, meteo_eto, x_gboost):
        x_train = x_train.to(torch.float32)
        x_train = x_train.view([-1, x_train.shape[1]])
        meteo_eto = meteo_eto.view([-1, 1, meteo_eto.shape[1], meteo_eto.shape[2]])
        
        meteo_eto = self.convolute(meteo_eto, self.convs_1, self.convs_1_c)
        meteo_eto = self.relu(meteo_eto)
        meteo_eto = self.convolute(meteo_eto, self.convs_2, self.convs_2_c)
        meteo_eto = self.relu(meteo_eto)
        meteo_eto = self.maxpool(meteo_eto)
        
        meteo_eto = self.convolute(meteo_eto, self.convs_3, self.convs_3_c)
        meteo_eto = self.relu(meteo_eto)
        meteo_eto = self.convolute(meteo_eto, self.convs_4, self.convs_4_c)
        meteo_eto = self.relu(meteo_eto)
        meteo_eto = self.maxpool(meteo_eto)
        
        meteo_eto = self.convolute(meteo_eto, self.convs_5, self.convs_5_c)
        meteo_eto = self.relu(meteo_eto)
        meteo_eto = self.convolute(meteo_eto, self.convs_6, self.convs_6_c)
        meteo_eto = self.relu(meteo_eto)
        meteo_eto = self.maxpool(meteo_eto)
        
        meteo_eto = torch.flatten(meteo_eto,1)
        x_gboost = x_gboost.view([-1,1])
        x1 = torch.cat((meteo_eto, x_train), dim=1)
        x1 = self.layers(x1)
        x2 = torch.cat((x1, x_gboost), dim=1)
        x3 = self.common_layer(x2)
        return x3
        
#summary(Net())