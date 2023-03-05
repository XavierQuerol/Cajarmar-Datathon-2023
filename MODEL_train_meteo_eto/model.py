# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 22:03:45 2023

@author: xavid
"""

import torch
from torch import nn
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convs_1 = []
        for f in range(399):
            self.convs_1.append(nn.Conv1d(1,1,5, padding=2))
        
        self.convs_2 = []
        for f in range(399):
            self.convs_2.append(nn.Conv1d(1,1,5, padding=2))
            
        self.convs_3 = []
        for f in range(100):
            self.convs_3.append(nn.Conv1d(1,1,5, padding=2))
            
        self.convs_4 = []
        for f in range(100):
            self.convs_4.append(nn.Conv1d(1,1,5, padding=2))
            
        self.convs_5 = []
        for f in range(50):
            self.convs_5.append(nn.Conv1d(1,1,5, padding=2))
            
        self.convs_6 = []
        for f in range(50):
            self.convs_6.append(nn.Conv1d(1,1,5, padding=2))

        self.maxpool = nn.MaxPool2d((5,1), stride=(2,1))
                
        self.linear_1 = nn.Linear(399, 100)
        self.linear_2 = nn.Linear(100, 50)
        self.linear_3 = nn.Linear(50, 10)
        
        self.relu = nn.ReLU()
        
        self.layers = nn.Sequential(
          nn.Linear(1307, 700),
          nn.ReLU(),
          nn.Linear(700, 200),
          nn.ReLU(),
          nn.Linear(200, 100)
          )
        
        self.common_layers = nn.Sequential(
            nn.Linear(530, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU()
            )
        self.last_layer = nn.Linear(20, 1)
            
    
    def convolute(self, x, convs):
        
        out = torch.zeros(x.shape)
        
        for i, conv in enumerate(convs):
             out[:,:,:,i] = conv(x[:,:,:,i])   
        return x
    
    def mlp(self, x, linear):
        c = x.view([x.shape[0]*x.shape[2], x.shape[1], -1])
        c = linear(c)
        c = self.relu(c)
        x = c.view([x.shape[0], x.shape[1], x.shape[2], -1])
        return x
    
    def encoder_meteo_eto(self, x):
        
        x = self.convolute(x, self.convs_1)
        x = self.relu(x)
        x = self.convolute(x, self.convs_2)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.mlp(x, self.linear_1)
        
        x = self.convolute(x, self.convs_3)
        x = self.relu(x)
        x = self.convolute(x, self.convs_4)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.mlp(x, self.linear_2)
        
        x = self.convolute(x, self.convs_5)
        x = self.relu(x)
        x = self.convolute(x, self.convs_6)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.mlp(x, self.linear_3)
        
        x = torch.flatten(x,1)
        
        return x
     
    def encoder_ds_train(self, x2):
        
        x2 = self.layers(x2)
        return x2
    
    
    def forward(self, x, x2):
        
        x = x.view((x.shape[0], 1, x.shape[1], x.shape[2]))
        
        encoding_meteo_eto = self.encoder_meteo_eto(x)
        encoding_ds_train = self.encoder_ds_train(x2)
        
        x = torch.cat((encoding_meteo_eto, encoding_ds_train), dim=1)
        
        x = self.common_layers(x)
        x = self.last_layer(x)
        
        return x

