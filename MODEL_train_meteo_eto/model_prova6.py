# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 02:35:55 2023

@author: xavid
"""

import torch
from torch import nn
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.convs_1 = nn.ModuleList([nn.Conv1d(1,3,5, padding=2) for i in range(55)])
        self.convs_1_c = 3
        self.convs_2 = nn.ModuleList([nn.Conv1d(3,3,5, padding=2) for i in range(55)])
        self.convs_2_c = 3
        self.convs_3 = nn.ModuleList([nn.Conv1d(3,3,5, padding=2) for i in range(55)])
        self.convs_3_c = 3
        self.convs_4 = nn.ModuleList([nn.Conv1d(3,3,5, padding=2) for i in range(55)])
        self.convs_4_c = 3
        self.convs_5 = nn.ModuleList([nn.Conv1d(3,3,5, padding=2) for i in range(10)])
        self.convs_5_c = 3
        self.convs_6 = nn.ModuleList([nn.Conv1d(3,3,5, padding=2) for i in range(10)])
        self.convs_6_c = 3
        self.convs_7 = nn.ModuleList([nn.Conv1d(3,3,5, padding=2) for i in range(10)])
        self.convs_7_c = 3
        self.convs_8 = nn.ModuleList([nn.Conv1d(3,3,5, padding=2) for i in range(10)])
        self.convs_8_c = 3

        self.maxpool = nn.MaxPool2d((10,1), stride=(3,1))
                
        self.linear_1 = nn.ModuleList([nn.Linear(55, 10) for i in range(3)])
        self.features_1 = 10
        self.linear_2 = nn.ModuleList([nn.Linear(10, 5) for i in range(3)])
        self.features_2 = 5

        self.relu = nn.ReLU()

        self.layers = nn.Sequential(
          nn.Linear(1307, 700),
          nn.ReLU(),
          nn.Linear(700, 200),
          nn.ReLU(),
          nn.Linear(200, 100)
          )
        
        self.common_layers = nn.Sequential(
            nn.Linear(130, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            nn.ReLU()
            )
        self.last_layer = nn.Linear(20, 1)
   
    
    def convolute(self, x, convs, channels):
        
        out = torch.zeros([x.shape[0],channels,x.shape[2],x.shape[3]])
        
        for i, conv in enumerate(convs):
             out[:,:,:,i] = conv(x[:,:,:,i])   

        return out
    
    def mlp(self, x, linears, features):

        c = x.view([x.shape[0]*x.shape[2], x.shape[1], -1])
        out = torch.zeros([x.shape[0]*x.shape[2], x.shape[1], features])
        for i, linear in enumerate(linears):
            aux = linear(c[:,i,:])
            aux = self.relu(aux)
            out[:,i,:] = aux
        x = out.view([x.shape[0], x.shape[1], x.shape[2], -1])
        return x
    
    def encoder_meteo_eto(self, x):
        
        x = self.convolute(x, self.convs_1, self.convs_1_c)
        x = self.relu(x)
        x = self.convolute(x, self.convs_2, self.convs_2_c)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.convolute(x, self.convs_3, self.convs_3_c)
        x = self.relu(x)
        x = self.convolute(x, self.convs_4, self.convs_4_c)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.mlp(x, self.linear_1, self.features_1)
        
        x = self.convolute(x, self.convs_5, self.convs_5_c)
        x = self.relu(x)
        x = self.convolute(x, self.convs_6, self.convs_6_c)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.convolute(x, self.convs_7, self.convs_7_c)
        x = self.relu(x)
        x = self.convolute(x, self.convs_8, self.convs_8_c)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.mlp(x, self.linear_2, self.features_2)
        
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
    
#summary(Net())