# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 22:03:25 2023

@author: xavid
"""

import torch.nn as nn
from testing import test

def train(model, device, train_loader, optimizer, epoch, loss_function):
    model.train()
    loss_values_train = 0
    batch_size = train_loader.batch_size
    
    
    for batch_idx, (meteo_eto,ds_train,y) in enumerate(train_loader):
        
        data1 = meteo_eto.to(device)
        data2 = ds_train.to(device)
        target = y.to(device)

        output = model(data1, data2)
        loss=0.0
        
        loss = loss_function(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_values_train += loss.item()

        
        
    return loss_values_train/batch_idx