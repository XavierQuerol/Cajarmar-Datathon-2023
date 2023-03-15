# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 22:03:25 2023

@author: xavid
"""

import torch.nn as nn
from testing import test

def train(model, device, train_loader, optimizer, epoch, loss_function):
    model.train()
    batch_size = train_loader.batch_size
    loss_values_train = 0
    
    
    for batch_idx, (X_train, meteo_eto, pred_xgboost, y) in enumerate(train_loader):
        
        data1 = X_train.to(device)
        data2 = meteo_eto.to(device)
        data3 = pred_xgboost.to(device)
        target = y.to(device)

        output = model(data1, data2, data3)
        loss=0.0
        
        loss = loss_function(output.view([-1]), target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
        loss_values_train += loss.item()
        
    return loss_values_train/batch_idx