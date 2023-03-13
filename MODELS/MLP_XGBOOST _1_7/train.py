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
    
    
    for batch_idx, (X_meteo_eto, pred_xgboost, y) in enumerate(train_loader):
        
        data1 = X_meteo_eto.to(device)
        data2 = pred_xgboost.to(device)
        target = y.to(device)

        output ,_ = model(data1, data2)
        loss=0.0
        
        loss = loss_function(output.view([-1]), target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
        loss_values_train += loss.item()
        
    return loss_values_train/batch_idx

def train2(model_prev, model, device, train_loader, optimizer, epoch, loss_function):
    model.train()
    batch_size = train_loader.batch_size
    loss_values_train = 0
    
    
    for batch_idx, (X_meteo_eto, pred_xgboost, y) in enumerate(train_loader):
        
        
        data1 = X_meteo_eto.to(device)
        data2 = pred_xgboost.to(device)
        target = y.to(device)
        
        _, output_first_mlp = model_prev(data)
        
        output, _ = model(data1, data2)
        loss=0.0
        
        loss = loss_function(output.view([-1]), target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
        loss_values_train += loss.item()
        
    return loss_values_train/batch_idx