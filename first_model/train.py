# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 01:29:03 2023

@author: xavid
"""

import torch.nn as nn
from testing import test

def train(model, device, train_loader, validation_loader, optimizer, epoch, log_interval, scheduler=None):
    model.train()
    loss_values_train = []
    loss_values_validation = []
    MSELoss= nn.MSELoss()  
    batch_size = train_loader.batch_size
    
    
    for batch_idx, (x,y) in enumerate(train_loader):
        
        optimizer.zero_grad()
        
        data = x.to(device)
        target = y.to(device)

        output = model(data)
        loss=0.0
        
        loss = MSELoss(output.view([-1]), target)
        
        loss_values_train.append(loss)
        loss.backward()
        optimizer.step()
        
        if batch_idx * batch_size % 4000 == 0:   
            
            loss_validation = test(model, device, validation_loader, MSELoss)
            
            loss_values_validation.append(loss_validation)
            
            print('Epoch: {} [{}/{} ({:.0f}%)]\tTrainLoss: {:.6f}\tValidationLoss: {:.6f}%'.format(
                epoch, 
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item(),
                loss_validation.item()
                ))
            
          
        if scheduler is not None:
            scheduler.step()
    
    return loss_values_train, loss_values_validation