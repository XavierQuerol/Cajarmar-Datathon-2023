# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 01:29:03 2023

@author: xavid
"""

import torch.nn as nn
from testing import test

def train(model, device, train_loader, optimizer, epoch, loss_function, scheduler=None):
    model.train()
    loss_values_train = 0
    batch_size = train_loader.batch_size
    
    
    for batch_idx, (x,y) in enumerate(train_loader):
        
        data = x.to(device)
        target = y.to(device)

        output = model(data)
        loss=0.0
        
        loss = loss_function(output.view([-1]), target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_values_train += loss.item()
            
          
        if scheduler is not None:
            scheduler.step()
        
        
    return loss_values_train/batch_idx