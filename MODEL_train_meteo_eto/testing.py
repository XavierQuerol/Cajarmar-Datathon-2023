# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 22:03:29 2023

@author: xavid
"""

import torch

def test(model, device, test_loader, MSELoss):

    loss = 0
    # Iterate through test dataset
    with torch.no_grad():
        model.eval()
        for batch_idx, (meteo_eto,ds_train,y) in enumerate(test_loader):
            data1 = meteo_eto.to(device)
            data2 = ds_train.to(device)
            target = y.to(device)

            output = model(data1, data2)
            loss += MSELoss(output, target)

        
        return loss.item()/batch_idx