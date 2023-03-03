# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:31:23 2023

@author: xavid
"""
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


from dataset import MyDataset
from model import Net
from train import train



df = pd.read_csv("./datasets/D_T_encoded_3.csv")

## Dataset creation
dataset = MyDataset(df)
dataset_train, dataset_validation = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

## Dataloader creation
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=32, shuffle=False)


## Model load
model = Net()


## Hyperparameters definition
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 50

#optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.33)
#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(test_loader))

loss_history_train = []
loss_history_validation = []
loss_history_test = []

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


for epoch in range(1, epochs + 1):
    loss_values_train, loss_values_validation = train(model, device, dataloader_train, dataloader_validation, optimizer, epoch, scheduler)
    loss_history_train += loss_values_train
    loss_history_validation += loss_values_validation 



#%% DESAR WEIGHTS MODEL

PATH = r"trained_models/"
torch.save(model.state_dict(), PATH + "model1.pth")