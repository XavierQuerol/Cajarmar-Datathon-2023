# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 00:31:23 2023

@author: xavid
"""
import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


from dataset import MyDataset
from model import Net
from train import train
from testing import test


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   

## Read dataset
df = pd.read_csv("../ds_tractats/df_train_tractat.csv")
df = df.astype({"CAMPAÑA": str, "ID_FINCA": str, "ID_ZONA": str, "ID_ESTACION": str, "ALTITUD": str, "VARIEDAD": str, "MODO": int, "TIPO": int, "COLOR": int})
df = df.drop(columns="CAMPAÑA")
## Encoding
encoder = ce.OneHotEncoder(cols=["ID_FINCA", "ID_ZONA", "ID_ESTACION", "VARIEDAD", "ALTITUD"])
df_encoded = encoder.fit_transform(df)

#X,Y
x=df_encoded.drop(axis = 1, columns = ["PRODUCCION"])
y=df_encoded.loc[:,["PRODUCCION"]]

## Normalization
def transform(dataset, columns):
    for c in columns:
        dataset[c] = (dataset[c] - dataset[c].mean()) / dataset[c].std()
    return dataset

norm_x = transform(x, ["SUPERFICIE", "ALTITUD_MIN", "ALTITUD_DIF"]).values

## Dataset creation
dataset = MyDataset(norm_x, y.values)
dataset_train, dataset_validation = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

## Dataloader creation
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=32, shuffle=False)


## Model load
model = Net()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
    
# Applying it to our net
model.apply(initialize_weights)

for name, param in model.named_parameters():
    if str(name) == "last_layer.bias":
        param.data =torch.Tensor([df["PRODUCCION"].mean()])

## Hyperparameters definition
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 50

#optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.33)
#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(validation_loader))

loss_history_train = []
loss_history_validation = []

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_function= nn.MSELoss() 


for epoch in range(epochs):
    loss_train = train(model, device, dataloader_train, optimizer, epoch, loss_function)
    loss_validation = test(model, device, dataloader_validation, loss_function)
                
    if scheduler:
        scheduler.step()
    print('Epoch: {} \tTrainLoss: {:.6f}\tValidationLoss: {:.6f}'.format(
        epoch, 
        loss_train,
        loss_validation
        ))
    
    loss_history_train.append(loss_train)
    loss_history_validation.append(loss_validation)
    plt.plot(range(len(loss_history_train)), loss_history_train)
    plt.plot(range(len(loss_history_validation)), loss_history_validation)
    plt.show()


#%%

loss_validation = test(model, device, dataloader_validation, loss_function)

#%% DESAR WEIGHTS MODEL

PATH = r"trained_models/"
torch.save(model.state_dict(), PATH + "model1.pth")

