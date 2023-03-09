# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 07:40:28 2023

@author: xavid
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 22:03:07 2023

@author: xavid
"""

import pandas as pd
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
import sys
import time


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


from dataset import MyDataset
from model7 import Net
from train import train
from testing import test
from read_tables import read_tables

file1 = "/home/caos/datathon/MODEL_train_meteo_eto/datasets/df_train_tractat.csv"
#file2 = "/home/caos/datathon/MODEL_train_meteo_eto/datasets/df_meteo_eto.csv"
file2 = "/home/caos/datathon/MODEL_train_meteo_eto/datasets/df_meteo_eto_57features.csv"

df = pd.read_csv(file1)
df = df.astype({"CAMPAÑA": str, "ID_FINCA": str, "ID_ZONA": str, "ID_ESTACION": str, "ALTITUD": str, "VARIEDAD": str, "MODO": int, "TIPO": int, "COLOR": int})

df = df.drop(df[(df["CAMPAÑA"]=="14")|(df["CAMPAÑA"]=="15")].index).reset_index(drop=True)

df_year_estacion_mostres = "20" + df["CAMPAÑA"].astype(str)+ "_" + df["ID_ESTACION"].astype(str)

df_year_estacion = read_tables(file2)


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

norm_x = transform(x, ["SUPERFICIE", "ALTITUD_MIN", "ALTITUD_DIF"])


x_train = norm_x
y_train = y
dataset_train = MyDataset(x_train.values, y_train.values, df_year_estacion, df_year_estacion_mostres)
    
    
## Dataloader creation
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers = 1)

#%%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.cuda.empty_cache()
model = Net(device)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
# Applying it to our net
model.apply(initialize_weights)

for name, param in model.named_parameters():
    if str(name) == "last_layer.bias":
        param.data =torch.Tensor([df["PRODUCCION"].mean()])

## Hyperparameters definition
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 50

#optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.33)
#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(validation_loader))
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

loss_history_train = []
loss_history_validation = []

# set device

model = model.to(device)
loss_function= nn.MSELoss() 

file_to_write = f"/home/caos/datathon/execution_losses/loss{sys.argv[1]}.txt"
image_to_write = f"/home/caos/datathon/execution_plots/plot{sys.argv[1]}.png"
for epoch in range(epochs):
    loss_train = train(model, device, dataloader_train, optimizer, epoch, loss_function)
                
    if scheduler:
        scheduler.step()
    print('Epoch: {} \tTime: {} \tTrainLoss: {:.6f}'.format(
        epoch, 
        time.strftime('%H:%M:%S'),
        loss_train
        ))
    f = open(file_to_write, "a")
    f.write('Epoch: {} \tTime: {} \tTrainLoss: {:.6f}\n'.format(
        epoch, 
        time.strftime('%H:%M:%S'),
        loss_train
        ))
    f.close()

    loss_history_train.append(loss_train)
    plt.plot(range(len(loss_history_train)), loss_history_train)
    plt.plot(range(len(loss_history_validation)), loss_history_validation)
    plt.savefig(image_to_write)

    PATH = r"/home/caos/datathon/MODEL_train_meteo_eto/models/"
    torch.save(model.state_dict(), PATH + "model7withDO.pth")