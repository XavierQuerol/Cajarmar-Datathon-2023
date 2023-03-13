# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.decomposition import PCA
import torch
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from utils import read_tables, pca_func

from model import Net, Net2
from dataset import MyDataset, MyDataset2
from train import train, train2
from testing import test, test2

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


torch.manual_seed(0)
## READ DATASETS

df_meteo_eto = pd.read_csv("../../DATASETS_TRACTATS/df_meteo_eto.csv")
df_train = pd.read_csv("../../DATASETS_TRACTATS/TOT_menys14_15.csv")

df_train = df_train.astype({"CAMPAÑA": str, "ID_FINCA": str, "ID_ZONA": str, "ID_ESTACION": str, "VARIEDAD": str, "MODO": str, "TIPO": str, "COLOR": str})

encoder = ce.OrdinalEncoder(cols=["ID_FINCA", "ID_ZONA", "ID_ESTACION", "ALTITUD", "VARIEDAD", "COLOR", "TIPO", "MODO"])
df_train = encoder.fit_transform(df_train)

df_train = df_train[df_train["CAMPAÑA"] != "22"].drop(columns = ["P/S"])
df_train = df_train[(df_train["CAMPAÑA"].isin(["20","21"]) == False) | ((df_train["CAMPAÑA"].isin(["20","21"]) == True) & (df_train["SUPERFICIE"] != 0))]

df_pca = df_meteo_eto.drop(columns = ["validTimeUtc", "ID_ESTACION"])
pca = PCA(n_components=0.99)
pdf_fitted = pca.fit(df_pca.values)



fitxer = "../../DATASETS_TRACTATS/df_meteo_eto.csv"
df_year_estacion = read_tables(fitxer)
taula_pca = pca_func(df_year_estacion, pdf_fitted)


X = df_train.drop(columns = ["PRODUCCION"])

X["ID_ESTACION2"] = X["ID_ESTACION"]
y = df_train["PRODUCCION"]

encoder = ce.OneHotEncoder(cols=["ID_FINCA", "ID_ZONA", "ID_ESTACION", "VARIEDAD", "ALTITUD",
                                 "MODO", "TIPO", "COLOR"])
X_encoded = encoder.fit_transform(X)

## Normalization
def transform(dataset, columns):
    for c in columns:
        dataset[c] = (dataset[c] - dataset[c].mean()) / dataset[c].std()
    return dataset

norm_x = transform(X, ["ALTITUD_MIN", "ALTITUD_DIF"])
norm_x_encoded = transform(X_encoded, ["ALTITUD_MIN", "ALTITUD_DIF"])


X_16_19 = norm_x[df_train["CAMPAÑA"].isin(["20", "21"]) == False]
X_20_21 = norm_x[df_train["CAMPAÑA"].isin(["20", "21"]) == True]

y_16_19 = y[df_train["CAMPAÑA"].isin(["20", "21"]) == False]
y_20_21 = y[df_train["CAMPAÑA"].isin(["20", "21"]) == True]

X_16_19_encoded = norm_x_encoded[df_train["CAMPAÑA"].isin(["20", "21"]) == False]
X_20_21_encoded = norm_x_encoded[df_train["CAMPAÑA"].isin(["20", "21"]) == True]
    
X_train_20_21, X_test_20_21, y_train_20_21, y_test_20_21 = train_test_split(X_20_21, 
                                                    y_20_21, 
                                                    test_size = 0.3, 
                                                    random_state = 5)

X_train_20_21_encoded, X_test_20_21_encoded, y_train_20_21_encoded, y_test_20_21_encoded = train_test_split(X_20_21_encoded, 
                                                    y_20_21, 
                                                    test_size = 0.3, 
                                                    random_state = 5)


df_X_train_20_21 = pd.DataFrame(X_train_20_21, columns = X.columns)
df_X_test_20_21 = pd.DataFrame(X_test_20_21, columns = X.columns)

df_X_train_20_21_encoded = pd.DataFrame(X_train_20_21_encoded, columns = X_encoded.columns)
df_X_test_20_21_encoded = pd.DataFrame(X_test_20_21_encoded, columns = X_encoded.columns)

#XGBOOST
df_X_train_part1 = pd.concat([df_X_train_20_21, X_16_19])

#MLP
df_X_train_part2 = pd.concat([df_X_train_20_21_encoded, X_16_19_encoded])

#BOTH XGBOOST AND MLP
df_y_train_part12 = pd.concat([y_train_20_21, y_16_19])

#MLP 1
df_year_estacion_mostres_train = "20" + df_X_train_part2["CAMPAÑA"].astype(str)+ "_" + df_X_train_part2["ID_ESTACION2"].astype(str)
df_year_estacion_mostres_test = "20" + df_X_test_20_21["CAMPAÑA"].astype(str)+ "_" + df_X_test_20_21["ID_ESTACION2"].astype(str)

#MLP2
df_year_estacion_mostres_train2 = "20" + df_X_train_20_21_encoded["CAMPAÑA"].astype(str)+ "_" + df_X_train_20_21_encoded["ID_ESTACION2"].astype(str)

df_superficie_train_part3 =  df_X_train_20_21["SUPERFICIE"]
df_superficie_test_part3= df_X_test_20_21["SUPERFICIE"]

df_X_train_part1 = df_X_train_part1.drop(columns = ["CAMPAÑA", "ID_ESTACION2", "SUPERFICIE"])
df_X_train_part2 = df_X_train_part2.drop(columns = ["CAMPAÑA", "ID_ESTACION2", "SUPERFICIE"])
df_X_test_20_21 = df_X_test_20_21.drop(columns = ["CAMPAÑA", "ID_ESTACION2", "SUPERFICIE"])
df_X_test_20_21_encoded = df_X_test_20_21_encoded.drop(columns = ["CAMPAÑA", "ID_ESTACION2", "SUPERFICIE"])

df_X_train_part3 = df_X_train_20_21_encoded.drop(columns = ["CAMPAÑA", "ID_ESTACION2", "SUPERFICIE"])

## XGBOOST
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(
    n_estimators= 100, 
    learning_rate = 0.15, 
    max_depth = 40, 
    min_samples_split = 5,
    min_impurity_decrease = 0.2,
    min_samples_leaf = 5,
    )



model.fit(df_X_train_part1.values, df_y_train_part12.values)

print(model.score(df_X_train_part1.values, df_y_train_part12))
y_train_xgboost = model.predict(df_X_train_part1.values)

print(model.score(df_X_test_20_21.values, y_test_20_21_encoded))
y_test_xgboost = model.predict(df_X_test_20_21.values)

## MLP

# df_pca --> diccionari
# df_year_estacion_mostres --> per cada mostra a quina fila ha d'accedir

## Dataset creation
dataset_train = MyDataset(df_X_train_part2.values, df_y_train_part12.values, y_train_xgboost, taula_pca, df_year_estacion_mostres_train.values)
dataset_validation = MyDataset(df_X_test_20_21_encoded.values, y_test_20_21_encoded.values, y_test_xgboost, taula_pca, df_year_estacion_mostres_test.values)

## Dataloader creation
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=len(y_test_xgboost), shuffle=False)

#%%

## MLP MODEL CREATION
model = Net()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
# Applying it to our net
model.apply(initialize_weights)

for name, param in model.named_parameters():
    if str(name) == "common_layer.bias":
        param.data =torch.Tensor([y.mean()])

## Hyperparameters definition
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
epochs = 20

#optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.33)
#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(validation_loader))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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
        scheduler.step(loss_train)
    print('Epoch: {} \tTrainLoss: {:.6f}\tValidationLoss: {:.6f}'.format(
        epoch, 
        loss_train,
        loss_validation
        ))
    
    
    if epoch != 0:
        loss_history_train.append(loss_train)
        loss_history_validation.append(loss_validation)
        plt.plot(range(len(loss_history_train)), loss_history_train)
        plt.plot(range(len(loss_history_validation)), loss_history_validation)
        plt.show()
        

#%% DESAR WEIGHTS MODEL

PATH = r"./TRAINED_MODELS/"
torch.save(model.state_dict(), PATH + "model1.pth")

#%%

## MLP2 MODEL CREATION

## Dataset creation
dataset_train = MyDataset2(df_X_train_part3.values, y_train_20_21_encoded.values, y_train_xgboost, taula_pca, df_year_estacion_mostres_train2.values, df_superficie_train_part3.values)
dataset_validation = MyDataset2(df_X_test_20_21_encoded.values, y_test_20_21_encoded.values, y_test_xgboost, taula_pca, df_year_estacion_mostres_test.values, df_superficie_test_part3.values)

## Dataloader creation
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=len(y_test_xgboost), shuffle=False)

for param in model.parameters():
    param.requires_grad = False
    
model2 = Net2()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
# Applying it to our net
model2.apply(initialize_weights)

for name, param in model2.named_parameters():
    if str(name) == "common_layer.bias":
        param.data =torch.Tensor([y.mean()])

## Hyperparameters definition
lr = 1e-3
optimizer = optim.Adam(model2.parameters(), lr=lr)
epochs = 50

#optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.33)
#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(validation_loader))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

loss_history_train = []
loss_history_validation = []

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2 = model2.to(device)
loss_function= nn.MSELoss() 


for epoch in range(epochs):
    loss_train = train2(model, model2, device, dataloader_train, optimizer, epoch, loss_function)
    loss_validation = test2(model, model2, device, dataloader_validation, loss_function)
                
    if scheduler:
        scheduler.step(loss_train)
    print('Epoch: {} \tTrainLoss: {:.6f}\tValidationLoss: {:.6f}'.format(
        epoch, 
        loss_train,
        loss_validation
        ))
    
    
    if epoch != 0:
        loss_history_train.append(loss_train)
        loss_history_validation.append(loss_validation)
        plt.plot(range(len(loss_history_train)), loss_history_train)
        plt.plot(range(len(loss_history_validation)), loss_history_validation)
        plt.show()