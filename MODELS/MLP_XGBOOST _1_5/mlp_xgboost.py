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
from utils_code import read_tables, pca_func
from sklearn.ensemble import GradientBoostingRegressor

import lightgbm as lgb
import catboost as cb

from model import Net
from dataset import MyDataset
from train import train
from testing import test

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


## READ DATASETS

df_meteo_eto = pd.read_csv("../../DATASETS_TRACTATS/df_meteo_eto.csv")
df_train = pd.read_csv("../../DATASETS_TRACTATS/TOT_menys14_15.csv")

df_train = df_train.astype({"CAMPAÑA": str, "ID_FINCA": str, "ID_ZONA": str, "ID_ESTACION": str, "VARIEDAD": str, "MODO": str, "TIPO": str, "COLOR": str})

encoder = ce.OrdinalEncoder(cols=["ID_FINCA", "ID_ZONA", "ID_ESTACION", "ALTITUD", "VARIEDAD", "COLOR", "TIPO", "MODO"])
df_train = encoder.fit_transform(df_train)

df_train = df_train[df_train["CAMPAÑA"] != "22"].drop(columns = ["SUPERFICIE", "P/S"])

df_pca = df_meteo_eto.drop(columns = ["validTimeUtc", "ID_ESTACION"])
pca = PCA(n_components=0.99)
pdf_fitted = pca.fit(df_pca.values)



fitxer = "../../DATASETS_TRACTATS/df_meteo_eto.csv"
df_year_estacion = read_tables(fitxer)
taula_pca = pca_func(df_year_estacion, pdf_fitted)


df_year_estacion_mostres = "20" + df_train["CAMPAÑA"].astype(str)+ "_" + df_train["ID_ESTACION"].astype(str)


X = df_train.drop(columns = ["PRODUCCION"])
X["ID_ESTACION2"] = X["ID_ESTACION"]
y = df_train["PRODUCCION"]

encoder = ce.OneHotEncoder(cols=["ID_FINCA", "ID_ZONA", "ID_ESTACION", "VARIEDAD", "ALTITUD"])
X_encoded = encoder.fit_transform(X)

## Normalization
def transform(dataset, columns):
    for c in columns:
        dataset[c] = (dataset[c] - dataset[c].mean()) / dataset[c].std()
    return dataset

norm_x = transform(X_encoded, ["ALTITUD_MIN", "ALTITUD_DIF"]).values
X_arr = X.values

X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(norm_x, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

X_train, X_test, y_train, y_test = train_test_split(X_arr, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


df_X_train_encoded = pd.DataFrame(X_train_encoded, columns = X_encoded.columns)
df_X_test_encoded = pd.DataFrame(X_test_encoded, columns = X_encoded.columns)
df_X_train = pd.DataFrame(X_train, columns = X.columns)
df_X_test = pd.DataFrame(X_test, columns = X.columns)


df_year_estacion_mostres_train = "20" + df_X_train["CAMPAÑA"].astype(str)+ "_" + df_X_train["ID_ESTACION2"].astype(str)
df_year_estacion_mostres_test = "20" + df_X_test["CAMPAÑA"].astype(str)+ "_" + df_X_test["ID_ESTACION2"].astype(str)

df_X_train = df_X_train.drop(columns = ["CAMPAÑA", "ID_ESTACION2"])
df_X_test = df_X_test.drop(columns = ["CAMPAÑA", "ID_ESTACION2"])
df_X_train_encoded = df_X_train_encoded.drop(columns = ["CAMPAÑA", "ID_ESTACION2"])
df_X_test_encoded = df_X_test_encoded.drop(columns = ["CAMPAÑA", "ID_ESTACION2"])

## XGBOOST

model = GradientBoostingRegressor(
    n_estimators= 100, 
    learning_rate = 0.15, 
    max_depth = 40, 
    min_samples_split = 5,
    min_impurity_decrease = 0.2,
    min_samples_leaf = 5,
    )

"""
model = lgb.LGBMRegressor(
        n_estimators = 100,
        learning_rate = 0.15,
        max_depth = 40,
        min_child_weight = 0.2,
        min_split_gain  = 0.5,
        colsample_bytree = 0.4,   
    )
"""

"""model = cb.CatBoostRegressor(
    loss_function='RMSE',
    iterations = 200,
    learning_rate = 0.1,
    depth = 6,
    )"""



model.fit(df_X_train.values, y_train)

y_train_xgboost = model.predict(df_X_train.values)

y_test_xgboost = model.predict(df_X_test.values)

## MLP

# df_pca --> diccionari
# df_year_estacion_mostres --> per cada mostra a quina fila ha d'accedir

## Dataset creation
dataset_train = MyDataset(df_X_train_encoded.values, y_train.values, y_train_xgboost, taula_pca, df_year_estacion_mostres_train.values)
dataset_validation = MyDataset(df_X_test_encoded.values, y_test.values, y_test_xgboost, taula_pca, df_year_estacion_mostres_test.values)

## Dataloader creation
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
dataloader_validation = DataLoader(dataset_validation, batch_size=len(y_test_xgboost), shuffle=False)



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
        param.data = torch.Tensor([y.mean()])

## Hyperparameters definition
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-4)
epochs = 50

#optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.33)
#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(dataloader_validation))
#♣scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

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
    
    
    if epoch != 0:
        loss_history_train.append(loss_train)
        loss_history_validation.append(loss_validation)
        plt.plot(range(len(loss_history_train)), loss_history_train)
        plt.plot(range(len(loss_history_validation)), loss_history_validation)
        plt.show()

### Test model on 20-21
#y = df_train.loc[(df_train["CAMPAÑA"].isin(["20", "21"])), :]["PRODUCCION"]
#x = 