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
import pickle
from sklearn.preprocessing import StandardScaler

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


# True si estem fent l'entrenament final amb totes les dades.
# False si encara estem fent train/validation
final_train = True

torch.manual_seed(0)

## DATASET METEO-ETO

df_meteo_eto = pd.read_csv("../../DATASETS_TRACTATS/df_meteo_eto.csv")
df_pca = df_meteo_eto.drop(columns = ["validTimeUtc", "ID_ESTACION"])
pca = PCA(n_components=0.99)
pdf_fitted = pca.fit(df_pca.values)
fitxer = "../../DATASETS_TRACTATS/df_meteo_eto.csv"
df_year_estacion = read_tables(fitxer)
taula_pca = pca_func(df_year_estacion, pdf_fitted)


# DATASET TRAIN 

df_train = pd.read_csv("../../DATASETS_FINALS/TOT_20_21.csv")

df_train = df_train.astype({"CAMPAÑA": str, "ID_FINCA": str, "ID_ZONA": str, "ID_ESTACION": str, "VARIEDAD": str, "MODO": str, "TIPO": str, "COLOR": str})

df_train = df_train.drop(columns = ["ID_FINCA", "ID_ZONA"])

# ORDINAL ENCODING
encoder = ce.OrdinalEncoder(cols=["ID_ESTACION", "ALTITUD", "VARIEDAD", "COLOR", "TIPO", "MODO"])
df_OrdEnc = encoder.fit_transform(df_train)
filehandler = open("./trained_models/ordinal_encoder_cas3.obj","wb")
pickle.dump(encoder, filehandler)

X_OrdEnc = df_OrdEnc.drop(columns = ["PRODUCCION", "CAMPAÑA"])

y = df_train["PRODUCCION"]


# ONE HOT ENCODING
encoder = ce.OneHotEncoder(cols=["ID_ESTACION", "VARIEDAD", "ALTITUD"])
df_OHEnc = encoder.fit_transform(df_train)
filehandler = open("./trained_models/one_hot_encoder_cas3.obj","wb")
pickle.dump(encoder, filehandler)

X_OHEnc = df_OHEnc.drop(columns = ["PRODUCCION", "CAMPAÑA"])

## Normalization

scaler = StandardScaler()
X_OHEnc[["ALTITUD_MIN", "ALTITUD_DIF", "SUPERFICIE"]] = scaler.fit_transform(X_OHEnc[["ALTITUD_MIN", "ALTITUD_DIF", "SUPERFICIE"]])
filehandler = open("./trained_models/scaler_cas3.obj","wb")
pickle.dump(scaler, filehandler)

if final_train:
    
    df_year_estacion_mostres_train = df_train["CAMPAÑA_ESTACION"]
    X_OrdEnc = X_OrdEnc.drop(columns = ["CAMPAÑA_ESTACION"])
    X_OHEnc = X_OHEnc.drop(columns = ["CAMPAÑA_ESTACION"])
    
else:
    
    X_train_OrdEnc, X_test_OrdEnc, y_train_OrdEnc, y_test_OrdEnc = train_test_split(X_OrdEnc, 
                                                                   y, 
                                                                   test_size = 0.2, 
                                                                   random_state = 0)
    
    X_train_OHEnc, X_test_OHEnc, y_train_OHEnc, y_test_OHEnc = train_test_split(X_OHEnc.values, 
                                                        y, 
                                                        test_size = 0.2, 
                                                        random_state = 0)
    
    
    df_X_train_OrdEnc = pd.DataFrame(X_train_OrdEnc, columns = X_OrdEnc.columns)
    df_X_test_OrdEnc = pd.DataFrame(X_test_OrdEnc, columns = X_OrdEnc.columns)
    df_X_train_OHEnc = pd.DataFrame(X_train_OHEnc, columns = X_OHEnc.columns)
    df_X_test_OHEnc = pd.DataFrame(X_test_OHEnc, columns = X_OHEnc.columns)
    
    df_year_estacion_mostres_train = df_X_train_OrdEnc["CAMPAÑA_ESTACION"]
    df_year_estacion_mostres_test = df_X_train_OrdEnc["CAMPAÑA_ESTACION"]
    
    df_X_train_OrdEnc = df_X_train_OrdEnc.drop(columns = ["CAMPAÑA_ESTACION"])
    df_X_test_OrdEnc = df_X_test_OrdEnc.drop(columns = ["CAMPAÑA_ESTACION"])
    df_X_train_OHEnc = df_X_train_OHEnc.drop(columns = ["CAMPAÑA_ESTACION"])
    df_X_test_OHEnc = df_X_test_OHEnc.drop(columns = ["CAMPAÑA_ESTACION"])

## XGBOOST

model = GradientBoostingRegressor(
    n_estimators= 100, 
    learning_rate = 0.15, 
    max_depth = 40, 
    min_samples_split = 5,
    min_impurity_decrease = 0.05,
    min_samples_leaf = 2,
    )



if final_train:
    
    model.fit(X_OrdEnc, y)
    y_predicted_xgboost_for_train = model.predict(X_OrdEnc)
    
    filehandler = open("./trained_models/gradientboosting_cas3.obj","wb")
    pickle.dump(model, filehandler)
    
else:
    
    model.fit(df_X_train_OrdEnc.values, y_train_OrdEnc)
    
    y_predicted_xgboost_for_train = model.predict(df_X_train_OrdEnc.values)
    
    y_predicted_xgboost_for_test = model.predict(df_X_test_OrdEnc.values)

## MLP

# df_pca --> diccionari
# df_year_estacion_mostres --> per cada mostra a quina fila ha d'accedir

## Dataset creation
if final_train:
    
    dataset_train = MyDataset(X_OHEnc.values, y.values, 
                              y_predicted_xgboost_for_train, taula_pca, 
                              df_year_estacion_mostres_train.values)
    
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

else:
    
    dataset_train = MyDataset(df_X_train_OHEnc.values, y_train_OHEnc.values, 
                              y_predicted_xgboost_for_train, taula_pca, 
                              df_year_estacion_mostres_train.values)
    
    dataset_validation = MyDataset(df_X_test_OHEnc.values, y_test_OHEnc.values, 
                                   y_predicted_xgboost_for_test, taula_pca, 
                                   df_year_estacion_mostres_test.values)
    
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_validation = DataLoader(dataset_validation, batch_size=len(y_predicted_xgboost_for_train), shuffle=False)



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
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-5)
epochs = 20

#optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.33)
#scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(dataloader_validation))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 4)

loss_history_train = []
loss_history_validation = []

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
loss_function= nn.MSELoss() 


for epoch in range(epochs):
    loss_train = train(model, device, dataloader_train, optimizer, epoch, loss_function)
    if final_train:
        print('Epoch: {} \tTrainLoss: {:.6f}'.format(
            epoch, 
            loss_train
            ))
        PATH = r"trained_models/"
        torch.save(model.state_dict(), PATH + "model_cas3.pth")
    else:
        loss_validation = test(model, device, dataloader_validation, loss_function)
                    
        
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
            
    if scheduler:
        scheduler.step(loss_train)
        

### Test model on 20-21
#y = df_train.loc[(df_train["CAMPAÑA"].isin(["20", "21"])), :]["PRODUCCION"]
#x = 