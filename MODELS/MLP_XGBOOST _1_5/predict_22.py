# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 07:27:15 2023

@author: xavid
"""

from sklearn.decomposition import PCA
from utils_code import read_tables, pca_func
from dataset import MyDataset

from model import Net
from dataset import MyDataset

import torch
import pandas as pd
import category_encoders as ce
from torch.utils.data import DataLoader, random_split

import pickle

## READ MODEL
PATH = "trained_models/"
model = Net()
model.load_state_dict(torch.load(PATH + "model1.pth", map_location=torch.device('cpu')))

## READ FILES

df_meteo_eto = pd.read_csv("../../DATASETS_TRACTATS/df_meteo_eto.csv")
df_train = pd.read_csv("../../DATASETS_TRACTATS/TOT_menys14_15.csv")

df_train = df_train.astype({"CAMPAÑA": str, "ID_FINCA": str, "ID_ZONA": str, "ID_ESTACION": str, "VARIEDAD": str, "MODO": str, "TIPO": str, "COLOR": str})

encoder = ce.OrdinalEncoder(cols=["ID_FINCA", "ID_ZONA", "ID_ESTACION", "ALTITUD", "VARIEDAD", "COLOR", "TIPO", "MODO"])
df_train = encoder.fit_transform(df_train)



df_pca = df_meteo_eto.drop(columns = ["validTimeUtc", "ID_ESTACION"])
pca = PCA(n_components=0.99)
pdf_fitted = pca.fit(df_pca.values)
fitxer = "../../DATASETS_TRACTATS/df_meteo_eto.csv"
df_year_estacion = read_tables(fitxer)
taula_pca = pca_func(df_year_estacion, pdf_fitted)

df_22 = df_train[df_train["CAMPAÑA"] == "22"].drop(columns = ["SUPERFICIE", "P/S"])

df_year_estacion_mostres = "20" + df_22["CAMPAÑA"].astype(str)+ "_" + df_22["ID_ESTACION"].astype(str)


X = df_22.drop(columns = ["PRODUCCION", "CAMPAÑA"])
y = df_22["PRODUCCION"]
#X = X.reset_index(drop=True)
#y = y.reset_index(drop=True)

df_train2 = df_train.drop(columns = ["PRODUCCION", "CAMPAÑA", "SUPERFICIE", "P/S"])
encoder = ce.OneHotEncoder(cols=["ID_FINCA", "ID_ZONA", "ID_ESTACION", "VARIEDAD", "ALTITUD"])
encoder.fit(df_train2)
X_encoded = encoder.transform(X)

## Normalization
def transform(dataset, columns):
    for c in columns:
        dataset[c] = (dataset[c] - dataset[c].mean()) / dataset[c].std()
    return dataset

norm_x = transform(X_encoded, ["ALTITUD_MIN", "ALTITUD_DIF"])

file = open("./trained_models/gradientboosting.obj", "rb")
model_gradient = pickle.load(file)
y_train_xgboost = model_gradient.predict(X)

dataset = MyDataset(norm_x.values, y.values, y_train_xgboost, taula_pca, df_year_estacion_mostres.values)
dataloader = DataLoader(dataset, batch_size=1075)

for batch_idx, (X_meteo_eto, pred_xgboost, y) in enumerate(dataloader):
    with torch.no_grad():
        model.eval()
        
        data1 = X_meteo_eto
        data2 = pred_xgboost
    
        output = model(data1, data2)


np_array_o = output.numpy()

DATASET_GROS = pd.read_csv("../UH_2023/UH_2023_TRAIN.txt", sep = "|", dtype = {'CAMPAÑA': int,
                                                                              'ID_FINCA': int,
                                                                              'ID_ZONA': int,
                                                                              'ID_ESTACION': int,
                                                                              'ALTITUD': str,
                                                                              "VARIEDAD": str,
                                                                              "MODO": int,
                                                                              "TIPO": str, 
                                                                              "COLOR": int, 
                                                                              "SUPERFICIE": float, 
                                                                              "PRODUCCION":float
                                                                              })

DATASET_GROS = DATASET_GROS[DATASET_GROS["CAMPAÑA"]==22]

DATASET_GROS["PRODUCCION"] = np_array_o



DATASET_GROS = DATASET_GROS.drop(columns=["ALTITUD", "ID_ZONA", "ID_ESTACION"])

DATASET_GROS = DATASET_GROS.reindex(columns=["ID_FINCA", "VARIEDAD", "MODO", "TIPO", "COLOR", "SUPERFICIE", "PRODUCCION"])
DATASET_GROS = DATASET_GROS.sort_values(by=["ID_FINCA", "VARIEDAD", "MODO", "TIPO", "COLOR", "SUPERFICIE"])

DATASET_GROS["PRODUCCION"] = DATASET_GROS["PRODUCCION"].apply(lambda x: '{:.2f}'.format(x) if type(x) is int or type(x) is float else x)
DATASET_GROS["SUPERFICIE"] = DATASET_GROS["SUPERFICIE"].apply(lambda x: '{:.2f}'.format(x) if type(x) is int or type(x) is float else x)

def fun(x): 
    x=str(x)
    if len(x) == 0:
        return '"' + '000' +'"'
    elif len(x) == 1:
        return '"' + '00' + x +'"'
    elif len(x) == 2:
        return '"' + '0' + x +'"'
    elif len(x) == 3:
        return '"' + x +'"'

def fun2(x): 
    x='"' + str(x) +'"'
    return x

DATASET_GROS["VARIEDAD"] = DATASET_GROS["VARIEDAD"].apply(fun)
DATASET_GROS["TIPO"] = DATASET_GROS["TIPO"].apply(fun2)


name_file = "UH2023_Universitat Autònoma de Barcelona (UAB)_Farts del vi_1.txt"
DATASET_GROS.to_csv(name_file, sep = "|", header=False, index=False)