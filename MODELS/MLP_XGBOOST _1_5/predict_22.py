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

## READ models
PATH = "trained_models/"
model = Net()
model.load_state_dict(torch.load(PATH + "model1.pth", map_location=torch.device('cpu')))

file = open("./trained_models/gradientboosting.obj", "rb")
model_gradient = pickle.load(file)

file = open("./trained_models/ordinal_encoder.obj", "rb")
ordinal_encoder = pickle.load(file)

file = open("./trained_models/one_hot_encoder.obj", "rb")
one_hot_encoder = pickle.load(file)


## DATASET METEO-ETO

df_meteo_eto = pd.read_csv("../../DATASETS_TRACTATS/df_meteo_eto.csv")
df_pca = df_meteo_eto.drop(columns = ["validTimeUtc", "ID_ESTACION"])
pca = PCA(n_components=0.99)
pdf_fitted = pca.fit(df_pca.values)
fitxer = "../../DATASETS_TRACTATS/df_meteo_eto.csv"
df_year_estacion = read_tables(fitxer)
taula_pca = pca_func(df_year_estacion, pdf_fitted)


## DATASET 22

df_22 = pd.read_csv("../../DATASETS_FINALS/df_22_cas1.csv")
df_22 = df_22.astype({"CAMPAÑA": str, "ID_FINCA": str, "ID_ZONA": str, "ID_ESTACION": str, "VARIEDAD": str, "MODO": str, "TIPO": str, "COLOR": str})

df_22_index = df_22["INDEX"]
df_22 = df_22.drop(columns=["INDEX", "SUPERFICIE"])
df_year_estacion_mostres_train = df_22["CAMPAÑA_ESTACION"]

# ENCODING

df_22_OrdEnc = ordinal_encoder.transform(df_22)
df_22_OHEnc = one_hot_encoder.transform(df_22)
y = df_22["PRODUCCION"]

df_22_OrdEnc = df_22_OrdEnc.drop(columns = ["PRODUCCION", "CAMPAÑA_ESTACION", "CAMPAÑA"])
df_22_OHEnc = df_22_OHEnc.drop(columns = ["PRODUCCION", "CAMPAÑA_ESTACION", "CAMPAÑA"])


## Normalization
def transform(dataset, columns):
    for c in columns:
        dataset[c] = (dataset[c] - dataset[c].mean()) / dataset[c].std()
    return dataset

norm_x = transform(df_22_OHEnc, ["ALTITUD_MIN", "ALTITUD_DIF"])


y_predicted_xgboost_for_train = model_gradient.predict(df_22_OrdEnc)

dataset = MyDataset(df_22_OHEnc.values, y.values, 
                          y_predicted_xgboost_for_train, taula_pca, 
                          df_year_estacion_mostres_train.values)

dataloader = DataLoader(dataset, batch_size=1075, shuffle=False)

for batch_idx, (X_meteo_eto, pred_xgboost, y) in enumerate(dataloader):
    with torch.no_grad():
        model.eval()
        
        data1 = X_meteo_eto
        data2 = pred_xgboost
    
        output = model(data1, data2)


new_22 = pd.DataFrame()
new_22["INDEX"] = df_22_index.values
new_22["PRODUCCION"] = output.numpy()

new_22.to_csv("../../PREDICTIONS/cas1.csv", index = False)
