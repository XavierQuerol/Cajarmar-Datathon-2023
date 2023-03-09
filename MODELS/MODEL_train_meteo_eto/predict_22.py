# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 07:27:15 2023

@author: xavid
"""

from model_prova7 import Net
from read_tables2 import read_tables
from dataset import MyDataset

import torch
import pandas as pd
import category_encoders as ce
from torch.utils.data import DataLoader, random_split

## READ MODEL
PATH = "trained_models/"
model = Net()
model.load_state_dict(torch.load(PATH + "model7withDO_final_train_PLTscheduler_normal.pth", map_location=torch.device('cpu')))

## READ FILES

predict_22 = pd.read_csv("../ds_tractats/final_22.csv")

file = "../ds_tractats/df_meteo_eto_57features.csv"
df_year_estacion = read_tables(file)
df_year_estacion_mostres = "20" + predict_22["CAMPAÑA"].astype(str)+ "_" + predict_22["ID_ESTACION"].astype(str)

##

predict_22 = predict_22.drop(columns="CAMPAÑA")

##

df = pd.read_csv("../ds_tractats/df_train_tractat.csv")
df = df.astype({"CAMPAÑA": str, "ID_FINCA": str, "ID_ZONA": str, "ID_ESTACION": str, "ALTITUD": str, "VARIEDAD": str, "MODO": int, "TIPO": int, "COLOR": int})

df = df.drop(df[(df["CAMPAÑA"]=="14")|(df["CAMPAÑA"]=="15")].index).reset_index(drop=True)

df = df.drop(columns="CAMPAÑA")
## Encoding
encoder = ce.OneHotEncoder(cols=["ID_FINCA", "ID_ZONA", "ID_ESTACION", "VARIEDAD", "ALTITUD"])
encoder.fit(df)

df_encoded = encoder.transform(predict_22)

#X,Y
x=df_encoded.drop(axis = 1, columns = ["PRODUCCION"])
y=df_encoded.loc[:,["PRODUCCION"]]

## Normalization
def transform(dataset, columns):
    for c in columns:
        dataset[c] = (dataset[c] - dataset[c].mean()) / dataset[c].std()
    return dataset

norm_x = transform(x, ["SUPERFICIE", "ALTITUD_MIN", "ALTITUD_DIF"])

dataset = MyDataset(norm_x.values, y.values, df_year_estacion, df_year_estacion_mostres)
dataloader = DataLoader(dataset, batch_size=1075)

for batch_idx, (meteo_eto,ds_train,y) in enumerate(dataloader):
    with torch.no_grad():
        model.eval()
        data1 = meteo_eto
        data2 = ds_train
    
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