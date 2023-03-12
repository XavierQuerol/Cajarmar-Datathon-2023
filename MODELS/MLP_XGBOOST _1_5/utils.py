# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 23:06:35 2023

@author: xavid
"""
import pandas as pd
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler


def read_tables(file):
    df_meteo_eto = pd.read_csv(file, index_col = False)
    df_meteo_eto.validTimeUtc = pd.to_datetime(df_meteo_eto.validTimeUtc)
    df_meteo_eto.loc[:, "validTimeUtc"] = df_meteo_eto.loc[:, "validTimeUtc"] + pd.Timedelta(days=185)
    years = df_meteo_eto["validTimeUtc"].dt.year.unique()
    estacions = df_meteo_eto["ID_ESTACION"].unique()
    columns_to_store = df_meteo_eto.columns
    columns_to_store = columns_to_store.drop(["ID_ESTACION", "validTimeUtc"])
    
    df_meteo_eto[columns_to_store] = StandardScaler().fit_transform(df_meteo_eto[columns_to_store])

    meteo_eto_dict = {}

    for year in years[:-1]:
        for estacion in estacions:
            df_aux = df_meteo_eto.loc[(df_meteo_eto["ID_ESTACION"] == estacion) & (df_meteo_eto["validTimeUtc"].dt.year == year),columns_to_store]
            if year == 2016 or year == 2020:
                df_aux2 = torch.tensor(df_aux.values, dtype=torch.float32)[1+240:,:]
            else:
                df_aux2 = torch.tensor(df_aux.values, dtype=torch.float32)[240:,:]
            meteo_eto_dict[f"{year}_{estacion}"] = df_aux2
            
    return meteo_eto_dict

def pca_func(taules, pca_fitted):
    taula_nova = dict()
    for k,v in taules.items():
        taula_nova[k] = torch.tensor(pca_fitted.transform(v), dtype=torch.float32)
    return taula_nova    

