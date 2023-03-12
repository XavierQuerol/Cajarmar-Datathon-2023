# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 22:03:35 2023

@author: xavid
"""

import torch
from torch.utils.data import Dataset

import pandas as pd

class MyDataset(Dataset):
 
  def __init__(self, y, y_xgboost, df_year_estacion, df_year_estacion_mostres):

    self.y_train=torch.tensor(y,dtype=torch.float32)
    self.y_xgboost=torch.tensor(y_xgboost,dtype=torch.float32)
    self.df_year_estacion = df_year_estacion
    self.df_year_estacion_mostres = df_year_estacion_mostres
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
      a = self.df_year_estacion[self.df_year_estacion_mostres[idx]]
      b = self.y_xgboost[idx]
      c = self.y_train[idx]
      
      return a,b,c