# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 22:03:35 2023

@author: xavid
"""

import torch
from torch.utils.data import Dataset

import pandas as pd

class MyDataset(Dataset):
 
  def __init__(self, x, y, df_year_estacion, df_year_estacion_mostres):

    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)
    self.df_year_estacion = df_year_estacion
    self.df_year_estacion_mostres = df_year_estacion_mostres
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
      a = self.df_year_estacion[self.df_year_estacion_mostres[idx]]
      b = self.x_train[idx]
      c = self.y_train[idx]
      return a,b,c