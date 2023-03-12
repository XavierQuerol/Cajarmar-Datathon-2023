# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 22:03:35 2023

@author: xavid
"""

import torch
from torch.utils.data import Dataset

import pandas as pd

class MyDataset(Dataset):
 
  def __init__(self, x_train, y, y_xgboost, df_year_estacion, df_year_estacion_mostres):

    self.y_train=torch.tensor(y,dtype=torch.float32)
    self.y_xgboost=torch.tensor(y_xgboost,dtype=torch.float32)
    self.df_year_estacion = df_year_estacion
    self.df_year_estacion_mostres = df_year_estacion_mostres
    self.x_train = x_train
    
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
      #print(type(self.x_train[idx]))
      #print(type(self.df_year_estacion[self.df_year_estacion_mostres[idx]].flatten()))
      #print(list(self.x_train[idx]))
      #print(type(torch.from_numpy(self.x_train[idx])))
      a = torch.cat((torch.from_numpy(self.x_train[idx].astype('float32')), self.df_year_estacion[self.df_year_estacion_mostres[idx]].flatten()))
      b = self.y_xgboost[idx]
      c = self.y_train[idx]
      
      return a,b,c
  
class MyDataset(Dataset):
 
  def __init__(self, x_train, y, y_xgboost, df_year_estacion, df_year_estacion_mostres):

    self.y_train=torch.tensor(y,dtype=torch.float32)
    self.y_xgboost=torch.tensor(y_xgboost,dtype=torch.float32)
    self.df_year_estacion = df_year_estacion
    self.df_year_estacion_mostres = df_year_estacion_mostres
    self.x_train = x_train
    
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
      #print(type(self.x_train[idx]))
      #print(type(self.df_year_estacion[self.df_year_estacion_mostres[idx]].flatten()))
      #print(list(self.x_train[idx]))
      #print(type(torch.from_numpy(self.x_train[idx])))
      a = torch.cat((torch.from_numpy(self.x_train[idx].astype('float32')), self.df_year_estacion[self.df_year_estacion_mostres[idx]].flatten()))
      b = self.y_xgboost[idx]
      c = self.y_train[idx]
      
      return a,b,c