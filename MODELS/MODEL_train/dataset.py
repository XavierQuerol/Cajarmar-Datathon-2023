# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 01:22:43 2023

@author: xavid
"""

import torch
from torch.utils.data import Dataset

import pandas as pd

class MyDataset(Dataset):
 
  def __init__(self, x, y):

    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]