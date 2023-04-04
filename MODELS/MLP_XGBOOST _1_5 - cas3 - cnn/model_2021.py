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



df_train = pd.read_csv("../../DATASETS_TRACTATS/TOT_menys14_15.csv")

