# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:47:48 2023

@author: xavid
"""

import pandas as pd
from darts import TimeSeries

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# Read a pandas DataFrame
df = pd.read_csv("DailyDelhiClimateTrain.csv", delimiter=",")


# Create a TimeSeries, specifying the time and value columns
#series = TimeSeries.from_dataframe(df, "date", ["meantemp", "humidity", "wind_speed", "meanpressure"])

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
#df["theta"] = 2 * 3.1416 * (df["month"]-1) / 12
df["theta"] = 2 * 3.1416 * (df["day"]-1+df["month"]*30) / 365
df["x"] = df["meantemp"] * np.sin(df["theta"])
df["y"] = df["meantemp"] * np.cos(df["theta"])
df["z"] = df.year

import matplotlib.pyplot as plt

#plt.plot(df["x"], df["y"])

import plotly.express as px
import plotly.io as io
io.renderers.default='browser'

fig = go.Figure()

for i, year in enumerate(df["year"].unique()):
    df2 = df[df["year"]==year]
    fig.add_trace(go.Scatter3d(x = df2["x"], y = df2["y"], z = df2["z"], mode="lines", line={"width":5}))

#fig = px.line_3d(df, x="x", y="y", z="z")
fig.show()