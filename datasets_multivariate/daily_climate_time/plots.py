# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 11:38:41 2023

@author: xavid
"""

import pandas as pd
from darts import TimeSeries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Read a pandas DataFrame
df = pd.read_csv("DailyDelhiClimateTrain.csv", delimiter=",")


# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(df, "date", ["meantemp", "humidity", "wind_speed", "meanpressure"])

#%%
#sns.lineplot(df, x = "DATE", y = "IPG2211A2N")
fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Mean temperature", "Humidity"),
            shared_xaxes=True,
            vertical_spacing =0.3)
fig.add_trace(go.Scatter(x = df["date"], y = df["meantemp"], mode="lines"), row=1, col=1)
fig.add_trace(go.Scatter(x = df["date"], y = df["humidity"], mode="lines"), row=2, col=1)
fig.show()

#%%
fig = px.scatter(df, x = "meantemp", y="humidity")
fig.show()

#%%
fig = px.scatter_matrix(df)
fig.show()

#%%
#not okay
pd.plotting.lag_plot(df["humidity"], lag=1)

#%%
pd.plotting.autocorrelation_plot(df["meantemp"])