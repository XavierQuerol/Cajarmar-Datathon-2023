# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:43:51 2023

@author: xavid
"""

import pandas as pd
from darts import TimeSeries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Read a pandas DataFrame
df = pd.read_csv("Electric_Production.csv", delimiter=",")


# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(df, "DATE", "IPG2211A2N")
df["DATE"] = pd.to_datetime(df["DATE"])
df["month"] = df["DATE"].dt.month
df["year"] = df["DATE"].dt.year

import plotly.io as io
io.renderers.default='svg'
#%%
sns.lineplot(df, x = "DATE", y = "IPG2211A2N")
#fig =px.line(df, x = "DATE", y = "IPG2211A2N")
#ig.show()
#%%
sns.lineplot(df, x = "month", y = "IPG2211A2N", hue = "year")
#fig =px.line(df, x = "month", y = "IPG2211A2N", color = "year")
#fig.show()

#%%
sns.lineplot(df, x = "year", y = "IPG2211A2N", hue = "month")
#fig =px.line(df, x = "year", y = "IPG2211A2N", color = "month")
#fig.show()
#%%
sns.lineplot(df, x = "year", y = "IPG2211A2N", hue = "month")
#fig =px.line(df, x = "year", y = "IPG2211A2N", color = "month")
#fig.show()
#%%
df["month2"] = df["month"].astype(str)
fig = px.line_polar(df, r='IPG2211A2N', theta='month2', 
                    color='year', line_close=False,
                    title='Polar seasonall plot',
                    width=600, height=500)

fig.show()

#%%
from statsmodels.graphics.tsaplots import month_plot
df2 = df.set_index("DATE")
month_plot(df2["IPG2211A2N"], ylabel='IPG2211A2N');

#%%
