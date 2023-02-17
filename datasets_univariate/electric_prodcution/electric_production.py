# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 20:25:36 2023

@author: xavid
"""

import pandas as pd
from darts import TimeSeries
from darts.models import Prophet, TCNModel, RNNModel, TransformerModel
from darts.metrics import mape
import matplotlib.pyplot as plt
from darts.dataprocessing.transformers import Scaler

# Read a pandas DataFrame
df = pd.read_csv("Electric_Production.csv", delimiter=",")


# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(df, "DATE", "IPG2211A2N")
#series.plot()
# Set aside the last 36 months as a validation series
train, val = series.split_after(0.8)

scaler = Scaler()
train_transformed = scaler.fit_transform(train)
val_transformed = scaler.transform(val)
series_transformed = scaler.transform(series)


models = []
models.append(Prophet())
#models.append(TCNModel(input_chunk_length=14, output_chunk_length=2, kernel_size = 5, n_epochs=20))
#models.append(RNNModel(input_chunk_length=14))
#models.append(TransformerModel(input_chunk_length=14, output_chunk_length=2))

backtests = [model.historical_forecasts(series,
                            start=.7,
                            forecast_horizon=3,
                            retrain = False)
             for model in models]

model = TCNModel(input_chunk_length=14, output_chunk_length=2, 
             kernel_size = 5, n_epochs=20)
backtest = model.historical_forecasts(series,
                            start=.7,
                            forecast_horizon=3,
                            retrain = False)



series.plot(label='data')
for i, m in enumerate(models):
    err = mape(backtests[i], series)
    backtests[i].plot(lw=3, label='{}, MAPE={:.2f}%'.format(m, err))

plt.title('Backtests with 3-months forecast horizon')
plt.legend()