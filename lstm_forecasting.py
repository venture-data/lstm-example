import json
import sys

import joblib
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from feature_engine.timeseries.forecasting import LagFeatures
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from statsmodels.tsa.seasonal import seasonal_decompose
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, TensorDataset

from SimpleLSTMModel import SimpleLSTMModel

pl.seed_everything(42, workers=True)
import random

random.seed(42)
import numpy as np

np.random.seed(42)

import pandas as pd

pd.options.mode.chained_assignment = None
import warnings

data_path = sys.argv[1]
# data_path = "drive/MyDrive/DATAFORMODELtrain250724.xlsx"

forecast_start_date = sys.argv[2]
# forecast_start_date = "2024-06-01 23:00"

forecast_end_date = sys.argv[3]
# forecast_end_date = "2024-07-01 23:00"

variables = sys.argv[4]
# variables = "[ROSOLGEN,GAS,CO2,T2MALL,WS10MRO,RORRO,RORSE,AT_HU,DEWINDGEN]"
variables = list(map(str.strip, variables.lstrip("[").rstrip("]").split(",")))


def add_moving_averages(df, column_list, windows):
    for column in column_list:
        for window in windows:
            df[f"{column}_ma_{window}"] = df[column].rolling(window=window).mean()
    return df


def recursive_forecast(
    df, start_idx, end_idx, seq_len, model, feature_scaler, target_scaler, ma_windows
):

    forecast_df = pd.DataFrame()
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.weekday
    df["is_weekend"] = df["Date"].dt.weekday.isin([5, 6]).apply(int)

    df["hour"] = df["Date"].dt.hour

    df["peak_event"] = (df["Date"].dt.year.isin([2023, 2022])) | (
        df["Date"] >= pd.to_datetime("2024/06/01")
    ).apply(int)

    for i in range(12):
        df[f"month_{i}"] = (df["Date"].dt.month == i).values.astype(int)

    for i in range(7):
        df[f"week_{i}"] = (df["Date"].dt.weekday == i).values.astype(int)

    for i in range(24):
        df[f"hour_{i}"] = (df["Date"].dt.hour == i).values.astype(int)

    df.loc[start_idx:end_idx, "PriceSK"] = 0
    cols = (
        variables
        + [
            "peak_event",
            "month",
            "hour",
            "week",
            "is_weekend",
        ]
        + [f"month_{i}" for i in range(12)]
        + [f"week_{i}" for i in range(7)]
        + [f"hour_{i}" for i in range(24)]
    )
    ignore_last_cols = 5 + 12 + 7 + 24
    t_plus_1_feature_len = len(cols)

    for current_idx in range(start_idx, end_idx + 1):
        if current_idx < seq_len - 1:
            print("Not enough data to start forecasting")
            forecast_df = pd.concat(
                [
                    forecast_df,
                    pd.DataFrame(
                        {"Date": [df.iloc[current_idx]["Date"]], "Forecast": [0]}
                    ),
                ]
            )

        # Prepare the input sequence
        df = add_moving_averages(df, ["PriceSK"], windows)
        feature_cols = df.drop(columns=["PriceSK", "Date"]).columns
        X_last = df.iloc[current_idx - seq_len : current_idx][feature_cols]
        y_last = df.iloc[current_idx - seq_len : current_idx][["PriceSK"]]
        X_t_plus_1_features = df.iloc[[current_idx]][feature_cols]

        X_last_scaled = feature_scaler.transform(
            X_last[feature_scaler.get_feature_names_out()]
        )
        y_last_scaled = target_scaler.transform(y_last)
        X_last_scaled_combined = np.hstack(
            [X_last_scaled, y_last_scaled],
        ).reshape(1, seq_len, -1)

        X_t_plus_1_features_scaled = feature_scaler.transform(
            X_t_plus_1_features[feature_scaler.get_feature_names_out()]
        )[:, :t_plus_1_feature_len]

        # Convert the last sequence to a tensor
        X_tensor = torch.tensor(X_last_scaled_combined, dtype=torch.float32)
        t_plus_1_features_tensor = torch.tensor(
            X_t_plus_1_features_scaled, dtype=torch.float32
        )

        # Make prediction
        with torch.no_grad():
            y_pred_scaled = model(X_tensor, t_plus_1_features_tensor).numpy()

        # Inverse transform the prediction
        y_pred = target_scaler.inverse_transform(y_pred_scaled)

        # Update the DataFrame with the predicted value
        df.at[current_idx, "PriceSK"] = y_pred[0, 0]

        forecast_df = pd.concat(
            [
                forecast_df,
                pd.DataFrame(
                    {"Date": [df.iloc[current_idx]["Date"]], "Forecast": [y_pred[0, 0]]}
                ),
            ]
        )

    return forecast_df


seq_len = 24
windows = [12, 24, 36, 48, 7 * 24]  # Moving Avg Windows
feature_scaler = joblib.load("feature_scaler.joblib")
target_scaler = joblib.load("target_scaler.joblib")

model = SimpleLSTMModel(
    **json.load(open("lstm_hyperparameters.json")),
    target_scaler=target_scaler,
)
model.load_state_dict(torch.load("lstm_network_state_es.pth", weights_only=False))
model = model.to("cpu")
print("Loaded the model")


df = pd.read_excel(data_path)
df["Date"] = pd.to_datetime(df["Date"])
df["Date"] = df["Date"].apply(lambda x: x.round(freq="h"))


forecast_idx_list = df[
    (df["Date"] >= pd.to_datetime(forecast_start_date))
    & (df["Date"] <= pd.to_datetime(forecast_end_date))
].index
start_idx = forecast_idx_list[0]
num_steps = len(forecast_idx_list)
output = recursive_forecast(
    df,
    start_idx,
    forecast_idx_list[-1],
    seq_len,
    model,
    feature_scaler,
    target_scaler,
    windows,
)
output.to_csv("forecast.csv", index=False)
print("Forecasting Completed & result is stored")
