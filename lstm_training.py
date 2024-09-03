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
from pytorch_lightning import Trainer
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from torch.utils.data import DataLoader, TensorDataset

from SimpleLSTMModel import SimpleLSTMModel

pl.seed_everything(42, workers=True)
import random

random.seed(42)
import numpy as np

np.random.seed(42)

import pandas as pd

pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None
import warnings


def add_moving_averages(df, column_list, windows):
    for column in column_list:
        for window in windows:
            df[f"{column}_ma_{window}"] = df[column].rolling(window=window).mean()
    return df


data_path = sys.argv[1]
# data_path = "drive/MyDrive/DATAFORMODELtrain250724.xlsx"

train_start_date = sys.argv[2]
# train_start_date = "2016-01-01 0:00" # Data used for Training

train_end_date = sys.argv[3]
# train_end_date = "2024-05-31 23:00" # Data used to validate the weights assigned, rest of the data is used for testing

train_start_date = pd.Timestamp(train_start_date)
train_end_date = pd.Timestamp(train_end_date)

variables = sys.argv[4]
# variables = "[ROSOLGEN,GAS,CO2,T2MALL,WS10MRO,RORRO,RORSE,AT_HU,DEWINDGEN]"

variables = list(map(str.strip, variables.lstrip("[").rstrip("]").split(",")))
# Windows For Moving Average
windows = [12, 24, 36, 48, 7 * 24]


# Calculate the total time span
total_days = (train_end_date - train_start_date).days

# Define split ratios
train_ratio = 0.85
val_ratio = 0.10
test_ratio = 0.05

# Calculate the number of days for each period
train_days = int(total_days * train_ratio)
val_days = int(total_days * val_ratio)
test_days = (
    total_days - train_days - val_days
)  # Ensures the total period is correctly split

# Define the date splits
train_end_date = train_start_date + pd.Timedelta(days=train_days)
val_start_date = train_end_date + pd.Timedelta(days=1)
val_end_date = val_start_date + pd.Timedelta(days=val_days)
test_start_date = val_end_date + pd.Timedelta(days=1)
test_end_date = test_start_date + pd.Timedelta(days=test_days)


print(data_path)
print(train_start_date)
print(train_end_date)
print(variables)
print(windows)
# read in dataset
print("Reading Data")


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

# Define sequence length (number of time steps/ lag features)
seq_len = 24

df = pd.read_excel(data_path)
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y %H:%M")
df["Date"] = df["Date"].apply(lambda x: x.round(freq="h"))

df = df.dropna(subset=["PriceSK"])

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

# Splitting on specific times
train_df = df[(df["Date"] <= pd.to_datetime(train_end_date))]
val_df = df[
    (df["Date"] > pd.to_datetime(train_end_date))
    & (df["Date"] <= pd.to_datetime(val_end_date))
]
test_df = df[
    (df["Date"] > pd.to_datetime(val_end_date))
    & (df["Date"] <= pd.to_datetime(test_end_date))
]


t_plus_1_feature_len = len(cols)
train_df = train_df[["Date"] + cols + ["PriceSK"]]
test_df = test_df[["Date"] + cols + ["PriceSK"]]


train_df = add_moving_averages(train_df, ["PriceSK"], windows)
val_df = add_moving_averages(val_df, ["PriceSK"], windows)
test_df = add_moving_averages(test_df, ["PriceSK"], windows)


# Drop rows with NaN values created by moving averages
train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()

# Initialize the scalers
feature_scaler = RobustScaler()
target_scaler = RobustScaler()

# Define feature and target columns
feature_cols = train_df.drop(columns=["PriceSK", "Date"]).columns

# Define X and Y
X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
X_test = test_df[feature_cols]

y_train = train_df[["PriceSK"]]
y_val = val_df[["PriceSK"]]
y_test = test_df[["PriceSK"]]

# Scale Fit + Transform features and target separately
X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled = feature_scaler.transform(X_val)
X_test_scaled = feature_scaler.transform(X_test)

y_train_scaled = target_scaler.fit_transform(y_train)
y_val_scaled = target_scaler.transform(y_val)
y_test_scaled = target_scaler.transform(y_test)


def create_sequences(data, target, seq_len, t_plus1_feature_len):
    """Create Lag, T+1 features & Target sequences"""
    sequences = []
    targets = []
    t_plus_1_features = []

    for i in range(len(data) - seq_len):
        seq = data[i : i + seq_len]
        lagged_targets = target[i : i + seq_len]
        seq_with_lags = np.hstack((seq, lagged_targets.reshape(-1, 1)))

        target_seq = target[i + seq_len]

        t_plus_1 = data[i + seq_len][:t_plus1_feature_len]

        sequences.append(seq_with_lags)
        targets.append(target_seq)
        t_plus_1_features.append(t_plus_1)
    return np.array(sequences), np.array(t_plus_1_features), np.array(targets)


# Create sequences
X_train_seq, T_plus1_train_seq, y_train_seq = create_sequences(
    X_train_scaled, y_train_scaled, seq_len, t_plus_1_feature_len
)
X_val_seq, T_plus1_val_seq, y_val_seq = create_sequences(
    X_val_scaled, y_val_scaled, seq_len, t_plus_1_feature_len
)
X_test_seq, T_plus1_test_seq, y_test_seq = create_sequences(
    X_test_scaled, y_test_scaled, seq_len, t_plus_1_feature_len
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
T_plus1_train_tensor = torch.tensor(T_plus1_train_seq, dtype=torch.float32)


X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
T_plus1_val_tensor = torch.tensor(T_plus1_val_seq, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)
T_plus1_test_tensor = torch.tensor(T_plus1_test_seq, dtype=torch.float32)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, T_plus1_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, T_plus1_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, T_plus1_test_tensor, y_test_tensor)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define model parameters
input_size = X_train_tensor.shape[2]
t_plus1_dim = T_plus1_test_seq.shape[1]
output_size = 1


def objective(trial):
    # Hyperparameters to tune
    hidden_size = trial.suggest_int("hidden_size", 32, 150)
    hidden_size1 = trial.suggest_int("hidden_size1", 32, 150)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)

    # Instantiate the model with trial hyperparameters
    model = SimpleLSTMModel(
        input_size=input_size,
        t_plus1_dim=t_plus1_dim,
        hidden_size=hidden_size,
        hidden_size1=hidden_size1,
        num_layers=num_layers,
        output_size=output_size,
        learning_rate=learning_rate,
        target_scaler=target_scaler,
    )

    # Use an existing Trainer instance
    trainer = Trainer(
        max_epochs=4,
        accelerator="auto",
        precision=32,  # Use mixed precision if available
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Evaluate on validation set
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # Number of trials
best_params = study.best_params

best_params["input_size"] = input_size
best_params["output_size"] = output_size
best_params["t_plus1_dim"] = t_plus1_dim
print("Hyperparameter Tuning Ended")

print("Initiating Model Training with best Hyperparameters")
with open("hyperparameters_finetuning.json", "w") as file:
    json.dump(best_params, file, indent=4)

# Instantiate the best model
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor
    dirpath="checkpoints",  # Directory to save checkpoints
    filename="best-model",  # File name of the best checkpoint
    save_top_k=1,  # Save only the best model
    mode="min",  # Mode for monitoring: 'min' for loss, 'max' for accuracy
)
early_stopping_callback = pl.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,  # Allow more epochs without improvement
    mode="min",
    verbose=3,
)

model = SimpleLSTMModel(
    best_params["input_size"],
    t_plus1_dim=best_params["t_plus1_dim"],
    hidden_size=best_params["hidden_size"],
    hidden_size1=best_params["hidden_size1"],
    output_size=best_params["output_size"],
    num_layers=best_params["num_layers"],
    learning_rate=best_params["learning_rate"],
    target_scaler=target_scaler,
)

# Initialize PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    callbacks=[checkpoint_callback, early_stopping_callback],
)

# Train the model
trainer.fit(
    model,
    train_loader,
    val_loader,
)

# Saving data
with open("lstm_hyperparameters.json", "w") as file:
    json.dump(best_params, file, indent=4)
joblib.dump(feature_scaler, "feature_scaler.joblib")
joblib.dump(target_scaler, "target_scaler.joblib")

model = SimpleLSTMModel.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    **json.load(open("lstm_hyperparameters.json")),
)
torch.save(model.state_dict(), "lstm_network_state_es.pth")
model = model.to("cpu")

print("Model & Scaler saved successfully....")

# Test the model
trainer.test(model, test_loader)
