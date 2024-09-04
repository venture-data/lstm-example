import json
import sys
import joblib
import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from sklearn.preprocessing import RobustScaler
from SimpleLSTMModel import SimpleLSTMModel
import random
import os

torch.set_float32_matmul_precision('medium')

pl.seed_everything(106, workers=True)
random.seed(106)
np.random.seed(106)

pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None
import warnings


def add_moving_averages(df, column_list, windows):
    # Create a dictionary to store new columns
    ma_dict = {}
    for column in column_list:
        for window in windows:
            ma_dict[f"{column}_ma_{window}"] = df[column].rolling(window=window, min_periods=1).mean()
    # Use pd.concat to add all columns at once
    return pd.concat([df, pd.DataFrame(ma_dict)], axis=1)


# Reading command-line arguments
data_path = sys.argv[1]
train_start_date = pd.Timestamp(sys.argv[2])
train_end_date = pd.Timestamp(sys.argv[3])
variables = list(map(str.strip, sys.argv[4].lstrip("[").rstrip("]").split(",")))
print(f"Loaded parameters:\nData path: {data_path}\nTrain start date: {train_start_date}\nTrain end date: {train_end_date}\nVariables: {variables}")

# Moving average window sizes
windows = [12, 24, 36, 48, 7 * 24]
print(f"Moving average window sizes: {windows}")

# Read and preprocess the dataset
print("Reading dataset...")
df = pd.read_excel(data_path)
print(f"Dataset loaded with shape: {df.shape}")

df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y %H:%M").apply(lambda x: x.round(freq="h"))
assert 'PriceHU' in df.columns, "Target variable 'PriceHU' not found in the dataset!"
df = df.dropna(subset=["PriceHU"])
print(f"Data after dropping NaN in 'PriceHU': {df.shape}")

# Extract time-based features
df["month"] = df["Date"].dt.month
df["week"] = df["Date"].dt.weekday
df["is_weekend"] = df["Date"].dt.weekday.isin([5, 6]).apply(int)
df["hour"] = df["Date"].dt.hour
df["peak_event"] = (df["Date"].dt.year.isin([2023, 2022])) | (df["Date"] >= pd.to_datetime("2024/06/01")).apply(int)

# One-hot encoding for time-based features
print("Creating one-hot encoded time-based features...")
for i in range(12):
    df[f"month_{i}"] = (df["Date"].dt.month == i).astype(int)
for i in range(7):
    df[f"week_{i}"] = (df["Date"].dt.weekday == i).astype(int)
for i in range(24):
    df[f"hour_{i}"] = (df["Date"].dt.hour == i).astype(int)

# Split data into train, validation, and test sets
val_ratio = 0.1
test_ratio = 0.05
total_days = (train_end_date - train_start_date).days
train_days = int(total_days * (1 - val_ratio - test_ratio))
val_days = int(total_days * val_ratio)
test_days = total_days - train_days - val_days

print(f"Splitting data:\nTrain days: {train_days}\nValidation days: {val_days}\nTest days: {test_days}")

val_start_date = train_start_date + pd.Timedelta(days=train_days)
val_end_date = val_start_date + pd.Timedelta(days=val_days)
test_start_date = val_end_date + pd.Timedelta(days=1)
test_end_date = test_start_date + pd.Timedelta(days=test_days)

train_df = df[(df["Date"] >= train_start_date) & (df["Date"] <= val_start_date)]
val_df = df[(df["Date"] > val_start_date) & (df["Date"] <= val_end_date)]
test_df = df[(df["Date"] > val_end_date) & (df["Date"] <= test_end_date)]

print(f"Train set size: {train_df.shape}, Validation set size: {val_df.shape}, Test set size: {test_df.shape}")

# Apply moving averages
print("Applying moving averages...")
relevant_columns = variables + ["PriceHU"]
train_df = add_moving_averages(train_df, relevant_columns, windows)
val_df = add_moving_averages(val_df, relevant_columns, windows)
test_df = add_moving_averages(test_df, relevant_columns, windows)

# Drop rows with NaN values
print("Dropping rows with NaN values...")
train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()
print(f"Data after dropping NaN values:\nTrain: {train_df.shape}\nValidation: {val_df.shape}\nTest: {test_df.shape}")

# Initialize scalers and scale features/target
print("Initializing and applying scalers...")
feature_scaler = RobustScaler()
target_scaler = RobustScaler()
feature_cols = train_df.drop(columns=["PriceHU", "Date"]).columns

X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
X_test = test_df[feature_cols]

y_train = train_df[["PriceHU"]]
y_val = val_df[["PriceHU"]]
y_test = test_df[["PriceHU"]]

X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled = feature_scaler.transform(X_val)
X_test_scaled = feature_scaler.transform(X_test)

y_train_scaled = target_scaler.fit_transform(y_train)
y_val_scaled = target_scaler.transform(y_val)
y_test_scaled = target_scaler.transform(y_test)

print("Data scaling completed.")

def create_sequences(data, target, seq_len, t_plus1_feature_len):
    sequences, targets, t_plus_1_features = [], [], []
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

# Define sequence length and number of T+1 features
seq_len = 24
t_plus_1_feature_len = len(variables)

# Create sequences
print("Creating sequences for LSTM input...")
X_train_seq, T_plus1_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_len, t_plus_1_feature_len)
X_val_seq, T_plus1_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, seq_len, t_plus_1_feature_len)
X_test_seq, T_plus1_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_len, t_plus_1_feature_len)

# Convert sequences to tensors
print("Converting sequences to PyTorch tensors...")
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
T_plus1_train_tensor = torch.tensor(T_plus1_train_seq, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32)
T_plus1_val_tensor = torch.tensor(T_plus1_val_seq, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)
T_plus1_test_tensor = torch.tensor(T_plus1_test_seq, dtype=torch.float32)

# Create DataLoaders
print("Creating DataLoaders...")
train_dataset = TensorDataset(X_train_tensor, T_plus1_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, T_plus1_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, T_plus1_test_tensor, y_test_tensor)

# Determine the number of available CPU cores
num_workers = os.cpu_count()

# Create DataLoaders with dynamically determined num_workers
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers)

# Hyperparameter tuning using Optuna
print("Starting hyperparameter tuning with Optuna...")
def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 32, 150)
    hidden_size1 = trial.suggest_int("hidden_size1", 32, 150)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    # learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)

    model = SimpleLSTMModel(
        input_size=X_train_tensor.shape[2],
        t_plus1_dim=T_plus1_test_seq.shape[1],
        hidden_size=hidden_size,
        hidden_size1=hidden_size1,
        num_layers=num_layers,
        output_size=1,
        learning_rate=learning_rate,
        target_scaler=target_scaler,
    )

    trainer = Trainer(
        max_epochs=4,
        accelerator="auto",
        devices=1,
        precision=32,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_loss = trainer.callback_metrics["val_loss"].item()
    print(f"Trial {trial.number} completed with validation loss: {val_loss}")
    return val_loss

# Run hyperparameter tuning
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")

best_params["input_size"] = X_train_tensor.shape[2]
best_params["output_size"] = 1
best_params["t_plus1_dim"] = T_plus1_test_seq.shape[1]

with open("hyperparameters_finetuning.json", "w") as file:
    json.dump(best_params, file, indent=4)

# Initialize the best model
print("Initializing the best model with tuned hyperparameters...")
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints",
    filename="best-model",
    save_top_k=1,
    mode="min",
)
early_stopping_callback = pl.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    mode="min",
    verbose=True,
)

model = SimpleLSTMModel(
    input_size=best_params["input_size"],
    t_plus1_dim=best_params["t_plus1_dim"],
    hidden_size=best_params["hidden_size"],
    hidden_size1=best_params["hidden_size1"],
    output_size=best_params["output_size"],
    num_layers=best_params["num_layers"],
    learning_rate=best_params["learning_rate"],
    target_scaler=target_scaler,
)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    devices=1,
    callbacks=[checkpoint_callback, early_stopping_callback],
)

print("Starting model training...")
trainer.fit(model, train_loader, val_loader)

# Save scalers and model state
print("Saving scalers and model state...")
joblib.dump(feature_scaler, "feature_scaler.joblib")
joblib.dump(target_scaler, "target_scaler.joblib")

model = SimpleLSTMModel.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    **json.load(open("hyperparameters_finetuning.json")),
)
torch.save(model.state_dict(), "lstm_network_state_es.pth")

print("Model & Scaler saved successfully....")

print("Testing the model...")
trainer.test(model, test_loader)
