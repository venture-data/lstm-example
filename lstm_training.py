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

pl.seed_everything(42, workers=True)

# Check and set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None
import warnings


def add_moving_averages(df, column_list, windows):
    """
    Applies moving averages to the specified columns for the given windows.
    Parameters:
    - df: pandas DataFrame containing the data.
    - column_list: List of column names to which moving averages will be applied.
    - windows: List of window sizes for which moving averages will be computed.
    
    Returns:
    - df: pandas DataFrame with moving averages added.
    """
    for column in column_list:
        for window in windows:
            df[f"{column}_ma_{window}"] = df[column].rolling(window=window, min_periods=1).mean()
    return df


# Reading command-line arguments
data_path = sys.argv[1]  # e.g., "DATAFORMODELtrain200824.xlsx"
train_start_date = sys.argv[2]  # e.g., "2018-01-01 23:00"
train_end_date = sys.argv[3]  # e.g., "2024-06-30 23:00"
variables = sys.argv[4]  # e.g., "[PriceRO, PriceSK, PriceCZ, COAL, GAS, AT_HU, COALTOGAS, CO2, UNAVHYDRALL, UNAVLGNSK, UNAVHYDRBG]"

# Convert input strings to appropriate Python objects
train_start_date = pd.Timestamp(train_start_date)
train_end_date = pd.Timestamp(train_end_date)
variables = list(map(str.strip, variables.lstrip("[").rstrip("]").split(",")))

# Check that input data is correctly formatted
if not all(isinstance(v, str) for v in variables):
    raise ValueError("The variables list must contain string column names.")

# Moving average window sizes
windows = [12, 24, 36, 48, 7 * 24]

# Read the dataset
df = pd.read_excel(data_path)
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y %H:%M").apply(lambda x: x.round(freq="h"))

# Ensure the target variable 'PriceHU' is in the dataset
assert 'PriceHU' in df.columns, "Target variable 'PriceHU' not found in the dataset!"

# Remove rows where target variable is NaN
df = df.dropna(subset=["PriceHU"])

# Extract time-based features
df["month"] = df["Date"].dt.month
df["week"] = df["Date"].dt.weekday
df["is_weekend"] = df["Date"].dt.weekday.isin([5, 6]).apply(int)
df["hour"] = df["Date"].dt.hour
df["peak_event"] = (df["Date"].dt.year.isin([2023, 2022])) | (df["Date"] >= pd.to_datetime("2024/06/01")).apply(int)

# Add one-hot encoded columns for month, week, and hour
for i in range(12):
    df[f"month_{i}"] = (df["Date"].dt.month == i).astype(int)
for i in range(7):
    df[f"week_{i}"] = (df["Date"].dt.weekday == i).astype(int)
for i in range(24):
    df[f"hour_{i}"] = (df["Date"].dt.hour == i).astype(int)

# Define date ranges for validation and test splits
val_ratio = 0.1  # 10% for validation
test_ratio = 0.05  # 5% for testing

total_days = (train_end_date - train_start_date).days
train_days = int(total_days * (1 - val_ratio - test_ratio))
val_days = int(total_days * val_ratio)
test_days = total_days - train_days - val_days

# Calculate the end date for train, validation, and test sets
val_start_date = train_start_date + pd.Timedelta(days=train_days)
val_end_date = val_start_date + pd.Timedelta(days=val_days)
test_start_date = val_end_date + pd.Timedelta(days=1)
test_end_date = test_start_date + pd.Timedelta(days=test_days)

# Split the data into train, validation, and test sets
train_df = df[(df["Date"] >= train_start_date) & (df["Date"] <= val_start_date)]
val_df = df[(df["Date"] > val_start_date) & (df["Date"] <= val_end_date)]
test_df = df[(df["Date"] > val_end_date) & (df["Date"] <= test_end_date)]

# Apply moving averages to all relevant columns including the target column
relevant_columns = variables + ["PriceHU"]
train_df = add_moving_averages(train_df, relevant_columns, windows)
val_df = add_moving_averages(val_df, relevant_columns, windows)
test_df = add_moving_averages(test_df, relevant_columns, windows)

# Drop rows with any NaN values created by moving averages or already present
train_df = train_df.dropna()
val_df = val_df.dropna()
test_df = test_df.dropna()

# Initialize scalers
feature_scaler = RobustScaler()
target_scaler = RobustScaler()

# Define feature and target columns
feature_cols = train_df.drop(columns=["PriceHU", "Date"]).columns

# Prepare features (X) and target (y)
X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
X_test = test_df[feature_cols]

y_train = train_df[["PriceHU"]]
y_val = val_df[["PriceHU"]]
y_test = test_df[["PriceHU"]]

# Scale features and target separately
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

# Define sequence length (number of time steps/ lag features)
seq_len = 24  # Example: Using 24 time steps (e.g., 24 hours if working with hourly data)

# Determine the number of T+1 features
t_plus_1_feature_len = len(variables)  # Number of features to predict at the next time step

# Create sequences for LSTM input
X_train_seq, T_plus1_train_seq, y_train_seq = create_sequences(
    X_train_scaled, y_train_scaled, seq_len, t_plus_1_feature_len
)
X_val_seq, T_plus1_val_seq, y_val_seq = create_sequences(
    X_val_scaled, y_val_scaled, seq_len, t_plus_1_feature_len
)
X_test_seq, T_plus1_test_seq, y_test_seq = create_sequences(
    X_test_scaled, y_test_scaled, seq_len, t_plus_1_feature_len
)

# Convert sequences to PyTorch tensors
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
train_dataset = TensorDataset(X_train_tensor, T_plus1_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, T_plus1_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, T_plus1_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define model parameters
input_size = X_train_tensor.shape[2]
t_plus1_dim = T_plus1_test_seq.shape[1]
output_size = 1

# Hyperparameter tuning using Optuna
def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 32, 150)
    hidden_size1 = trial.suggest_int("hidden_size1", 32, 150)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)

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

    # Create a PyTorch Lightning Trainer
    trainer = Trainer(
        max_epochs=4,
        accelerator="auto",
        devices=1,
        precision=32,  # Use mixed precision if available
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)  # Number of trials
best_params = study.best_params

best_params["input_size"] = input_size
best_params["output_size"] = output_size
best_params["t_plus1_dim"] = t_plus1_dim

print("Hyperparameter Tuning Ended")

# Save hyperparameters
with open("hyperparameters_finetuning.json", "w") as file:
    json.dump(best_params, file, indent=4)

# Instantiate the best model
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
    devices=1,
    callbacks=[checkpoint_callback, early_stopping_callback],
)

# Train the model with best hyperparameters
trainer.fit(model, train_loader, val_loader)

# Save scalers
joblib.dump(feature_scaler, "feature_scaler.joblib")
joblib.dump(target_scaler, "target_scaler.joblib")

# Save model checkpoint
model = SimpleLSTMModel.load_from_checkpoint(
    checkpoint_callback.best_model_path,
    **json.load(open("lstm_hyperparameters.json")),
)
torch.save(model.state_dict(), "lstm_network_state_es.pth")

print("Model & Scaler saved successfully....")

# Test the model
trainer.test(model, test_loader)
