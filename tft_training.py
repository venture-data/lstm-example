# tft_training.py

import sys
import pandas as pd
import numpy as np
import json
import optuna
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tft_implementation import create_tft_model, prepare_dataloader


def read_and_prepare_data(file_path, start_date, end_date, columns):
    print(f"Reading data from: {file_path}")
    df = pd.read_excel(file_path, parse_dates=["Date"])

    print("Converting 'Date' column to datetime format...")
    # Specify the format that matches your datetime strings
    # df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d %H:%M:%S")
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%y %H:%M").apply(lambda x: x.round(freq="h"))

    print(f"Filtering data between {start_date} and {end_date}...")
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    print("Selecting required columns...")
    print(f"Columns to include: {columns + ['PriceSK']}")
    df = df[["Date"] + columns + ["PriceSK"]]  # Include target variable

    print("Handling missing values with linear interpolation...")
    df = df.interpolate(method='linear', limit_direction='both')

    print("Extracting time-based features...")
    df["month"] = df["Date"].dt.month
    df["weekday"] = df["Date"].dt.weekday
    df["is_weekend"] = df["Date"].dt.weekday.isin([5, 6]).astype(int)
    df["hour"] = df["Date"].dt.hour

    print("Data preparation completed. Returning processed DataFrame.")
    return df



def objective(trial, train_dataloader):
    print("Starting hyperparameter optimization trial...")
    # Define hyperparameters to optimize
    hidden_size = trial.suggest_int("hidden_size", 8, 64)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)

    print(f"Trial hyperparameters - hidden_size: {hidden_size}, learning_rate: {learning_rate}, dropout: {dropout}")

    # Create TFT model with given hyperparameters
    model = create_tft_model(train_dataloader, max_encoder_length=168, max_prediction_length=24, hidden_size=hidden_size, learning_rate=learning_rate, dropout=dropout)

    print("Model created, initializing optimizer...")
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 10  # Number of epochs for training
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            # Forward pass
            output = model(batch)
            loss = output.loss  # Using loss from PyTorch Forecasting
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed.")

    print("Training completed, evaluating model...")
    # After training, evaluate the model
    val_loss = evaluate_model(model, train_dataloader)

    print(f"Validation loss: {val_loss}")
    return val_loss


def evaluate_model(model, dataloader):
    print("Evaluating model performance...")
    model.eval()  # Set model to evaluation mode
    true_values = []
    predictions = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for batch in dataloader:
            # Get the model predictions
            output = model(batch)
            preds = output.output.detach().cpu().numpy()  # Model predictions
            true = batch["target"].detach().cpu().numpy()  # True values

            predictions.extend(preds)
            true_values.extend(true)

    # Convert to NumPy arrays for metric calculation
    true_values = np.array(true_values)
    predictions = np.array(predictions)

    # Calculate evaluation metrics
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(true_values, predictions)

    # Print evaluation metrics
    print(f"Evaluation Metrics -> MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}")

    # Return the main metric to optimize (e.g., MAE)
    return mae


def main(file_path, start_date, end_date, columns, time_varying_unknown_reals):
    print("Starting data preparation...")
    # Read and prepare the data
    data = read_and_prepare_data(file_path, start_date, end_date, columns)

    print("Preparing dataloader for training...")
    # Prepare data for TimeSeriesDataSet
    train_dataloader = prepare_dataloader(data, max_encoder_length=168, max_prediction_length=24, batch_size=64, time_varying_unknown_reals=time_varying_unknown_reals)

    print("Initializing hyperparameter optimization with Optuna...")
    # Hyperparameter optimization using Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_dataloader), n_trials=50)

    print("Optimization completed. Saving best hyperparameters...")
    # Save the best hyperparameters
    best_params = study.best_params
    with open("best_hyperparameters.json", "w") as json_file:
        json.dump(best_params, json_file)

    print("Best hyperparameters saved to best_hyperparameters.json")


if __name__ == "__main__":
    print("Parsing command-line arguments...")
    # Get command-line arguments
    file_path = sys.argv[1]
    start_date = pd.to_datetime(sys.argv[2])
    end_date = pd.to_datetime(sys.argv[3])
    columns = json.loads(sys.argv[4].replace("'", '"'))  # Convert list string to actual list
    time_varying_unknown_reals = json.loads(sys.argv[5].replace("'", '"'))  # Convert list string to actual list

    print(f"Arguments received: file_path={file_path}, start_date={start_date}, end_date={end_date}, columns={columns}, time_varying_unknown_reals={time_varying_unknown_reals}")

    # Run the main function
    main(file_path, start_date, end_date, columns, time_varying_unknown_reals)
