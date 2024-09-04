# tft_training.py

import sys
import pandas as pd
import numpy as np
import json
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tft_implementation import create_tft_model, prepare_dataloader


def read_and_prepare_data(file_path, start_date, end_date, columns):
    # Read Excel file
    df = pd.read_excel(file_path, parse_dates=["Date"])

    # Filter data by date range
    df = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]

    # Select only the required columns
    df = df[["Date"] + columns + ["PriceSK"]]  # Include target variable

    # Handle missing values using interpolation
    df = df.interpolate(method='linear', limit_direction='both')

    # Extract time-based features
    df["month"] = df["Date"].dt.month
    df["weekday"] = df["Date"].dt.weekday
    df["is_weekend"] = df["Date"].dt.weekday.isin([5, 6]).astype(int)
    df["hour"] = df["Date"].dt.hour

    return df


def objective(trial, train_dataloader):
    # Define hyperparameters to optimize
    hidden_size = trial.suggest_int("hidden_size", 8, 64)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    
    # Create TFT model with given hyperparameters
    model = create_tft_model(train_dataloader, max_encoder_length=168, max_prediction_length=24, hidden_size=hidden_size, learning_rate=learning_rate, dropout=dropout)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 10  # Number of epochs for training
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            # Forward pass
            output = model(batch)
            loss = output.loss  # Using loss from PyTorch Forecasting
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    # After training, evaluate the model
    val_loss = evaluate_model(model, train_dataloader)

    return val_loss


def evaluate_model(model, dataloader):
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
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, MAPE: {mape}")

    # Return the main metric to optimize (e.g., MAE)
    return mae


def main(file_path, start_date, end_date, columns, time_varying_unknown_reals):
    # Read and prepare the data
    data = read_and_prepare_data(file_path, start_date, end_date, columns)

    # Prepare data for TimeSeriesDataSet
    train_dataloader = prepare_dataloader(data, max_encoder_length=168, max_prediction_length=24, batch_size=64, time_varying_unknown_reals=time_varying_unknown_reals)

    # Hyperparameter optimization using Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_dataloader), n_trials=50)

    # Save the best hyperparameters
    best_params = study.best_params
    with open("best_hyperparameters.json", "w") as json_file:
        json.dump(best_params, json_file)

    print("Best hyperparameters saved to best_hyperparameters.json")


if __name__ == "__main__":
    # Get command-line arguments
    file_path = sys.argv[1]
    start_date = pd.to_datetime(sys.argv[2])
    end_date = pd.to_datetime(sys.argv[3])
    columns = json.loads(sys.argv[4].replace("'", '"'))  # Convert list string to actual list
    time_varying_unknown_reals = json.loads(sys.argv[5].replace("'", '"'))  # Convert list string to actual list

    # Run the main function
    main(file_path, start_date, end_date, columns, time_varying_unknown_reals)
