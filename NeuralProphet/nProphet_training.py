import sys
import pandas as pd
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt

# Read command-line arguments
input_file = sys.argv[1]
start_date = sys.argv[2]
end_date = sys.argv[3]

# Define the forecast period
forecast_start_date = "2024-07-01 00:00"
forecast_end_date = "2024-07-21 23:00"

print(f"Loading dataset from {input_file}...")
# Load the dataset
df = pd.read_csv(input_file)

# Convert the 'ds' column to datetime format
df['ds'] = pd.to_datetime(df['ds'])
print(f"Dataset loaded with {len(df)} rows.")

# Filter the dataset based on the provided training date range
print(f"Filtering data for training from {start_date} to {end_date}...")
df_filtered = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]
print(f"Training dataset prepared with {len(df_filtered)} rows.")

# Initialize the NeuralProphet model with customized parameters
print("Initializing NeuralProphet model...")
model = NeuralProphet(
    growth="linear",  # Assuming no saturation in fuel prices; use "logistic" if there is a saturation point.
    n_changepoints=40,  # Increased number of changepoints to handle high variability
    changepoints_range=0.95,  # Allow changepoints across most of the training data
    trend_reg=0.1,  # Small regularization to avoid overfitting trend changes
    trend_global_local="global",  # Use a global trend model
    yearly_seasonality="auto",  # Disable yearly seasonality unless it's relevant
    weekly_seasonality="auto",  # Let the model automatically detect weekly seasonality
    daily_seasonality="auto",  # Let the model automatically detect daily seasonality
    seasonality_mode="additive",  # Additive seasonality to handle independent seasonal effects
    seasonality_reg=0.1,  # Small regularization to avoid overfitting seasonal patterns
    n_forecasts=504,  # Predict the next 3 weeks (24 hours * 21 days)
    learning_rate=0.005,  # Set a stable learning rate
    epochs=1000,  # Use a high number of epochs due to large dataset size
    batch_size=1024,  # Batch size based on available hardware, use higher if running on a GPU
    optimizer="AdamW",  # Use AdamW optimizer for efficient training with weight decay
    impute_missing=True,  # Enable missing value imputation
    normalize="auto",  # Automatic normalization to scale the data appropriately
)

# Add all columns except 'ds' and 'y' as regressors
print("Adding future regressors to the model...")
for col in df_filtered.columns:
    if col not in ['ds', 'y']:
        model = model.add_future_regressor(name=col)
print("All future regressors added.")

# Train the model
print("Training the model...")
model.fit(df_filtered, freq='h')
print("Model training complete.")

# Save the trained model to a file
model_file = 'neuralprophet_model.pth'
model.save(model_file)
print(f"Model saved to {model_file}.")

# Create a dataframe for future predictions using the actual future data from 'data_prep.csv'
print(f"Preparing future dataframe for predictions from {forecast_start_date} to {forecast_end_date}...")
df_future = df[(df['ds'] >= forecast_start_date) & (df['ds'] <= forecast_end_date)].copy()
print(f"Future dataframe prepared with {len(df_future)} rows.")

# Ensure future regressors are available in df_future
if 'is_weekend' in df_future.columns:
    print("'is_weekend' values are correctly loaded for future dates.")

# Make predictions
print("Making predictions...")
forecast = model.predict(df_future)
print("Predictions completed.")

# Save the forecast to a CSV file
forecast_output_file = 'nProphet_forecast.csv'
forecast.to_csv(forecast_output_file, index=False)
print(f"Forecast results saved to {forecast_output_file}.")

print("Forecasting process completed successfully.")