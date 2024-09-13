import sys
import pandas as pd
import warnings
from neuralprophet import NeuralProphet
import os
from sklearn.feature_selection import mutual_info_regression
import numpy as np
# import matplotlib.pyplot as plt
# Suppress all warnings
warnings.filterwarnings("ignore")

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the file name for the model
model_file = os.path.join(script_dir, 'neuralprophet_model.pkl') 
feature_csv_file = os.path.join(script_dir, 'features_for_nProphet.csv') 

def add_lagged_features(data, column, lags):
    """
    Adds lagged features to the DataFrame for the specified column.

    Parameters:
    - data: DataFrame containing the data.
    - column: The column to create lagged features for.
    - lags: A list of lag periods to use.

    Returns:
    - DataFrame with lagged features added.
    """
    for lag in lags:
        data[f'lag_{lag}'] = data[column].shift(lag)
    return data

def add_rolling_window_features(data, column, windows):
    """
    Adds rolling window features (mean and std) to the DataFrame for the specified column.

    Parameters:
    - data: DataFrame containing the data.
    - column: The column to create rolling window features for.
    - windows: A list of window sizes to use.

    Returns:
    - DataFrame with rolling window features added.
    """
    for window in windows:
        data[f'rolling_mean_{window}h'] = data[column].rolling(window).mean()
        data[f'rolling_std_{window}h'] = data[column].rolling(window).std()
    return data

def add_exponential_moving_average(data, column, ema_windows):
    """
    Adds exponential moving average (EMA) features to the DataFrame for the specified column.

    Parameters:
    - data: DataFrame containing the data.
    - column: The column to create EMA features for.
    - ema_windows: A list of EMA window sizes to use.

    Returns:
    - DataFrame with EMA features added.
    """
    for window in ema_windows:
        data[f'ema_{window}h'] = data[column].ewm(span=window).mean()
    return data
# Read command-line arguments
input_file = sys.argv[1]
start_date = sys.argv[2]
end_date = sys.argv[3]

# Check the number of command-line arguments
if len(sys.argv) == 4:
    is_automatic = True
elif len(sys.argv) == 5:
    manual_regressors = sys.argv[4]
    print(f"'manual_regressors' argument exists: {manual_regressors}")
    is_automatic = False
else:
    print("Invalid number of arguments passed. Expected 3 or 4 arguments.")
    sys.exit(1)

min_date = pd.to_datetime('2017-01-01 00:00')
cutoff_date = pd.to_datetime("2024-08-20 23:00")

# Validate the dates
# if start_date < min_date or end_date > cutoff_date:
#     print(f"Error: The provided date range [{start_date}, {end_date}] is out of bounds.")
#     print(f"Start date cannot be less than {min_date} and end date cannot exceed {cutoff_date}.")
#     sys.exit(1)

print(f"Loading dataset from {input_file}...")
# Load the dataset
df = pd.read_csv(input_file)
print(f"Dataset loaded with {len(df)} rows.")
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')  # Adjust the format if necessary
df.loc[:, "Date"] = df["Date"].dt.round('h')

print(f"Defining minimum and maximum dates data for training from {min_date} to {cutoff_date}...")
data = df[(df['Date'] >= min_date) & (df['Date'] <= cutoff_date)]
data["Date"] = data["Date"].dt.round('h')

if is_automatic:
    print("Started Automatic feature selection")
    columns_to_drop = [
    'Y', 'M', 'Day', 'H', 'Y2016',	'Y2017',	'Y2018',	'Y2019',	'Y2020',	'Y2021',	'Y2022',	'Y2023',	'Y2024',
    'M1',	'M2',	'M3',	'M4',	'M5',	'M6',	'M7',	'M8',	'M9',	'M10',	'M11',	'M12',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
    'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19',
    'h20', 'h21', 'h22', 'h23', 'h24',
    'PriceCZ', 'PriceSK', 'PriceRO'
    ]
    data = data.drop(columns=columns_to_drop)
    
    # Feature Engineering
    print("Simple Time based features generated")
    data['hour'] = data['Date'].dt.hour
    data['day_of_week'] = data['Date'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    data['month'] = data['Date'].dt.month

    # Cyclical time-based features
    print("Cyclical time-based features generated")
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['sin_day_of_week'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['cos_day_of_week'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    column = 'PriceHU'
    
    print("Added Lags")
    lags = [1, 3, 6, 12, 24, 48, 72, 168]
    data = add_lagged_features(data, column, lags)
    
    print("Added rolling windows")
    windows = [3, 6, 12, 24, 168]
    data = add_rolling_window_features(data, column, windows)
    
    print("Added ema windows")
    ema_windows = [12, 24, 168]
    data = add_exponential_moving_average(data, column, ema_windows)
    
    # data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    data = data.dropna()
    
    # Mutual Information for Feature Selection
    X = data.drop(['PriceHU', 'Date'], axis=1)
    y = data['PriceHU']

    print("Started Mutual Information Score calculation. this will take a short while.")
    mi = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi, name="MI Scores", index=X.columns).sort_values(ascending=False)
    
    top_features = mi_scores[mi_scores > 0.3].index  # For example, keep features with MI score > 0.5
    X_filtered = X[top_features]
    top_features = top_features.tolist()
    
    columns_to_save = top_features + ['Date', 'PriceHU', 'WDAY']
else:
    # Convert the string argument to a list of provided manual regressors
    provided_features = eval(manual_regressors)


data = data[columns_to_save]
data.to_csv(feature_csv_file, index=False)
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Convert the 'ds' column to datetime format
data['ds'] = data['Date']
data['y'] = data['PriceHU']
data = data.drop(columns=['Date', 'PriceHU'])
data['ds'] = data['ds'].dt.round('h')

print(f"Training dataset prepared with {len(data)} rows.")

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
    # epochs=1000,  # Use a high number of epochs due to large dataset size
    # batch_size=1024,  # Batch size based on available hardware, use higher if running on a GPU
    optimizer="AdamW",  # Use AdamW optimizer for efficient training with weight decay
    impute_missing=True,  # Enable missing value imputation
    normalize="auto",  # Automatic normalization to scale the data appropriately
)

# Add future regressors and lagged regressors
print("Adding future regressors and lagged regressors to the model...")
for col in data.columns:
    if col not in ['ds', 'y']:
            model = model.add_future_regressor(name=col)

print("All future and lagged regressors added.")

# Train the model
print("Training the model...")
model.fit(data, freq='h')
print("Model training complete.")

# Create a dataframe for future predictions using the actual future data from 'data_prep.csv'
print(f"Preparing future dataframe for predictions from {end_date} to {cutoff_date}...")
df_future = data[(data['ds'] > end_date) & (data['ds'] <= cutoff_date)].copy()
print(f"Future dataframe prepared with {len(df_future)} rows.")

# Make predictions
print("Making predictions...")
forecast = model.predict(df_future)
print("Predictions completed.")

# Save the forecast to a CSV file
forecast_output_file = 'nProphet_forecast.csv'
forecast.to_csv(forecast_output_file, index=False)
print(f"Forecast results saved to {forecast_output_file}.")

print("Forecasting process completed successfully.")