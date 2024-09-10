import sys
import pandas as pd
import pickle
from neuralprophet import NeuralProphet
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import os

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the file name for the model
model_file = os.path.join(script_dir, 'neuralprophet_model.pkl') 

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
if start_date < min_date or end_date > cutoff_date:
    print(f"Error: The provided date range [{start_date}, {end_date}] is out of bounds.")
    print(f"Start date cannot be less than {min_date} and end date cannot exceed {cutoff_date}.")
    sys.exit(1)

print(f"Loading dataset from {input_file}...")
# Load the dataset
df = pd.read_excel(input_file)
print(f"Dataset loaded with {len(df)} rows.")



print(f"Defining minimum and maximum dates data for training from {min_date} to {cutoff_date}...")
data = df[(df['Date'] >= min_date) & (df['Date'] <= cutoff_date)]

if is_automatic:
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
    data['hour'] = data['Date'].dt.hour
    data['day_of_week'] = data['Date'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
    data['month'] = data['Date'].dt.month

    # Cyclical time-based features
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['sin_day_of_week'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['cos_day_of_week'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    
    column = 'PriceHU'
    
    lags = [1, 3, 6, 12, 24, 48, 72, 168]
    data = add_lagged_features(data, column, lags)
    
    # Step 3: Add Rolling Window Features
    windows = [3, 6, 12, 24, 168]
    data = add_rolling_window_features(data, column, windows)
    
    # Step 4: Add Exponential Moving Average Features
    ema_windows = [12, 24, 168]
    data = add_exponential_moving_average(data, column, ema_windows)
    
    data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    data = data.dropna()
    
    # Mutual Information for Feature Selection
    X = data.drop(['PriceHU', 'Date'], axis=1)
    y = data['PriceHU']

    mi = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi, name="MI Scores", index=X.columns).sort_values(ascending=False)
    
    top_features = mi_scores[mi_scores > 0.3].index  # For example, keep features with MI score > 0.5
    X_filtered = X[top_features]
    top_features = top_features.tolist()
    
    columns_to_save = top_features + ['Date', 'PriceHU', 'WDAY']
else:
    # Convert the string argument to a list of provided manual regressors
    provided_features = eval(manual_regressors)

    # Drop all columns except the specified features, 'Date', and 'PriceHU'
    columns_to_keep = provided_features + ['Date', 'PriceHU']
    data = data[columns_to_keep]  # Filter the DataFrame to keep only the desired columns

    # Define the column for which to create features
    column = 'PriceHU'

    # Add Lagged Features
    lags = [1, 3, 6, 12, 24, 48, 72, 168]
    data = add_lagged_features(data, column, lags)

    # Add Rolling Window Features
    windows = [3, 6, 12, 24, 168]
    data = add_rolling_window_features(data, column, windows)

    # Add Exponential Moving Average Features
    ema_windows = [12, 24, 168]
    data = add_exponential_moving_average(data, column, ema_windows)

    data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    # Drop any rows with NaN values that were created by the feature engineering
    data = data.dropna()

    # Combine the manually provided features with the newly created lag, rolling, and EMA features
    # Get all column names after feature engineering
    engineered_features = list(set(data.columns) - set(['Date', 'PriceHU']))  # Exclude 'Date' and 'PriceHU'
    # Ensure we keep only the manually provided features and the newly created features
    columns_to_save = list(set(provided_features).intersection(engineered_features)) + ['Date', 'PriceHU']

    # Filter the DataFrame to include only the selected columns
    data = data[columns_to_save]
    
data = data[columns_to_save]

# Convert the 'ds' column to datetime format
df['ds'] = df['Date']
df['y'] = df['PriceHU']
df = df.drop(columns=['Date', 'PriceHU'])

print(f"Training dataset prepared with {len(data)} rows.")

# Initialize the NeuralProphet model with customized parameters
print("Initializing NeuralProphet model...")
model = NeuralProphet(
    growth="linear",
    n_changepoints=40,
    changepoints_range=0.95,
    trend_reg=0.1,
    trend_global_local="global",
    yearly_seasonality="auto",
    weekly_seasonality="auto",
    daily_seasonality="auto",
    seasonality_mode="additive",
    seasonality_reg=0.1,
    n_forecasts=504,
    learning_rate=0.005,
    epochs=1000,
    batch_size=1024,
    optimizer="AdamW",
    impute_missing=True,
    normalize="auto",
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

# Save the trained model to a pickle file
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_file}.")