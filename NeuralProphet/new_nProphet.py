import sys
import pandas as pd
import warnings
from neuralprophet import NeuralProphet
import os
from sklearn.feature_selection import mutual_info_regression
import numpy as np

# Suppress all warnings
warnings.filterwarnings("ignore")

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the file name for the model and feature CSV
model_file = os.path.join(script_dir, 'neuralprophet_model.pkl')
feature_csv_file = os.path.join(script_dir, 'features_for_nProphet.csv')

def add_lagged_features(data, column, lags):
    for lag in lags:
        data[f'lag_{lag}'] = data[column].shift(lag)
    return data

def add_rolling_window_features(data, column, windows):
    for window in windows:
        data[f'rolling_mean_{window}h'] = data[column].rolling(window).mean()
        data[f'rolling_std_{window}h'] = data[column].rolling(window).std()
    return data

def add_exponential_moving_average(data, column, ema_windows):
    for window in ema_windows:
        data[f'ema_{window}h'] = data[column].ewm(span=window).mean()
    return data

# Read command-line arguments
input_file = sys.argv[1]
start_date = sys.argv[2]
end_date = sys.argv[3]

# Convert start_date and end_date to datetime
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

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
df = pd.read_csv(input_file)
print(f"Dataset loaded with {len(df)} rows.")
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
df['Date'] = df['Date'].dt.round('h')

print(f"Defining minimum and maximum dates data for training from {min_date} to {cutoff_date}...")
data = df[(df['Date'] >= min_date) & (df['Date'] <= cutoff_date)].copy()

if is_automatic:
    print("Started Automatic feature selection")
    columns_to_drop = [
        'Y', 'M', 'Day', 'H', 'Y2016', 'Y2017', 'Y2018', 'Y2019', 'Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024',
        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12',
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
        'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19',
        'h20', 'h21', 'h22', 'h23', 'h24',
        'PriceCZ', 'PriceSK', 'PriceRO'
    ]
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors='ignore')

    # Feature Engineering
    print("Simple Time-based features generated")
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

    # Target variable
    column = 'PriceHU'

    print("Adding Lags")
    lags = [1, 3, 6, 12, 24, 48, 72, 168]
    data = add_lagged_features(data, column, lags)

    print("Adding rolling windows")
    windows = [3, 6, 12, 24, 168]
    data = add_rolling_window_features(data, column, windows)

    print("Adding EMA windows")
    ema_windows = [12, 24, 168]
    data = add_exponential_moving_average(data, column, ema_windows)

    # Drop any rows with NaN values
    data = data.dropna()

    # Mutual Information for Feature Selection
    X = data.drop(['PriceHU', 'Date'], axis=1)
    y = data['PriceHU']

    print("Calculating Mutual Information Scores...")
    mi = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi, name="MI Scores", index=X.columns).sort_values(ascending=False)

    # Exclude lagged features of 'PriceHU'
    top_features = [feature for feature in mi_scores[mi_scores > 0.3].index
                    if not (feature.startswith('lag_') or feature.startswith('rolling_') or feature.startswith('ema_'))]

    columns_to_save = top_features + ['Date', 'PriceHU']
else:
    # Use provided manual regressors
    provided_features = eval(manual_regressors)
    columns_to_save = provided_features + ['Date', 'PriceHU']

data = data[columns_to_save]
data.to_csv(feature_csv_file, index=False)

# Filter data between start_date and end_date
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()

# Prepare data for model training
data['ds'] = data['Date']
data['y'] = data['PriceHU']
data = data.drop(columns=['Date', 'PriceHU'])
data['ds'] = data['ds'].dt.round('h')

print(f"Training dataset prepared with {len(data)} rows.")

# Initialize the NeuralProphet model with customized parameters
print("Initializing NeuralProphet model...")
model = NeuralProphet(
    growth="linear",
    n_changepoints=40,
    changepoints_range=0.95,
    trend_reg=0.1,
    trend_global_local="local",
    global_normalization=False,
    yearly_seasonality="auto",
    weekly_seasonality="auto",
    daily_seasonality="auto",
    seasonality_mode="additive",
    seasonality_reg=0.1,
    learning_rate=0.005,
    optimizer="AdamW",
    impute_missing=True,
    normalize="auto",
)

# Add future regressors (excluding lagged features of 'PriceHU')
print("Adding future regressors to the model...")
for col in data.columns:
    if col not in ['ds', 'y'] and not (col.startswith('lag_') or col.startswith('rolling_') or col.startswith('ema_')):
        model = model.add_future_regressor(name=col)

print("Future regressors added.")

# Train the model
print("Training the model...")
model.fit(data, freq='h')
print("Model training complete.")

# Load the feature CSV file that includes future regressors
print("Loading feature CSV file for forecasting...")
forecast_data = pd.read_csv(feature_csv_file)

# Ensure 'Date' column is in datetime format
forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
forecast_data['ds'] = forecast_data['Date']
forecast_data = forecast_data.drop(columns=['Date'])

# Exclude 'PriceHU' or 'y' if present
forecast_data = forecast_data.drop(columns=['PriceHU', 'y'], errors='ignore')

# Ensure all columns are numeric
for col in forecast_data.columns:
    if col != 'ds':
        forecast_data[col] = pd.to_numeric(forecast_data[col], errors='coerce')

# Handle any missing values
forecast_data = forecast_data.fillna(method='ffill').fillna(method='bfill')

# Filter the data for the forecasting period
forecast_data = forecast_data[(forecast_data['ds'] > end_date) & (forecast_data['ds'] <= cutoff_date)].copy()
print(f"Forecasting dataset prepared with {len(forecast_data)} rows.")

# Ensure all necessary regressors are present
required_regressors = [reg['name'] for reg in model.config_covar]
missing_regressors = [reg for reg in required_regressors if reg not in forecast_data.columns]
if missing_regressors:
    print(f"Error: The following regressors are missing in forecast data: {missing_regressors}")
    sys.exit(1)

# Make predictions
print("Making forecasts...")
forecast = model.predict(forecast_data)
print("Predictions completed.")

# Save the forecast to a CSV file
forecast_output_file = 'nProphet_forecast.csv'
forecast.to_csv(forecast_output_file, index=False)
print(f"Forecast results saved to {forecast_output_file}.")

print("Forecasting process completed successfully.")