import sys
import pandas as pd
import pickle
from neuralprophet import NeuralProphet
import numpy as np
from sklearn.feature_selection import mutual_info_regression
# import matplotlib.pyplot as plt

# Read command-line arguments
input_file = sys.argv[1]
start_date = sys.argv[2]
end_date = sys.argv[3]
manual_regressors = sys.argv[4]

is_automatic = True

print(f"Loading dataset from {input_file}...")
# Load the dataset
df = pd.read_excel(input_file)
print(f"Dataset loaded with {len(df)} rows.")

print(f"Filtering data for training from {start_date} to {end_date}...")
data = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

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
    
    # Step 2: Lagged Features
    lags = [1, 3, 6, 12, 24, 48, 72, 168]  # Example lags (adjust as needed)
    for lag in lags:
        data[f'lag_{lag}'] = data['PriceHU'].shift(lag)
    
    # Step 3: Rolling Window Features
    windows = [3, 6, 12, 24, 168]  # Example windows (adjust as needed)
    for window in windows:
        data[f'rolling_mean_{window}h'] = data['PriceHU'].rolling(window).mean()
        data[f'rolling_std_{window}h'] = data['PriceHU'].rolling(window).std()
    
    # Step 4: Exponential Moving Average
    ema_windows = [12, 24, 168]  # Example EMA windows (adjust as needed)
    for window in ema_windows:
        data[f'ema_{window}h'] = data['PriceHU'].ewm(span=window).mean()
    
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


    
data = data[columns_to_save]

# print(f"Loading dataset from {input_file}...")
# # Load the dataset
# df = pd.read_csv(input_file)

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
model_file = 'neuralprophet_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_file}.")