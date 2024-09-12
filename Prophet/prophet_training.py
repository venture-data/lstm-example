import sys
import os
import pickle
import numpy as np
import pandas as pd
from prophet import Prophet
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, 'prophet_model.pkl')
feature_csv_file = os.path.join(script_dir, 'features_for_Prophet.csv')

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

def create_fourier_features(data, period, order):
    t = np.arange(len(data))
    for i in range(1, order + 1):
        data[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
        data[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)

def create_interaction_features(df, columns, add=""):
    """
    Creates interaction features for the given DataFrame `df` using the specified `columns`.
    """
    print(f"\t{add}Creating interaction features...")
    interaction_features = pd.DataFrame(index=df.index)

    # Generate all possible combinations of the specified columns
    for col1, col2 in combinations(columns, 2):
        interaction_col_name = f"{col1}_x_{col2}"
        interaction_features[interaction_col_name] = df[col1] * df[col2]

    return interaction_features

# Read command-line arguments
input_file = sys.argv[1]
start_date = pd.to_datetime(sys.argv[2])  # Convert start_date to datetime
end_date = pd.to_datetime(sys.argv[3])

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

# min_date = pd.to_datetime('2017-01-01 00:00')
# cutoff_date = pd.to_datetime("2024-08-20 23:00")

# Validate the dates
if start_date < min_date or end_date > cutoff_date:
    print(f"Error: The provided date range [{start_date}, {end_date}] is out of bounds.")
    print(f"Start date cannot be less than {min_date} and end date cannot exceed {cutoff_date}.")
    sys.exit(1)

print(f"Loading dataset from {input_file}...")

# Determine the file extension
file_extension = os.path.splitext(input_file)[1].lower()

# Load the dataset based on the file extension
if file_extension == '.xlsx' or file_extension == '.xls':
    df = pd.read_excel(input_file)
elif file_extension == '.csv':
    df = pd.read_csv(input_file)
else:
    raise ValueError("Unsupported file type. Please provide an Excel (.xlsx, .xls) or CSV (.csv) file.")

print(f"Dataset loaded with {len(df)} rows.")

# print(f"Finding best feature based on entire, valid, dataset.")
# data = df[(df['Date'] >= min_date) & (df['Date'] <= cutoff_date)]
# data["Date"] = data["Date"].dt.round('h')
print(f"Dataset loaded with {len(df)} rows.")

# Ensure 'Date' is in datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')  # Adjust the format if necessary
df.loc[:, "Date"] = df["Date"].dt.round('h')

# Check for any conversion errors
if df['Date'].isna().any():
    print("Warning: Some date values could not be converted. Check the format of your 'Date' column.")

# Filter the DataFrame based on date range
data = df.loc[(df['Date'] >= min_date) & (df['Date'] <= cutoff_date)].copy()  # Use .copy() to ensure it's a new DataFrame
print(f"Filtered dataset has {len(data)} rows.")

# Filter the DataFrame using .loc and ensure it is done safely
print(f"Finding best feature based on training dates only.")
# data = df.loc[(df['Date'] >= min_date) & (df['Date'] <= cutoff_date)].copy()  # Use .copy() to ensure it's a new DataFrame

# Use .loc to modify the 'Date' column safely
data.loc[:, "Date"] = data["Date"].dt.round('h')

data_feature_engg = data.copy()
data_feature_engg['Date'] = data_feature_engg['Date'].dt.round('h')
data_feature_engg = data_feature_engg.loc[(data_feature_engg['Date'] >= min_date) & (data_feature_engg['Date'] <= end_date)].copy()

if is_automatic:
    print("Started Automatic feature selection")
    columns_to_drop = [
    # 'Y', 'M', 'Day', 'H',
    'Y2016',	'Y2017',	'Y2018',	'Y2019',	'Y2020',	'Y2021',	'Y2022',	'Y2023',	'Y2024',
    'M1',	'M2',	'M3',	'M4',	'M5',	'M6',	'M7',	'M8',	'M9',	'M10',	'M11',	'M12',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
    'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19',
    'h20', 'h21', 'h22', 'h23', 'h24',
    'PriceCZ', 'PriceSK', 'PriceRO'
    ]
    data_feature_engg = data_feature_engg.drop(columns=columns_to_drop)
    
    # Feature Engineering
    print("\tSimple Time based features generated")
    data_feature_engg['week_of_year'] = data_feature_engg['Date'].dt.isocalendar().week
    data_feature_engg['hour_of_week'] = data_feature_engg['Day'] * 24 + data_feature_engg['H']
    data_feature_engg['quarter'] = data_feature_engg['Date'].dt.quarter

    # Cyclical time-based features
    print("\tCyclical time-based features generated")
    data_feature_engg['sin_hour'] = np.sin(2 * np.pi * data_feature_engg['H'] / 24)
    data_feature_engg['cos_hour'] = np.cos(2 * np.pi * data_feature_engg['H'] / 24)
    data_feature_engg['sin_day_of_week'] = np.sin(2 * np.pi * data_feature_engg['Day'] / 7)
    data_feature_engg['cos_day_of_week'] = np.cos(2 * np.pi * data_feature_engg['Day'] / 7)
    
    print("\tFourier features generated")
    create_fourier_features(data_feature_engg, period=168, order=5)  # Weekly seasonality example
    
    # Polynomial Features
    print("\tPolynomial features generated")
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(data_feature_engg[['H', 'Day', 'WDAY']])
    poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['H', 'Day', 'WDAY']))
    data_feature_engg = pd.concat([data_feature_engg, poly_features_df], axis=1)
    
    target_column = 'PriceHU'
    
    print("\tAdded Lags")
    lags = [1, 3, 6, 12, 24, 48, 72, 168]
    data_feature_engg = add_lagged_features(data_feature_engg, target_column, lags)
    
    print("\tAdded rolling windows")
    windows = [3, 6, 12, 24, 168]
    data_feature_engg = add_rolling_window_features(data_feature_engg, target_column, windows)
    
    print("\tAdded ema windows")
    ema_windows = [12, 24, 168]
    data_feature_engg = add_exponential_moving_average(data_feature_engg, target_column, ema_windows)
    
    # data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    data_feature_engg = data_feature_engg.dropna()
    
    # Mutual Information for Feature Selection
    X = data_feature_engg.drop(['PriceHU', 'Date'], axis=1)
    y = data_feature_engg['PriceHU']

    # Assuming X and y are already defined
    print("\tStarted Mutual Information Score calculation. This will take a short while.")
    mi = mutual_info_regression(X, y)

    # Create a Series with MI scores
    mi_scores = pd.Series(mi, name="MI Scores", index=X.columns).sort_values(ascending=False)

    # Convert the Series to a DataFrame to include column names
    mi_scores_df = mi_scores.reset_index()
    mi_scores_df.columns = ['Column Name', 'MI Score']  # Rename columns for clarity

    # Save the MI scores to a CSV file
    mi_scores_df.to_csv("my_mi_scores.csv", index=False)
    print("MI scores saved to 'my_mi_scores.csv'")
    
    top_mi_features = mi_scores.head(10).index.tolist()     
    
    # Automate Interaction Feature Selection
    print("\tSelecting top columns for interaction features based on MI scores...")
    interaction_columns = [col for col in top_mi_features if col not in ['PriceHU', 'Date']]
    print(f"\tSelected interaction columns: {interaction_columns}")

    # Generate Interaction Features
    interaction_features = create_interaction_features(data_feature_engg, interaction_columns)
    data_feature_engg = pd.concat([data_feature_engg, interaction_features], axis=1)
    
    X = data_feature_engg.drop(['PriceHU', 'Date'], axis=1)
    y = data_feature_engg['PriceHU']
    mi = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi, name="MI Scores", index=X.columns).sort_values(ascending=False)
    
    top_features = mi_scores[mi_scores > 0.3].index  # For example, keep features with MI score > 0.5
    X_filtered = X[top_features]
    top_features = top_features.tolist()
    
    columns_to_save = top_features + ['Date', 'PriceHU']
    print(f"\tTop features selected: {top_features}")
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

    # data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    # Drop any rows with NaN values that were created by the feature engineering
    data = data.dropna()

    # Combine the manually provided features with the newly created lag, rolling, and EMA features
    # Get all column names after feature engineering
    engineered_features = list(set(data.columns) - set(['Date', 'PriceHU']))  # Exclude 'Date' and 'PriceHU'
    # Ensure we keep only the manually provided features and the newly created features
    columns_to_save = list(set(provided_features).intersection(engineered_features)) + ['Date', 'PriceHU']

print("Now for original data:")
# After Completing the Feature Selection the data will be cut off based on input end date
# Simple Time-Based Features
print("\t[O]\tSimple Time-based features generated")
data['week_of_year'] = data['Date'].dt.isocalendar().week
data['hour_of_week'] = data['Day'] * 24 + data['H']
data['quarter'] = data['Date'].dt.quarter

# Cyclical Time-Based Features
print("\t[O]\tCyclical time-based features generated")
data['sin_hour'] = np.sin(2 * np.pi * data['H'] / 24)
data['cos_hour'] = np.cos(2 * np.pi * data['H'] / 24)
data['sin_day_of_week'] = np.sin(2 * np.pi * data['Day'] / 7)
data['cos_day_of_week'] = np.cos(2 * np.pi * data['Day'] / 7)

# Fourier Features
print("\t[O]\tFourier features generated")
def create_fourier_features(data, period, order):
    t = np.arange(len(data))
    for i in range(1, order + 1):
        data[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
        data[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)

create_fourier_features(data, period=168, order=5)  # Weekly seasonality example

# Polynomial Features
print("\t[O]\tPolynomial features generated")
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(data[['H', 'Day', 'WDAY']])
poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['H', 'Day', 'WDAY']))
data = pd.concat([data, poly_features_df], axis=1)

target_column = 'PriceHU'

print("\tAdded Lags")
lags = [1, 3, 6, 12, 24, 48, 72, 168]
data = add_lagged_features(data, target_column, lags)

print("\tAdded rolling windows")
windows = [1, 3, 6, 12, 24, 168]
data = add_rolling_window_features(data, target_column, windows)

print("\tAdded ema windows")
ema_windows = [3, 6, 12, 24, 168]
data = add_exponential_moving_average(data, target_column, ema_windows)

# Generate Interaction Features
interaction_features = create_interaction_features(data, interaction_columns, add="[O]\t")
data = pd.concat([data, interaction_features], axis=1)

# Remove rows with missing values
# data = data.dropna()
data = data[columns_to_save]
data.to_csv(feature_csv_file, index=False)

# Filter data for the training range
print(f"Filtering Data for training from {start_date} till {end_date}")
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)
data['Date'] = pd.to_datetime(data['Date'])  # Ensure 'Date' is datetime
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Convert 'Date' to 'ds' and target column to 'y'
data['ds'] = data['Date']
data['y'] = data['PriceHU']
data = data.drop(columns=['Date', 'PriceHU'])

print(f"Training dataset prepared with {len(data)} rows.")

# Ensure all future regressor columns are numeric
for col in data.columns:
    if col not in ['ds', 'y']:
        # Convert the column to a Series explicitly
        column_series = data[col]
        if isinstance(column_series, pd.Series):
            data[col] = pd.to_numeric(column_series, errors='coerce')
            if data[col].isna().any():
                print(f"Warning: Column '{col}' has non-numeric values. Check data for issues.")
        else:
            print(f"Error: '{col}' is not a valid pandas Series for conversion.")

# Initialize the Prophet model
print("Initializing Prophet model...")
model = Prophet(
    changepoint_prior_scale=1.5,
    seasonality_prior_scale=15.0,
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_range=1
)

# Add future regressors to the model
print("Adding future regressors to the model...")
for col in data.columns:
    if col not in ['ds', 'y']:
        model.add_regressor(col)
print("All future regressors added.")

# Fit the Prophet model
print("Fitting the Prophet model...")
model.fit(data)
print("Model fitting completed.")

# Save the trained model to a pickle file
with open(model_file, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {model_file}.")