import sys
import os
import pickle
import numpy as np
import pandas as pd
from prophet import Prophet
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression

# Set file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(script_dir, 'prophet_model.pkl')
feature_csv_file = os.path.join(script_dir, 'features_for_Prophet.csv')

def add_lagged_features(data, columns, lags):
    """Add lagged features to the data for specified columns."""
    lagged_data = {}
    for column in columns:
        for lag in lags:
            lagged_data[f'{column}_lag_{lag}'] = data[column].shift(lag)
    lagged_df = pd.DataFrame(lagged_data)
    data = pd.concat([data, lagged_df], axis=1)
    return data

def add_rolling_window_features(data, columns, windows):
    """Add rolling window statistics to the data for specified columns."""
    rolling_data = {}
    for column in columns:
        for window in windows:
            rolling_mean = data[column].rolling(window).mean()
            rolling_std = data[column].rolling(window).std()
            rolling_data[f'{column}_rolling_mean_{window}h'] = rolling_mean
            rolling_data[f'{column}_rolling_std_{window}h'] = rolling_std
    rolling_df = pd.DataFrame(rolling_data)
    data = pd.concat([data, rolling_df], axis=1)
    return data

def add_exponential_moving_average(data, columns, ema_windows):
    """Add exponential moving averages to the data for specified columns."""
    ema_data = {}
    for column in columns:
        for window in ema_windows:
            ema = data[column].ewm(span=window).mean()
            ema_data[f'{column}_ema_{window}h'] = ema
    ema_df = pd.DataFrame(ema_data)
    data = pd.concat([data, ema_df], axis=1)
    return data

def create_fourier_features(data, period, order):
    """ Create Fourier series features for seasonality """
    t = np.arange(len(data))
    for i in range(1, order + 1):
        data[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
        data[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)

def create_interaction_features(df, columns, add=""):
    """ Creates interaction features between specified columns excluding lag, EMA, and rolling window features """
    print(f"\t{add}Creating interaction features...")
    # Exclude lag, EMA, and rolling window features
    exclude_columns = [col for col in df.columns if 'lag_' in col or 'rolling_' in col or 'ema_' in col]
    interaction_columns = [col for col in columns if col not in exclude_columns]

    interaction_features = pd.DataFrame(index=df.index)
    for col1, col2 in combinations(interaction_columns, 2):
        interaction_col_name = f"{col1}_x_{col2}"
        interaction_features[interaction_col_name] = df[col1] * df[col2]
    return interaction_features

def load_dataset(input_file):
    """ Load dataset based on file extension """
    print(f"Loading dataset from {input_file}...")
    file_extension = os.path.splitext(input_file)[1].lower()
    if file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    elif file_extension == '.csv':
        df = pd.read_csv(input_file)
    else:
        raise ValueError("Unsupported file type. Please provide an Excel (.xlsx, .xls) or CSV (.csv) file.")
    print(f"Dataset loaded with {len(df)} rows.")
    return df

def preprocess_data(df, min_date, cutoff_date):
    """ Preprocess data by converting 'Date' and filtering """
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date'] = df['Date'].dt.round('h')
    df = df.loc[(df['Date'] >= min_date) & (df['Date'] <= cutoff_date)].copy()
    print(f"Filtered dataset has {len(df)} rows.")
    return df

def apply_feature_engineering(data):
    """ Apply all feature engineering steps to the given data except lag, EMA, and rolling window features """
    print("\tGenerating time-based features...")
    data['week_of_year'] = data['Date'].dt.isocalendar().week
    data['hour_of_week'] = data['Day'] * 24 + data['H']
    data['quarter'] = data['Date'].dt.quarter
    data['sin_hour'] = np.sin(2 * np.pi * data['H'] / 24)
    data['cos_hour'] = np.cos(2 * np.pi * data['H'] / 24)
    data['sin_day_of_week'] = np.sin(2 * np.pi * data['Day'] / 7)
    data['cos_day_of_week'] = np.cos(2 * np.pi * data['Day'] / 7)

    create_fourier_features(data, period=168, order=5)
    print("\tFourier features generated.")
    
    return data

# Ensure all necessary columns are in the DataFrame before dropping them
def check_columns_to_drop(df, columns):
    """ Ensure that all columns to be dropped are present in the DataFrame """
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: The following columns are not in the DataFrame and will be skipped: {missing_columns}")
    return [col for col in columns if col in df.columns]

def main(input_file, start_date, end_date, is_automatic=True, manual_regressors=None):
    # Load and preprocess data
    df = load_dataset(input_file)
    min_date = pd.to_datetime('2017-01-01 00:00')
    cutoff_date = pd.to_datetime("2024-08-20 23:00")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')  # Adjust the format if necessary
    df.loc[:, "Date"] = df["Date"].dt.round('h')
    df = df.loc[(df['Date'] >= min_date) & (df['Date'] <= cutoff_date)].copy()
    
    
    historical_data = preprocess_data(df.copy(), min_date, end_date)

    # Ensure 'PriceHU' is present in historical_data
    if 'PriceHU' not in historical_data.columns:
        raise KeyError("'PriceHU' column is missing in the historical data. Please check your input.")
    
    if start_date < min_date or end_date > cutoff_date:
        raise ValueError(f"Date range [{start_date}, {end_date}] is out of bounds.")

    # Feature Engineering on historical data
    if is_automatic:
        print("Automatic feature selection started.")
        columns_to_drop = [ # Columns to drop
            'Y2016', 'Y2017', 'Y2018', 'Y2019', 'Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
            'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19',
            'h20', 'h21', 'h22', 'h23', 'h24', 'PriceCZ', 'PriceSK', 'PriceRO'
        ]
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])  # Drop only existing columns
        historical_data = historical_data.drop(columns=[col for col in columns_to_drop if col in historical_data.columns])

        # Calculate mutual information scores
        print("\tCalculating mutual information scores...")
        X = historical_data.drop(columns=['Date', 'PriceHU'])
        y = historical_data['PriceHU']
        mi = mutual_info_regression(X, y)
        mi_scores = pd.Series(mi, name="MI Scores", index=X.columns).sort_values(ascending=False)

        # Select top features (you can adjust the threshold or number of features)
        top_features = mi_scores.head(10).index.tolist()
        print(f"\tTop features selected: {top_features}")
        
        # Apply feature engineering to historical data
        # historical_data = apply_feature_engineering(historical_data)

        # Define the features for which to calculate lagged, rolling, and EMA features
        selected_features = ['GAS', 'COAL', 'PMIHU', 'CO2', 'UNAVGASRO', 'COALTOGAS', 'UNAVGASHU', 'UNAVTPPBG', 'UNAVGASALL']

        # Apply feature engineering to the entire dataset
        print("\tApplying feature engineering to selected features...")

        # Apply time-based features (if needed)
        df = apply_feature_engineering(df)
        # Add lagged features
        df = add_lagged_features(df, selected_features, lags=[1, 3, 6, 12, 24, 48, 72, 168])
        # Add rolling window features
        df = add_rolling_window_features(df, selected_features, windows=[3, 6, 12, 24, 168])
        # Add exponential moving averages
        df = add_exponential_moving_average(df, selected_features, ema_windows=[12, 24, 168])

        print("\tFeature engineering completed.")

        # # Remove any rows with NaN values after merging
        df = df.dropna()  

        # Columns to be excluded
        columns_to_exclude = ['PriceHU', 'Date']

        # Ensure columns exist before dropping
        columns_to_exclude = check_columns_to_drop(df, columns_to_exclude)

        # Drop the columns
        X = df.drop(columns=columns_to_exclude, axis=1)
        y = df['PriceHU']

        # Calculate mutual information
        mi = mutual_info_regression(X, y)
        mi_scores = pd.Series(mi, name="MI Scores", index=X.columns).sort_values(ascending=False)

        # Select features based on MI scores
        top_mi_features = mi_scores.head(10).index.tolist() 
        print("\tSelecting interaction columns based on MI scores...")
        interaction_columns = [col for col in top_mi_features if col not in ['PriceHU', 'Date']]
        print(f"\tSelected interaction columns: {interaction_columns}")

        interaction_features = create_interaction_features(df, interaction_columns)
        df = pd.concat([df, interaction_features], axis=1)

        # Select and save features
        top_features = mi_scores[mi_scores > 0.3].index.tolist()
        columns_to_save = top_features + ['Date', 'PriceHU']
        df.dropna()
        df = df[columns_to_save]
        df.to_csv(feature_csv_file, index=False)
        print(f"\tTop features selected and saved: {top_features}")
    else:
        # Manual Regressors Mode
        print("Manual feature selection started.")


    # Remove any rows with NaN values (due to lagging)
    df = df.dropna()

    # Prepare the final dataset for Prophet
    print(f"Preparing data for training from {start_date} to {end_date}...")
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    # Set up 'ds' and 'y' for Prophet
    df['ds'] = df['Date']
    df['y'] = df['PriceHU']

    # Drop unnecessary columns
    df = df.drop(columns=['Date', 'PriceHU'])

    # Convert all regressor columns to numeric
    for col in df.columns:
        if col not in ['ds', 'y']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                print(f"Warning: Column '{col}' has non-numeric values after conversion.")

    # Initialize and fit the Prophet model
    print("Initializing and fitting the Prophet model...")
    model = Prophet(
        changepoint_prior_scale=1.5,
        seasonality_prior_scale=15.0,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_range=1
    )

    # Add regressors to the model
    for col in df.columns:
        if col not in ['ds', 'y']:
            model.add_regressor(col)

    print("All future regressors added.")
    model.fit(df)
    print("Model fitting completed.")

    # Save the trained model
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_file}.")
    
if __name__ == "__main__":
    # Read command-line arguments
    input_file = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]

    if len(sys.argv) == 4:
        main(input_file, start_date, end_date, is_automatic=True)
    elif len(sys.argv) == 5:
        manual_regressors = sys.argv[4]
        main(input_file, start_date, end_date, is_automatic=False, manual_regressors=manual_regressors)
    else:
        print("Invalid number of arguments passed. Expected 3 or 4 arguments.")
        sys.exit(1)