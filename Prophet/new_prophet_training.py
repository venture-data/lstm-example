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

# Feature Engineering Functions
def add_lagged_features(data, column, lags):
    """ Add lagged features to the data """
    for lag in lags:
        data[f'lag_{lag}'] = data[column].shift(lag)
    return data

def add_rolling_window_features(data, column, windows):
    """ Add rolling window statistics to the data """
    for window in windows:
        data[f'rolling_mean_{window}h'] = data[column].rolling(window).mean()
        data[f'rolling_std_{window}h'] = data[column].rolling(window).std()
    return data

def add_exponential_moving_average(data, column, ema_windows):
    """ Add exponential moving averages to the data """
    for window in ema_windows:
        data[f'ema_{window}h'] = data[column].ewm(span=window).mean()
    return data

def create_fourier_features(data, period, order):
    """ Create Fourier series features for seasonality """
    t = np.arange(len(data))
    for i in range(1, order + 1):
        data[f'sin_{period}_{i}'] = np.sin(2 * np.pi * i * t / period)
        data[f'cos_{period}_{i}'] = np.cos(2 * np.pi * i * t / period)

def create_interaction_features(df, columns, add=""):
    """ Creates interaction features between specified columns """
    print(f"\t{add}Creating interaction features...")
    interaction_features = pd.DataFrame(index=df.index)
    for col1, col2 in combinations(columns, 2):
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
    """ Apply all feature engineering steps to the given data """
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
    
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(data[['H', 'Day', 'WDAY']])
    poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['H', 'Day', 'WDAY']))
    data = pd.concat([data, poly_features_df], axis=1)
    print("\tPolynomial features generated.")

    return data

def main(input_file, start_date, end_date, is_automatic=True, manual_regressors=None):
    # Load and preprocess data
    df = load_dataset(input_file)
    min_date = pd.to_datetime('2017-01-01 00:00')
    cutoff_date = pd.to_datetime("2024-08-20 23:00")
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')  # Adjust the format if necessary
    df.loc[:, "Date"] = df["Date"].dt.round('h')
    
    if start_date < min_date or end_date > cutoff_date:
        raise ValueError(f"Date range [{start_date}, {end_date}] is out of bounds.")

    # Separate historical data for feature engineering
    historical_data = preprocess_data(df.copy(), min_date, end_date)
    
    # Apply feature engineering to historical data
    historical_data = apply_feature_engineering(historical_data)

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
        df = df.drop(columns=columns_to_drop)
        historical_data = historical_data.drop(columns=columns_to_drop)

        # Mutual Information for feature selection
        print("\tStarting Mutual Information score calculation...")
        historical_data = historical_data.dropna()
        X = historical_data.drop(['PriceHU', 'Date'], axis=1)
        y = historical_data['PriceHU']
        mi = mutual_info_regression(X, y)
        mi_scores = pd.Series(mi, name="MI Scores", index=X.columns).sort_values(ascending=False)

        # Select features based on MI scores
        top_mi_features = mi_scores.head(10).index.tolist() 
        print("\tSelecting interaction columns based on MI scores...")
        interaction_columns = [col for col in top_mi_features if col not in ['PriceHU', 'Date']]
        print(f"\tSelected interaction columns: {interaction_columns}")

        interaction_features = create_interaction_features(historical_data, interaction_columns)
        historical_data = pd.concat([historical_data, interaction_features], axis=1)
        
        # Mutual Information for feature selection
        print("\tStarting Mutual Information score calculation again...")
        historical_data = historical_data.dropna()
        X = historical_data.drop(['PriceHU', 'Date'], axis=1)
        y = historical_data['PriceHU']
        mi = mutual_info_regression(X, y)
        mi_scores = pd.Series(mi, name="MI Scores", index=X.columns).sort_values(ascending=False)

        # Propagate the features to the original dataset
        print("\tApplying the same feature engineering to the entire original dataset...")
        df = apply_feature_engineering(df)
        
        # Merge interaction features from historical_data to df based on 'Date'
        print("\tMerging interaction columns with the original dataset based on 'Date'...")
        interaction_features = create_interaction_features(df, interaction_columns)
        df = pd.concat([df, interaction_features], axis=1)
        
        # Select and save features
        top_features = mi_scores[mi_scores > 0.3].index.tolist()
        columns_to_save = top_features + ['Date', 'PriceHU']
        df = df[columns_to_save]
        df.to_csv(feature_csv_file, index=False)
        print(f"\tTop features selected and saved: {top_features}")
    else:
        # Manual Regressors Mode
        print("Manual feature selection started.")
        provided_features = eval(manual_regressors)
        columns_to_keep = provided_features + ['Date', 'PriceHU']
        df = df[columns_to_keep].copy()

        df = add_lagged_features(df, 'PriceHU', [1, 3, 6, 12, 24, 48, 72, 168])
        df = add_rolling_window_features(df, 'PriceHU', [3, 6, 12, 24, 168])
        df = add_exponential_moving_average(df, 'PriceHU', [12, 24, 168])

        df = df.dropna()
        engineered_features = list(set(df.columns) - set(['Date', 'PriceHU']))
        columns_to_save = list(set(provided_features).intersection(engineered_features)) + ['Date', 'PriceHU']

        df[columns_to_save].to_csv(feature_csv_file, index=False)

    df = df.loc[:, ~df.columns.duplicated()].copy()
    # Prepare the final dataset for training
    print(f"Preparing data for training from {start_date} to {end_date}...")
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    df['ds'] = df['Date']
    df['y'] = df['PriceHU']
    df = df.drop(columns=['Date', 'PriceHU'])

    # Convert all future regressor columns to numeric
    for col in df.columns:
        if col not in ['ds', 'y']:
            # Convert the column to a Series explicitly
            column_series = df[col]
            if isinstance(column_series, pd.Series):
                df[col] = pd.to_numeric(column_series, errors='coerce')
                if df[col].isna().any():
                    print(f"Warning: Column '{col}' has non-numeric values. Check data for issues.")
            else:
                print(f"Error: '{col}' is not a valid pandas Series for conversion.")

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