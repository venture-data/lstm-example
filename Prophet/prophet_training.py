import sys
import pandas as pd
from prophet import Prophet
import pickle

# Command-line arguments
csv_file = sys.argv[1]  # CSV file name
start_date = pd.to_datetime(sys.argv[2])  # Convert Start date to datetime
end_date = pd.to_datetime(sys.argv[3])  # Convert End date to datetime

print(f"Received arguments:\nCSV File: {csv_file}\nStart Date: {start_date}\nEnd Date: {end_date}")

# Load data from CSV file
try:
    data = pd.read_csv(csv_file)
    print(f"Data loaded from {csv_file}. Number of rows: {len(data)}, Number of columns: {len(data.columns)}")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# Ensure 'ds' (date column) is in datetime format and round to the nearest hour
data['ds'] = pd.to_datetime(data['ds'], errors='coerce').dt.round('h')
print("Converted 'ds' column to datetime format and rounded to the nearest hour.")

# Check for NaT (Not a Time) values in the 'ds' column and remove them
num_nat = data['ds'].isna().sum()
if num_nat > 0:
    print(f"Number of NaT values in 'ds' column after conversion: {num_nat}. Removing these rows...")
    data = data.dropna(subset=['ds'])
    print(f"Removed rows with NaT values. Remaining rows: {len(data)}.")

# Filter data based on the provided date range
data = data[(data['ds'] >= start_date) & (data['ds'] <= end_date)]
print(f"Filtered data to the date range from {start_date} to {end_date}. Number of rows after filtering: {len(data)}")

# Ensure data is sorted by 'ds' in ascending order
data = data.sort_values(by='ds', ascending=True)
print("Data sorted by 'ds' in ascending order.")

# Prepare data for Prophet
prophet_data = data.copy()
print(f"Prepared data for Prophet. First few rows:\n{prophet_data.head()}")

# Identify regressor columns (all columns except 'ds' and 'y')
regressor_columns = [col for col in data.columns if col not in ['ds', 'y']]
print(f"Regressor columns identified: {regressor_columns}")

# Check for NaN values in the regressor columns
for regressor in regressor_columns:
    num_na = data[regressor].isna().sum()
    if num_na > 0:
        print(f"Regressor '{regressor}' has {num_na} missing values. Filling with 0 or consider another strategy.")
        data[regressor].fillna(0, inplace=True)  # You may want to use a different strategy, like interpolation.

# Initialize the Prophet model with custom parameters
model = Prophet(
    changepoint_prior_scale=0.3,  # Increased flexibility to better capture peaks and dips
    seasonality_prior_scale=15.0,  # Allow more complex seasonality patterns
    daily_seasonality=True,  # Enable daily seasonality explicitly
    weekly_seasonality=True,  # Enable weekly seasonality explicitly
    yearly_seasonality=True,  # Enable yearly seasonality explicitly
    seasonality_mode='multiplicative'
)
print("Initialized Prophet model with custom parameters.")

# Add each regressor to the model
for regressor in regressor_columns:
    model.add_regressor(regressor)
    print(f"Added regressor to the model: {regressor}")

# Fit the Prophet model
print("Fitting the Prophet model...")
model.fit(prophet_data)
print("Model fitting completed.")

# Save the trained model to a file
model_file = 'trained_prophet_model_PriceHU.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained and saved as '{model_file}'.")