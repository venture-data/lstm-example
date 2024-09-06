import sys
import pandas as pd
from prophet import Prophet  # Updated import for Prophet
import pickle

# Command-line arguments
csv_file = sys.argv[1]  # CSV file name
start_date = sys.argv[2]  # Start date for training data
end_date = sys.argv[3]  # End date for training data

print(f"Received arguments:\nCSV File: {csv_file}\nStart Date: {start_date}\nEnd Date: {end_date}")

# Load data from CSV file
data = pd.read_csv(csv_file)
print(f"Data loaded from {csv_file}. Number of rows: {len(data)}, Number of columns: {len(data.columns)}")

# Ensure 'Date' is in datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
print("Converted 'Date' column to datetime format.")

# Check for NaT (Not a Time) values in the 'Date' column
num_nat = data['Date'].isna().sum()
print(f"Number of NaT values in 'Date' column after conversion: {num_nat}")

# Filter data based on the provided date range
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
print(f"Filtered data to the date range from {start_date} to {end_date}. Number of rows after filtering: {len(data)}")

# Prepare data for Prophet
prophet_data = data.rename(columns={'Date': 'ds', 'PriceHU': 'y'})
print(f"Prepared data for Prophet. First few rows:\n{prophet_data.head()}")

# Identify regressor columns (all columns except 'ds' and 'y')
regressor_columns = [col for col in data.columns if col not in ['Date', 'PriceHU']]
print(f"Regressor columns identified: {regressor_columns}")

# Initialize the Prophet model with custom parameters
model = Prophet(
    changepoint_prior_scale=0.1,  # Adjust this value to control trend flexibility
    seasonality_prior_scale=10.0  # Increase this value to make seasonality more flexible
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
model_file = 'trained_prophet_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained and saved as '{model_file}'.")