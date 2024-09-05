import sys
import pandas as pd
from fbprophet import Prophet
import pickle

# Command-line arguments
csv_file = sys.argv[1]  # CSV file name
start_date = sys.argv[2]  # Start date for training data
end_date = sys.argv[3]  # End date for training data

# Load data from CSV file
data = pd.read_csv(csv_file)

# Ensure 'Date' is in datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Filter data based on the provided date range
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Prepare data for Prophet
prophet_data = data.rename(columns={'Date': 'ds', 'PriceSK': 'y'})

# Identify regressor columns (all columns except 'ds' and 'y')
regressor_columns = [col for col in data.columns if col not in ['Date', 'PriceSK']]

# Initialize the Prophet model with custom parameters
model = Prophet(
    changepoint_prior_scale=0.1,  # Adjust this value to control trend flexibility
    seasonality_prior_scale=10.0  # Increase this value to make seasonality more flexible
)

# Add each regressor to the model
for regressor in regressor_columns:
    model.add_regressor(regressor)

# Fit the Prophet model
model.fit(prophet_data)

# Save the trained model to a file
with open('trained_prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained with custom parameters and saved as 'trained_prophet_model.pkl'.")