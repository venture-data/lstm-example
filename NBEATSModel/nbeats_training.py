import sys
import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
from sklearn.preprocessing import MinMaxScaler
import pickle

# Command-line arguments
csv_file = sys.argv[1]  # CSV file name
start_date = pd.to_datetime(sys.argv[2])  # Convert start date to datetime
end_date = pd.to_datetime(sys.argv[3])  # Convert end date to datetime

print(f"Received arguments:\nCSV File: {csv_file}\nStart Date: {start_date}\nEnd Date: {end_date}")

# Load your preprocessed data
data = pd.read_csv(csv_file)

# Ensure 'ds' is in datetime format
data['ds'] = pd.to_datetime(data['ds'])
print("Converted 'ds' column to datetime format.")

# Set the 'ds' column as the index
data.set_index('ds', inplace=True)

# Select the target column ('y') and the additional covariates (PCA components and 'is_weekend')
target_column = 'y'
covariates_columns = [col for col in data.columns if col.startswith('PCA_') or col == 'is_weekend']

print(f"Target column: {target_column}")
print(f"Covariates columns: {covariates_columns}")

# Handle missing values (if any)
data.fillna(method='ffill', inplace=True)  # Forward-fill missing values
print("Handled missing values.")

# Normalize the target and covariates data
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
print("Normalized the data.")

# Convert to Darts TimeSeries objects
series = TimeSeries.from_dataframe(data_scaled, value_cols=target_column)
covariates = TimeSeries.from_dataframe(data_scaled, value_cols=covariates_columns)

# Filter the data for the specified date range
series = series.drop_after(end_date)
covariates = covariates.drop_after(end_date)

# Split into train and validation sets
train, _ = series.split_after(pd.Timestamp(end_date))
covariates_train, _ = covariates.split_after(pd.Timestamp(end_date))

print(f"Training data and covariates prepared from {start_date} to {end_date}.")

# Define and train the NBEATS model
model = NBEATSModel(
    input_chunk_length=24,     # Number of past observations to use for predicting the future
    output_chunk_length=24,    # Number of future observations to predict
    generic_architecture=True, # Use generic architecture
    num_stacks=10,             # Number of stacks
    num_blocks=4,              # Number of blocks per stack
    num_layers=4,              # Number of fully connected layers per block
    layer_widths=512,          # Width of the fully connected layers
    n_epochs=100,              # Number of training epochs
    batch_size=32,             # Batch size
    random_state=42            # Random state for reproducibility
)

# Fit the model on training data with covariates
print("Training the NBEATS model...")
model.fit(series=train, past_covariates=covariates_train)
print("Model training completed.")

# Save the trained model to a file
model_file = 'trained_nbeats_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained and saved as '{model_file}'.")