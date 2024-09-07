import sys
import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import numpy as np

# Command-line arguments
csv_file = sys.argv[1]  # CSV file name
train_start_date = pd.to_datetime(sys.argv[2])  # Convert start date to datetime
train_end_date = pd.to_datetime(sys.argv[3])  # Convert end date to datetime

print(f"Received arguments:\nCSV File: {csv_file}\nTrain Start Date: {train_start_date}\nTrain End Date: {train_end_date}")

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

# Check for missing values and raise an error if any are found
if data.isnull().values.any():
    raise ValueError("Data contains missing values. Please check your input data.")

print("No missing values detected in the data.")

# Normalize the target and covariates data
scaler = MinMaxScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns).astype(np.float32)
print("Normalized the data and converted to float32.")

# Convert to Darts TimeSeries objects
series = TimeSeries.from_dataframe(data_scaled, value_cols=target_column)
covariates = TimeSeries.from_dataframe(data_scaled, value_cols=covariates_columns)

# Split the data into training and test sets
train_series = series.drop_after(train_end_date)
train_covariates = covariates.drop_after(train_end_date)
test_series = series.drop_before(pd.Timestamp('2024-07-01 00:00'))
test_covariates = covariates.drop_before(pd.Timestamp('2024-07-01 00:00'))

print(f"Training data and covariates prepared from {train_start_date} to {train_end_date}.")
print(f"Test data and covariates prepared from '2024-07-01 00:00' to '2024-08-20 23:00'.")

# Define and train the NBEATS model
model = NBEATSModel(
    input_chunk_length=24 * 30,     # Number of past observations to use for predicting the future
    output_chunk_length=24 * 7 * 3,    # Number of future observations to predict
    generic_architecture=True, # Use generic architecture
    num_stacks=10,             # Number of stacks
    num_blocks=2,              # Number of blocks per stack
    num_layers=2,              # Number of fully connected layers per block
    layer_widths=256,          # Width of the fully connected layers
    n_epochs=12,               # Number of training epochs
    batch_size=32,             # Batch size
    random_state=42            # Random state for reproducibility
)

# Fit the model on training data with covariates
print("Training the NBEATS model...")
model.fit(series=train_series, past_covariates=train_covariates)
print("Model training completed.")

# Save the trained model to a file
model_file = '/Users/ammarahmad/Documents/Its IT Group/Fuel Price TimeSeries/lstm-example/NBEATSModel/trained_nbeats_model.pkl'
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained and saved as '{model_file}'.")

# Evaluate the model on the test set
print("Evaluating the model on the test set...")
predictions = model.predict(n=len(test_series), past_covariates=test_covariates)
predictions_df = predictions.pd_dataframe()
test_df = test_series.pd_dataframe()

# Inverse-transform the predictions to the original scale
predictions_df_original_scale = scaler.inverse_transform(predictions_df)
test_df_original_scale = scaler.inverse_transform(test_df)

# Calculate evaluation metrics
mae = mean_absolute_error(test_df_original_scale, predictions_df_original_scale)
mse = mean_squared_error(test_df_original_scale, predictions_df_original_scale)
rmse = np.sqrt(mse)

print(f"Evaluation Metrics:\nMAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}")

# Save evaluation metrics to a file
metrics_file = '/Users/ammarahmad/Documents/Its IT Group/Fuel Price TimeSeries/lstm-example/NBEATSModel/nbeats_model_metrics.txt'
with open(metrics_file, 'w') as f:
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")

print(f"Evaluation metrics saved to '{metrics_file}'.")