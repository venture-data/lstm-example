import sys
import pandas as pd
from neuralprophet import NeuralProphet

def main(input_file, start_date, end_date, target_features):
    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    df = pd.read_excel(input_file)
    
    # Convert the 'Date' column to datetime format and round to the nearest hour
    df['ds'] = pd.to_datetime(df['Date'], format='%m/%d/%y %H:%M').dt.round('h')
    print(f"Converted 'Date' to 'ds' format. Dataset loaded with {len(df)} rows.")

    # Convert start_date and end_date to datetime format
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the dataset based on the provided training date range
    print(f"Filtering data for training from {start_date} to {end_date}...")
    df_filtered = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]
    print(f"Training dataset prepared with {len(df_filtered)} rows.")

    # Check if the filtered DataFrame is empty
    if df_filtered.empty:
        print("Error: The filtered DataFrame is empty. Please check the date range or the input data.")
        return

    # Loop through each target feature and train a separate model
    for target in target_features:
        print(f"\nTraining model for target: {target}...")
        
        # Initialize the NeuralProphet model with appropriate settings
        model = NeuralProphet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='additive',  # Use 'additive' or 'multiplicative' depending on the data
            seasonality_reg=0.1,  # Regularization for seasonality
            n_lags=0,  # No lagged values in this context
            learning_rate=0.01,  # Learning rate; adjust based on performance
            epochs=1000,  # Number of training epochs
            batch_size=1024,  # Adjust based on the size of your data and hardware
        )

        # Add temporal feature regressors
        model = model.add_future_regressor(name='WDAY')  # Use only WDAY as a regressor
        
        # Prepare the training data for the target
        df_target = df_filtered[['ds', target, 'WDAY']].dropna()
        df_target = df_target.rename(columns={target: 'y'})  # Rename the target column to 'y'

        # Check if the target DataFrame is empty
        if df_target.empty:
            print(f"Error: The target DataFrame for {target} is empty after filtering and dropping NaNs.")
            continue

        # Train the model for the specific target
        metrics = model.fit(df_target, freq='h')  # 'h' for hourly frequency
        print(f"Model training complete for {target}. Metrics: {metrics}")

        # Forecast the next 3 weeks (504 hours)
        print(f"Forecasting next 3 weeks for {target}...")
        future_dates = model.make_future_dataframe(df=df_target, regressors_df=df_target[['ds', 'WDAY']], periods=504)
        forecast = model.predict(future_dates)
        print(f"Forecasting completed for {target}.")

        # Save the forecast results to a CSV file
        forecast_output_file = f'{target}_forecast.csv'
        forecast.to_csv(forecast_output_file, index=False)
        print(f"Forecast results saved to {forecast_output_file}.")

if __name__ == "__main__":
    # Read command-line arguments
    input_file = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    target_features = eval(sys.argv[4])  # Convert string argument to list

    # Call the main function
    main(input_file, start_date, end_date, target_features)