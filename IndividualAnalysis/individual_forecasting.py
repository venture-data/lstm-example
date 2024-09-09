import sys
import pandas as pd
from neuralprophet import NeuralProphet
import pickle

def main(data_file, model_path, start_date):
    # Load the trained model
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")

    # Load the data file to extract WDAY values
    print(f"Loading data from {data_file} to extract WDAY values...")
    df = pd.read_excel(data_file)  # Adjust based on your file format, e.g., .csv or .excel
    df['ds'] = pd.to_datetime(df['Date'], format='%m/%d/%y %H:%M').dt.round('H')
    print(f"Converted 'Date' to 'ds' format. Data loaded with {len(df)} rows.")

    # Define the start date for forecasting and convert to datetime format
    forecast_start_date = pd.to_datetime(start_date)
    print(f"Forecasting from {forecast_start_date}...")

    # Define the forecast period (next 3 weeks, 24 hours * 7 days * 3 weeks = 504 hours)
    forecast_period = 504

    # Extract future dates from the loaded data to use WDAY values
    df_future = df[df['ds'] >= forecast_start_date].copy()
    df_future = df_future[['ds', 'WDAY']].iloc[:forecast_period]
    print(f"Future WDAY data extracted for {len(df_future)} rows.")

    # Make future dataframe with WDAY values
    future_dates = model.make_future_dataframe(df=df_future, regressors_df=df_future, periods=forecast_period, freq='H')
    print(f"Future dataframe created with {len(future_dates)} rows including regressors.")

    # Make predictions
    print("Making predictions...")
    forecast = model.predict(future_dates)
    print("Forecasting completed.")

    # Extract model name from the model path for naming the output file
    model_name = model_path.split('/')[-1].replace('_nProphet.pkl', '')
    forecast_output_file = f'{model_name}_forecast.csv'

    # Save the forecast to a CSV file
    forecast.to_csv(forecast_output_file, index=False)
    print(f"Forecast results saved to {forecast_output_file}.")

if __name__ == "__main__":
    # Read command-line arguments
    data_file = sys.argv[1]
    model_path = sys.argv[2]
    start_date = sys.argv[3]

    # Call the main function
    main(data_file, model_path, start_date)