import pandas as pd
import sys
from prophet import Prophet

def fill_missing_values(df, target, regressors):
    # Spline interpolation for target and regressors
    df[target].interpolate(method='spline', order=3, inplace=True)  # Adjust order if needed

    for regressor in regressors:
        df[regressor].interpolate(method='spline', order=3, inplace=True)

def main(input_file, start_date, end_date, regressors):
    # Load the data
    df = pd.read_excel(input_file)

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Filter data between start_date and end_date
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # Prepare the data for Prophet
    df = df.rename(columns={'date': 'ds', 'PriceSK': 'y'})  # Adjust target variable name
    target = 'y'

    # Convert the regressors string to a list
    regressor_list = regressors.strip('[]').split(',')

    # Fill missing values in the target and regressors using spline interpolation
    fill_missing_values(df, target, regressor_list)

    # Initialize the Prophet model
    model = Prophet()

    # Add additional regressors
    for regressor in regressor_list:
        model.add_regressor(regressor)

    # Fit the model
    model.fit(df)

    # Forecasting future values
    future = model.make_future_dataframe(periods=365, freq='H')

    # Add the regressors for the future data
    for regressor in regressor_list:
        future[regressor] = df[regressor].values[:len(future)]  # Adjust this part for proper regressor extension

    # Predict future values
    forecast = model.predict(future)

    # Output the forecast results
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('forecast_output.csv', index=False)
    print("Forecasting complete. Results saved to 'forecast_output.csv'.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python prophet_training.py <input_file> <start_date> <end_date> <regressors>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]
    regressors = sys.argv[4]

    main(input_file, start_date, end_date, regressors)
