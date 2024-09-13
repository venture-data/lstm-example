import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# 1. Load the data
train_data = pd.read_csv('train_data.csv', parse_dates=['Date'], index_col='Date')
forecast_data = pd.read_csv('forecast_data.csv', parse_dates=['Date'], index_col='Date')

# 2. Prepare the data
y_train = train_data['PriceHU']
X_train = train_data.drop('PriceHU', axis=1)

X_forecast = forecast_data.drop('PriceHU', axis=1, errors='ignore')  # Use errors='ignore' in case 'PriceHU' isn't in the forecast_data

# 3. Ensure index is sorted
y_train = y_train.sort_index()
X_train = X_train.sort_index()
X_forecast = X_forecast.sort_index()

# 4. Fit the SARIMAX model
model = SARIMAX(endog=y_train, exog=X_train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
results = model.fit(disp=False)

# 5. Make predictions
start = X_forecast.index[0]
end = X_forecast.index[-1]
forecast = results.predict(start=start, end=end, exog=X_forecast)

# 6. Evaluate the model (Optional)
if 'PriceHU' in forecast_data.columns:
    y_actual = forecast_data['PriceHU']

    # Calculate Mean Absolute Error
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_actual, forecast)
    print(f'Mean Absolute Error: {mae}')

    # Plot the results
    plt.figure(figsize=(14,7))
    plt.plot(y_actual.index, y_actual, label='Actual PriceHU')
    plt.plot(forecast.index, forecast, label='Predicted PriceHU', alpha=0.7)
    plt.legend()
    plt.title('Actual vs Predicted PriceHU')
    plt.show()
else:
    # If actual values are not available
    plt.figure(figsize=(14,7))
    plt.plot(forecast.index, forecast, label='Predicted PriceHU')
    plt.legend()
    plt.title('Predicted PriceHU')
    plt.show()

# 7. Save the forecast
forecast_df = pd.DataFrame({'Date': forecast.index, 'Predicted_PriceHU': forecast.values})
forecast_df.to_csv('PriceHU_forecast.csv', index=False)
