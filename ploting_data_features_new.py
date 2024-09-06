# -*- coding: utf-8 -*-
"""ploting_data_features.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tSEO2wWLjq90e2lNF-sCsoURbjnmbJ1-
"""

"""
'rolling_mean_3h',     
'lag_1',               
'lag_2',               
'ewm_12h',             
'rolling_mean_7d',     
'lag_24',              
'ewm_24h',             
'lag_3',               
'rolling_mean_24h',    
'GAS',                 
'rolling_mean_12h',    
'COAL',                
'UNAVGASRO',           
'PMIHU',               
'COALTOGAS',           
'CO2',                 
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/lstm/lstm-example

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.seasonal import STL
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import SelectKBest, mutual_info_regression
# from sklearn.preprocessing import PolynomialFeatures

# Load actual data from the Excel file
actual_data = pd.read_excel("../DATAFORMODELtrain200824.xlsx")

# Convert 'Date' to datetime format
actual_data['Date'] = pd.to_datetime(actual_data['Date'], format='%m/%d/%Y %H:%M')

"""# Feature Engineering Basic"""

actual_data.head()

data = actual_data.copy()

# Define the cutoff date
cutoff_date = pd.to_datetime("2024-08-20 23:00")
# Keep data from 2017 onwards
start_date = pd.to_datetime('2017-01-01 00:00')
data = data[data['Date'] >= start_date]

# Filter data to only include rows up to the cutoff date
data = data[data['Date'] <= cutoff_date]

columns_to_drop = [
    'Y', 'M', 'Day', 'H', 'Y2016',	'Y2017',	'Y2018',	'Y2019',	'Y2020',	'Y2021',	'Y2022',	'Y2023',	'Y2024',
    'M1',	'M2',	'M3',	'M4',	'M5',	'M6',	'M7',	'M8',	'M9',	'M10',	'M11',	'M12',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
    'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19',
    'h20', 'h21', 'h22', 'h23', 'h24',
    'PriceCZ', 'PriceSK', 'PriceRO',
    'hour', 'day_of_week', 'month', 'WDAY',
]

data = data.drop(columns=columns_to_drop)

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Get the correlations of all columns with 'PriceSK'
price_sk_correlation = correlation_matrix['PriceHU'].sort_values(ascending=False)

# Display the most relevant features that correlate with 'PriceSK'
print(price_sk_correlation)

# Feature Engineering
data['hour'] = data['Date'].dt.hour
data['day_of_week'] = data['Date'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
data['month'] = data['Date'].dt.month
data['rolling_mean_7d'] = data['PriceHU'].rolling(window=7).mean()  # Example of rolling mean
data['lag_1'] = data['PriceHU'].shift(1)  # Example of lag feature

# Mutual Information for Feature Selection
X = data.drop(['PriceHU', 'Date'], axis=1)
y = data['PriceHU']

# Ensure all columns are numeric or properly formatted for mutual_info_regression
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

mi = mutual_info_regression(X, y)
mi_scores = pd.Series(mi, name="MI Scores", index=X.columns).sort_values(ascending=False)

print(mi_scores)

"""# New Feature Engineering"""

# Retain top N features based on MI scores
top_features = mi_scores[mi_scores > 0.5].index  # For example, keep features with MI score > 0.5
X_filtered = X[top_features]

lasso = LassoCV(cv=10)
lasso.fit(X_filtered, y)

# Get the coefficients of the features
lasso_coefficients = pd.Series(lasso.coef_, index=X_filtered.columns).sort_values(ascending=False)

# Filter features based on non-zero coefficients
selected_features = lasso_coefficients[lasso_coefficients != 0].index
X_selected = X_filtered[selected_features]
print("Selected Features based on Lasso:", selected_features)

# Lagged Features
data['lag_2'] = data['PriceHU'].shift(2)
data['lag_3'] = data['PriceHU'].shift(3)
data['lag_6'] = data['PriceHU'].shift(6)
data['lag_12'] = data['PriceHU'].shift(12)
data['lag_24'] = data['PriceHU'].shift(24)

# Rolling Statistics
data['rolling_mean_3h'] = data['PriceHU'].rolling(window=3).mean()
data['rolling_mean_12h'] = data['PriceHU'].rolling(window=12).mean()
data['rolling_mean_24h'] = data['PriceHU'].rolling(window=24).mean()

data['rolling_std_3h'] = data['PriceHU'].rolling(window=3).std()
data['rolling_std_12h'] = data['PriceHU'].rolling(window=12).std()
data['rolling_std_24h'] = data['PriceHU'].rolling(window=24).std()

data['ewm_12h'] = data['PriceHU'].ewm(span=12).mean()  # Exponentially weighted moving average with span of 12 hours
data['ewm_24h'] = data['PriceHU'].ewm(span=24).mean()  # Exponentially weighted moving average with span of 24 hours

# Fourier Transform (FFT) Features
fft_vals = fft(data['PriceHU'].values)

# Adding FFT features (real and imaginary parts)
data['fft_real'] = np.real(fft_vals)
data['fft_imag'] = np.imag(fft_vals)

# Optional: Adding the magnitude of the FFT values to capture the strength of frequency components
data['fft_magnitude'] = np.abs(fft_vals)

# Initialize the model
rf = RandomForestRegressor()
rfe = RFE(estimator=rf, n_features_to_select=10)  # Adjust the number of features to keep
rfe.fit(X_filtered, y)

# Get the ranking of features
feature_ranking = pd.Series(rfe.ranking_, index=X_filtered.columns).sort_values()
selected_rfe_features = feature_ranking[feature_ranking == 1].index
print("Selected Features based on RFE:", selected_rfe_features)

# Calculate the correlation matrix
corr_matrix = X_filtered.corr()

# Identify highly correlated features
correlated_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.9:  # Set your threshold here
            colname = corr_matrix.columns[i]
            correlated_features.add(colname)

# Drop highly correlated features
X_uncorrelated = X_filtered.drop(columns=correlated_features)
print("Uncorrelated Features:", X_uncorrelated.columns)

# Calculate the correlation matrix
corr_matrix = X_filtered.corr()

# Identify moderately correlated features (0.3 < |correlation| < 0.8)
corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates()
potential_interactions = corr_pairs[(corr_pairs > 0.3) & (corr_pairs < 0.8)]
print("Potential Feature Interactions based on Correlation:\n", potential_interactions)

# Select only numeric columns for spline interpolation
numeric_columns = data.select_dtypes(include=[np.number]).columns

# Verify that there are no NaN values left in numeric columns
print("Number of NaN values after spline interpolation:\n", data[numeric_columns].isna().sum())

# Drop rows with any remaining NaN values
data = data.dropna()

# Check if all NaN values are handled
print("Number of NaN values after dropping:\n", data.isna().sum())

# Convert the 'Date' column to the nearest hour to ensure minutes and seconds are 00
data['Date'] = data['Date'].dt.round('H')

# Print the filtered data to verify
print("\nData from 2017 onwards:")

# Filter pairs to keep only those with moderate correlation (0.3 < |correlation| < 0.8)
selected_pairs = corr_pairs[(corr_pairs.abs() > 0.5) & (corr_pairs.abs() < 0.8)]

# Print the selected feature pairs
print("Selected Feature Pairs for Interaction based on Correlation and Domain Knowledge:")
for (feature_1, feature_2), corr_value in selected_pairs.items():
    # Ensure we print each pair only once
    if feature_1 != feature_2:
        print(f"{feature_1} and {feature_2} with Correlation: {corr_value:.3f}")

# Creating interaction terms
for (feature1, feature2) in selected_pairs.index:  # Use .index to get the feature pairs
    interaction_name = f"{feature1}_x_{feature2}"
    data[interaction_name] = data[feature1] * data[feature2]

print("Interaction terms added to the DataFrame.")

# Creating polynomial terms for selected features
polynomial_features = ['GAS', 'COAL', 'CO2', 'SOLMAX', 'lag_1', 'rolling_mean_7d']

# Generate polynomial features up to degree 2
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(data[polynomial_features])

# Create a DataFrame for polynomial features
poly_feature_names = poly.get_feature_names_out(polynomial_features)
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

# Add these polynomial features back to the main DataFrame
data = pd.concat([data, poly_df], axis=1)

print("Created Interaction and Polynomial Features:\n", data.head())

# Combining results from different feature selection methods
selected_features_lasso = set(selected_features)
selected_features_rfe = set(selected_rfe_features)
selected_features_mi = set(top_features)

# Find common features across multiple selection methods
final_selected_features = selected_features_lasso.intersection(selected_features_rfe, selected_features_mi)

print("Final Selected Features from Multiple Methods:", final_selected_features)

# Convert the set of final selected features to a list
final_selected_features = list(final_selected_features)

# Calculate the correlation matrix for the final selected features
corr_matrix_final = data[final_selected_features].corr().abs()

# Identify highly correlated features
upper = corr_matrix_final.where(np.triu(np.ones(corr_matrix_final.shape), k=1).astype(bool))
highly_correlated = [column for column in upper.columns if any(upper[column] > 0.9)]

# Drop highly correlated features
final_uncorrelated_features = list(set(final_selected_features).difference(set(highly_correlated)))
print("Final Uncorrelated Features for Prophet:", final_uncorrelated_features)

# Define the target and feature set, excluding the target variable 'PriceSK'
X_poly = data[poly_feature_names]  # This should be your polynomial features DataFrame
y = data['PriceSK']  # Target variable

# Check and handle any remaining NaN values
X_poly = X_poly.ffill().bfill().dropna()
y = y.ffill().bfill().dropna()

# Verify that there are no remaining NaN values
print("Number of NaN values in X_poly:", X_poly.isna().sum().sum())
print("Number of NaN values in y:", y.isna().sum().sum())

# Fit LassoCV model
lasso = LassoCV(cv=10, random_state=42)
lasso.fit(X_poly, y)

# Get selected polynomial features from Lasso
selected_poly_features = X_poly.columns[(lasso.coef_ != 0)]
print("Selected Polynomial Features from LASSO:", selected_poly_features)

# Calculate the correlation matrix for the selected polynomial features
corr_matrix_poly = data[selected_poly_features].corr().abs()

# Identify highly correlated features (correlation > 0.9)
upper_tri = corr_matrix_poly.where(np.triu(np.ones(corr_matrix_poly.shape), k=1).astype(bool))
highly_correlated_poly = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

# Drop highly correlated polynomial features
final_poly_features = set(selected_poly_features).difference(set(highly_correlated_poly))
print("Final Selected Polynomial Features:", final_poly_features)

# Assuming you have your original final selected features in 'final_uncorrelated_features'
combined_features = list(final_poly_features) + list(final_uncorrelated_features)

print("Combined Features for Prophet:", combined_features)

# Save the full DataFrame with all calculated columns
data.to_csv("raw_columns_for_training.csv", index=False)

print("All calculated columns saved to CSV successfully!")

data.head()

# Combine relevant columns including Date and PriceSK
relevant_columns = ['Date', 'PriceSK'] + combined_features

final_data = data[relevant_columns]

# Convert 'Date' column to datetime format
final_data['Date'] = pd.to_datetime(final_data['Date'], errors='coerce')

# Ensure the 'Date' column is formatted properly as a string with hours and minutes only
final_data.loc[:, 'Date'] = final_data['Date'].dt.strftime('%Y-%m-%d %H:%M')  # Format without seconds

final_data.head()

# Ensure the 'Date' column is formatted properly as a string with hours and minutes only

# Save the DataFrame with relevant columns to CSV
final_data.to_csv("relevant_columns_for_training.csv", index=False)

print("Relevant columns saved to CSV with correct date format successfully!")

"""# Previous Feature Engineering"""

# Interaction Features
data['PriceCZ_GAS'] = data['PriceCZ'] * data['GAS']
data['COAL_CO2'] = data['COAL'] * data['CO2']

# Polynomial Features for Selected Variables
poly = PolynomialFeatures(degree=2, include_bias=False)
selected_features = ['GAS', 'COAL', 'CO2', 'PriceCZ', 'PriceHU']
poly_features = poly.fit_transform(data[selected_features])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(selected_features))

# Merge Polynomial Features back to the DataFrame
data = pd.concat([data, poly_df], axis=1)

# Dimensionality Reduction using PCA for Less Significant Features
less_significant_features = ['UNAVTPPRO', 'UNAVGASALL', 'UNAVTPPBG', 'UNAVLIGNRO']
pca = PCA(n_components=2)
pca_features = pca.fit_transform(data[less_significant_features])
pca_df = pd.DataFrame(pca_features, columns=[f'PCA_{i+1}' for i in range(2)])
data = pd.concat([data, pca_df], axis=1)

# Feature Selection using LASSO
X = data.drop(['PriceSK', 'Date'], axis=1).fillna(0)
y = data['PriceSK']
lasso = LassoCV(cv=10, random_state=42)
lasso.fit(X, y)

# Get selected features from LASSO
selected_features = X.columns[(lasso.coef_ != 0)]
print("Selected Features from LASSO:", selected_features)

"""# Outliers"""

# Visualize potential outliers in the target variable
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['PriceSK'])
plt.title('PriceSK Over Time')
plt.xlabel('Date')
plt.ylabel('PriceSK')
plt.show()
print(f'total datapoints {len(data)}')

# Calculate Z-scores for 'PriceSK'
z_scores = np.abs(stats.zscore(data['PriceSK']))

# Identify the indices of outliers (Z-score threshold > 3)
outliers = np.where(z_scores > 3)[0]  # Extract the first element to get the indices array

# Remove outliers from the DataFrame
data = data.drop(data.index[outliers])

# Verify outliers are removed
print("Outliers removed. Remaining data points:", len(data))

# Visualize potential outliers in the target variable
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['PriceSK'])
plt.title('PriceSK Over Time')
plt.xlabel('Date')
plt.ylabel('PriceSK')
plt.show()

"""# Ploting New Features"""

# Filter out valid polynomial feature names that are 1-dimensional
valid_features = [feature for feature in poly_feature_names if data[feature].ndim == 1]

# Plot some of the new polynomial features against the target variable
plt.figure(figsize=(15, 10))

for i, feature in enumerate(valid_features[:9], 1):  # Limiting to the first 9 features for a 3x3 grid
    plt.subplot(3, 3, i)
    sns.scatterplot(x=data[feature], y=data['PriceSK'])
    plt.title(f'Scatter Plot of {feature} vs PriceSK')

plt.tight_layout()
plt.show()

"""# Columns to Plot"""

# Define the column names to plot
column1 = 'PriceHU'  # Replace with the actual column name
column2 = 'PriceHU'  # Replace with the actual column name
column3 = 'GAS SOLMAX'  # Replace with the actual column name
column4 = 'lag_1^2'  # Replace with the actual column name
column5 = 'UNAVTPPCZ'  # Replace with the actual column name
column6 = 'UNAVNUCFR'  # Replace with the actual column name

"""# date range plotting"""

# Define the date range for which you want to plot the data
start_date = '2018-01-01'  # Replace with the desired start date
end_date = '2020-02-01'  # Replace with the desired end date
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter data for the specific date range
data_range = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()  # Use .copy() to avoid SettingWithCopyWarning

# Handle missing values in PriceSK and other columns
data_range['PriceHU'] = data_range['PriceHU'].interpolate(method='linear')
data_range['PriceHU'] = data_range['PriceHU'].bfill().ffill()  # Use bfill() and ffill() directly

# Initialize a figure with subplots
fig, axes = plt.subplots(6, 1, figsize=(10, 18), sharex=True)  # 6 separate subplots

# Plotting function to create subplots with secondary y-axes
def create_plot(ax, y1, y2, y1_label, y2_label, color1, color2):
    ax.plot(data_range['Date'], data_range[y1], label=y1_label, color=color1, linestyle='-')
    ax.set_ylabel(y1_label, color=color1)
    ax2 = ax.twinx()  # Create a secondary y-axis
    ax2.plot(data_range['Date'], data_range[y2], label=y2_label, color=color2, linestyle=':')
    ax2.set_ylabel(y2_label, color=color2)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.set_title(f'{y1_label} with {y2_label}')

# Create plots for each subplot
create_plot(axes[0], 'PriceHU', column1, 'PriceHU', column1, 'blue', 'red')
create_plot(axes[1], 'PriceHU', column2, 'PriceHU', column2, 'blue', 'green')
create_plot(axes[2], 'PriceHU', column3, 'PriceHU', column3, 'blue', 'purple')
create_plot(axes[3], 'PriceHU', column4, 'PriceHU', column4, 'blue', 'orange')
create_plot(axes[4], 'PriceHU', column5, 'PriceHU', column5, 'blue', 'brown')
create_plot(axes[5], 'PriceHU', column6, 'PriceHU', column6, 'blue', 'cyan')

# Formatting and display
plt.xlabel('Date and Hour')
plt.tight_layout()
plt.show()

# Pairplot to visualize interactions
sns.pairplot(data[['PriceSK', 'CO2', 'COALTOGAS', 'UNAVGASALL', 'SOLMAX', 'UNAVNUCFR', 'UNAVTPPCZ']])
plt.show()



"""# Ploting PriceHU Only"""

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])
data_interpolated['Date'] = pd.to_datetime(data_interpolated['Date'])

# Define the date range for plotting
start_date = pd.to_datetime('2018-01-01')
end_date = pd.to_datetime('2018-02-01')

# Filter the data within the specified date range
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
filtered_data_interpolated = data_interpolated[(data_interpolated['Date'] >= start_date) & (data_interpolated['Date'] <= end_date)]

# Plotting the original and interpolated data
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['Date'], filtered_data['PriceHU'], label='Original Data')
plt.plot(filtered_data_interpolated['Date'], filtered_data_interpolated['PriceHU'], label='Interpolated Data', linestyle='--')

plt.xlabel('Date')
plt.ylabel('PriceHU')
plt.title('PriceHU Over Time')
plt.legend()
plt.show()

