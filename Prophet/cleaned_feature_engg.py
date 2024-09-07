# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# %%
# Load the actual data from the Excel file
actual_data = pd.read_excel("../../DATAFORMODELtrain200824.xlsx")

# Convert 'Date' to datetime format and round to the nearest hour
actual_data['Date'] = pd.to_datetime(actual_data['Date'], format='%m/%d/%Y %H:%M').dt.round('h')

# %%
# Define the date range
start_date = pd.to_datetime('2017-01-01 00:00')
cutoff_date = pd.to_datetime("2024-08-20 23:00")

# %%
# Filter data within the date range and drop unnecessary columns
columns_to_drop = [
    'Y', 'M', 'Day', 'H', 'Y2016',	'Y2017',	'Y2018',	'Y2019',	'Y2020',	'Y2021',	'Y2022',	'Y2023',	'Y2024',
    'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
    'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19',
    'h20', 'h21', 'h22', 'h23', 'h24',
    'PriceCZ', 'PriceSK', 'PriceRO', 'WDAY'
]
data = actual_data[(actual_data['Date'] >= start_date) & (actual_data['Date'] <= cutoff_date)].drop(columns=columns_to_drop)


# %%
def check_NaN(data):
    # Select only numeric columns for spline interpolation
    numeric_columns = data.select_dtypes(include=[np.number]).columns

    print("Number of NaN values:\n", data[numeric_columns].isna().sum())

# %%
check_NaN(data)

# %%
# Feature Engineering - Date Features
data['hour'] = data['Date'].dt.hour
data['day_of_week'] = data['Date'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
data['month'] = data['Date'].dt.month

# %%
check_NaN(data)

# %%
# Lagged Features
lagged_features = [1, 2, 3, 6, 12, 24]
for lag in lagged_features:
    data[f'lag_{lag}'] = data['PriceHU'].shift(lag)
print("Lagged features created.")

# %%
data = data.dropna()

# %%
check_NaN(data)

# %%
# Rolling Statistics (Keep only important ones)
rolling_windows = [3, 12, 24, 36, 48, 7 * 24]
for window in rolling_windows:
    data[f'rolling_mean_{window}h'] = data['PriceHU'].rolling(window=window).mean()
    data[f'rolling_std_{window}h'] = data['PriceHU'].rolling(window=window).std()


# %%
data = data.dropna()

# %%
check_NaN(data)

# %%
# Exponentially Weighted Moving Averages
ewm_windows = [3, 12, 24]
for span in ewm_windows:
    data[f'ewm_{span}h'] = data['PriceHU'].ewm(span=span).mean()

# %%
check_NaN(data)

# %%
# FFT Features (Keep only the top frequencies)
fft_vals = fft(data['PriceHU'].fillna(0).values)  # Handle NaN with fillna
fft_magnitude = np.abs(fft_vals)
N = 3  # Reduce the number of frequencies to keep to avoid feature explosion
top_frequencies = np.argsort(fft_magnitude)[-N:]

for i, freq_idx in enumerate(top_frequencies):
    data[f'fft_magnitude_{i+1}'] = fft_magnitude[freq_idx]
print("FFT features created.")

# %%
check_NaN(data)

# %%
# Define feature matrix X and target y
X = data.drop(['PriceHU', 'Date'], axis=1)
y = data['PriceHU']


# %%
# Feature Importance using RandomForest and Mutual Information
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# %%
mi_scores = mutual_info_regression(X, y)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
print("Mutual information scores calculated.")
mi_scores

# %%
check_NaN(data)

# %%
# Debugging: Check for duplicate labels
print("RF Importance Index Length:", len(rf_importances.index))
print("MI Scores Index Length:", len(mi_scores.index))
print("Unique RF Importance Index Length:", len(set(rf_importances.index)))
print("Unique MI Scores Index Length:", len(set(mi_scores.index)))

# %%
# Normalize and combine the scores
feature_scores = pd.DataFrame({'RF_Score': rf_importances, 'MI_Score': mi_scores})
feature_scores['CombinedScore'] = (feature_scores['RF_Score'] + feature_scores['MI_Score']).rank()
print("Feature scores combined.")

# %%
# Select top features - Reduce to top 10-15
top_features = feature_scores.sort_values('CombinedScore', ascending=False).head(10).index.tolist()

# %%
top_features

# %%
# Generate Polynomial Features (Degree 2) only for top selected features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_features = poly.fit_transform(data[top_features])
poly_feature_names = poly.get_feature_names_out(top_features)
poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

# %%
check_NaN(data)

# %%
check_NaN(poly_df)

# %%
# Ensure that the indices of poly_df match those of the original data
poly_df.index = data.index

# %%
# Identify columns in poly_df that do not exist in data
new_columns = [col for col in poly_df.columns if col not in data.columns]
new_columns

# %%
data_backup = data.copy()

# %%
# data = data_backup.copy()

# %%
# Only add the unique columns from poly_df to data
data = pd.concat([data, poly_df[new_columns]], axis=1)

# %%
check_NaN(data)

# %%
# Remove Multicollinear Features
corr_matrix = data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
highly_correlated = [column for column in upper.columns if any(upper[column] > 0.9)]
selected_features = list(set(data.columns) - set(highly_correlated) - {'PriceHU', 'Date'})

# %%
# Append 'PriceHU' and 'Date' to selected_features list
selected_features.extend(['PriceHU', 'Date'])

# %%
selected_features

# %%
check_NaN(data[selected_features])

# %%
# Drop rows with NaN values (or you could choose to impute them)
data_cleaned = data[selected_features].dropna()

# %%
check_NaN(data_cleaned)

# %%
# Separate the features for PCA and the columns to retain
features_for_pca = data_cleaned.drop(['Date', 'PriceHU'], axis=1)

# %%
# Standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(features_for_pca)
print("Data standardized.")

# %%
# Apply PCA to determine the number of components explaining 95% of the variance
pca_full = PCA().fit(data_standardized)
explained_variances = np.cumsum(pca_full.explained_variance_ratio_)

# Find the optimal number of components
best_n_components = np.argmax(explained_variances >= 0.85) + 1
print(f"Number of PCA components explaining at least 90% of variance: {best_n_components}")

# %%
# Apply PCA with the determined number of components
pca = PCA(n_components=best_n_components)
pca_components = pca.fit_transform(data_standardized)
print(f"PCA applied to reduce dimensionality to {pca.n_components} components.")

# %%
# Create a DataFrame for PCA components
pca_columns = [f'PCA_{i+1}' for i in range(pca_components.shape[1])]
pca_df = pd.DataFrame(pca_components, columns=pca_columns, index=data_cleaned.index)
print("PCA components DataFrame created.")

# %%
# # Ensure that indices are aligned before merging back
# data_cleaned = data_cleaned.loc[pca_df.index]  # Align indices

# %%
# data_cleaned.head()

# %%
# Align indices and concatenate only the new PCA columns
data_for_prophet = pd.concat([data_cleaned[['Date', 'PriceHU']], pca_df], axis=1)
data_for_prophet = data_for_prophet.rename(columns={'Date': 'ds', 'PriceHU': 'y'})
print("Data prepared for Prophet with PCA components as regressors.")

# %%
# Print Final Selected Features
print("Final PCA Components for Prophet Regressors:", pca_columns)

# %%
check_NaN(data_for_prophet)

# %%
# Extract 'is_weekend' and 'Date' from the original data
is_weekend_data = data[['Date', 'is_weekend']]

# Load the prepared data
prepared_data = data_for_prophet.copy()

# Ensure 'ds' in prepared data is in datetime format
prepared_data['ds'] = pd.to_datetime(prepared_data['ds'], errors='coerce')

# Merge 'is_weekend' with prepared data using the date columns
prepared_data = pd.merge(prepared_data, is_weekend_data, left_on='ds', right_on='Date', how='left')

# Drop the extra 'Date' column that was added during the merge
prepared_data = prepared_data.drop(columns=['Date'])

# %%
prepared_data.head()

# %%
check_NaN(prepared_data)

# %%
# Export Data for Prophet
prepared_data.to_csv('prepared_data_for_prophet.csv', index=False)
print("Data exported to 'prepared_data_for_prophet.csv'.")