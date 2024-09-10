# Forecasting Project

This repository contains the implementation for time series forecasting using both **Prophet** and **NeuralProphet** models. The project aims to predict hourly prices _(PriceHU)_ based on a range of features, using data-driven modeling approaches.

## Project Overview

The data used in this project is hourly, and the goal is to predict `PriceHU` based on correctly chosen features. Due to the non-linear and unpredictable nature of these features, we can only forecast prices for the periods where we have data for those specific features (used as regressors). 

### Future Work

A potential future improvement could involve using known data like day, hour, year, month, and weekend/working days to predict values of the most correlated features first. Then, these predicted features could be used iteratively to predict other dependent features. Eventually, we would have a set of relevant features that could more effectively predict `PriceHU`.

**Note:** Using features such as `PriceCZ`, `PriceSK`, and `PriceRO` is explicitly excluded, as they represent similar prices in other regions, which would undermine the purpose of a true prediction model.

## Project Structure

```
main-repo/
│
├── Utility/
│   ├── evaluating_results.ipynb     # Notebook for evaluating forecasting results by visualizing them.
│   └── feature_extraction.ipynb     # Rough notebook used for feature extraction (integrated into the training scripts).
│
├── Prophet/
│   ├── prophet_training.py          # Script for training a model using Prophet.
│   └── prophet_forecasting.py       # Script for forecasting using a pre-trained Prophet model.
│
└── Neural Prophet/
    ├── nProphet_training.py         # Script for training a model using NeuralProphet.
    └── nProphet_forecasting.py      # Script for forecasting using a pre-trained NeuralProphet model.
```

## Model Initialization

### Prophet Initialization

```python
model = Prophet(
    changepoint_prior_scale=1.5,
    seasonality_prior_scale=15.0,
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_range=1
)
```

### NeuralProphet Initialization

```python
model = NeuralProphet(
    growth="linear",
    n_changepoints=40,
    changepoints_range=0.95,
    trend_reg=0.1,
    trend_global_local="global",
    yearly_seasonality="auto",
    weekly_seasonality="auto",
    daily_seasonality="auto",
    seasonality_mode="additive",
    seasonality_reg=0.1,
    n_forecasts=504,
    learning_rate=0.005,
    epochs=1000,
    batch_size=1024,
    optimizer="AdamW",
    impute_missing=True,
    normalize="auto",
)
```

## Results

In this successful iteration, we achieved the following results:

- **Mean Absolute Error (MAE):** 20.97
- **Mean Squared Error (MSE):** 962.82
- **Root Mean Squared Error (RMSE):** 31.03

### Dropped Features

During training (automatic mode) for both Prophet and NeuralProphet, the following columns were dropped to enhance model performance:

```python
columns_to_drop = [
    'Y', 'M', 'Day', 'H', 'Y2016',	'Y2017',	'Y2018',	'Y2019',	'Y2020',	'Y2021',	'Y2022',	'Y2023',	'Y2024',
    'M1',	'M2',	'M3',	'M4',	'M5',	'M6',	'M7',	'M8',	'M9',	'M10',	'M11',	'M12',
    'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10',
    'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18', 'h19',
    'h20', 'h21', 'h22', 'h23', 'h24',
    'PriceCZ', 'PriceSK', 'PriceRO'
]
```

## Getting Started

### Commands for Training

#### Automatic Mode

To train the model automatically with default settings:
```bash
python prophet_training.py “DATAFORMODELtrain200824.xlsx” "2017-01-08 13:00" "2024-06-30 23:00"
```

#### Manual Mode

To train the model manually by specifying a list of regressors:
```bash
python prophet_training.py “DATAFORMODELtrain200824.xlsx” "2017-01-08 13:00" "2024-06-30 23:00" "['WDAY']"
```

- **Note**: By adding a list of variables, the script will shift to manual training, meaning it will use only the specified regressors.

### Output from Training

1. **`features_for_Prophet.csv`**: Contains all the regressor information. This file is necessary to retain all the lag, rolling window, and EMA variables calculated during training and used as regressors.
2. **`prophet_model.pkl`**: The trained model file to be used for forecasting later.

### Command for Forecasting

To perform forecasting using the pre-trained model:
```bash
python prophet_forecasting.py features_for_Prophet.csv "2024-06-01 13:00" "2024-07-01 23:00"
```

- This will use the previously generated `features_for_Prophet.csv` as input, avoiding the need to recalculate all the values again.  
- The output will be saved as `prophet_forecast.csv`.

## Data and Results

All data, results, and model files are not included in the remote repository. Please refer to the `.gitignore` file for details on ignored file types:

- Excel files (`*.xlsx`)
- CSV files (`*.csv`)
- Pickle files (`*.pkl`)
- PyTorch model files (`*.pth`)

### Accessing Data and Models

You can access the data files and other resources via the Google Drive link:

[![Access Data on Google Drive](https://img.shields.io/badge/Google%20Drive-Access%20Data-blue)](https://drive.google.com/drive/folders/1HVvvnH4h5xnp4TEqqBkxwNLzjm4vM4t4?usp=sharing)

## Notebooks

Several rough, exploratory notebooks were used during the project for feature extraction and preliminary analysis. These notebooks are available on Google Colab:

- [Colab Notebook 1](https://colab.research.google.com/drive/1pF9GsS0VjW8r5y7iyG1R_4pj5aeJ8LLH?usp=sharing)
- [Colab Notebook 2](https://colab.research.google.com/drive/1WvPqGIYRe0NituZ20_995bA1H9XLi8Xi?usp=sharing)
- [Colab Notebook 3](https://colab.research.google.com/drive/1tSEO2wWLjq90e2lNF-sCsoURbjnmbJ1-?usp=sharing)

Feel free to explore these notebooks for additional insights and methodologies used in this project.


## Contributor(s)

**Ammar Ahmad**  
_Associate Data Scientist_
Email: [ammar.ahmad@venturedata.ai](mailto:ammar.ahmad@venturedata.ai)  
Phone: +92-316-2493305  
