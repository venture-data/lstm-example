# tft_implementation.py

import torch
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss


def create_tft_model(train_dataloader, max_encoder_length, max_prediction_length, hidden_size=16, learning_rate=1e-3, dropout=0.1):
    """
    Function to create the Temporal Fusion Transformer (TFT) model.

    Args:
        train_dataloader (DataLoader): DataLoader for the training data.
        max_encoder_length (int): Maximum length of the historical input sequence.
        max_prediction_length (int): Maximum length of the forecast output sequence.
        hidden_size (int): Hidden size for the TFT model. Default is 16.
        learning_rate (float): Learning rate for training. Default is 1e-3.
        dropout (float): Dropout rate for regularization. Default is 0.1.

    Returns:
        model (TemporalFusionTransformer): Configured Temporal Fusion Transformer model.
    """

    # Define the Temporal Fusion Transformer model
    tft_model = TemporalFusionTransformer.from_dataset(
        train_dataloader.dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        lstm_layers=1,  # You can increase the number of LSTM layers as needed
        attention_head_size=4,  # Number of attention heads for the attention mechanism
        dropout=dropout,  # Dropout rate to prevent overfitting
        hidden_continuous_size=8,  # Size of hidden layer for continuous variables
        output_size=7,  # Output size for quantile predictions, adjust if needed
        loss=QuantileLoss(),  # Quantile loss function
        logging_metrics=None,  # Can add metrics like MAE, MSE
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )

    return tft_model


def prepare_dataloader(data, max_encoder_length, max_prediction_length, batch_size=64, time_varying_unknown_reals=None):
    """
    Function to prepare the dataloader for the TFT model.

    Args:
        data (DataFrame): Input data in the form of a pandas DataFrame.
        max_encoder_length (int): Maximum length of the historical input sequence.
        max_prediction_length (int): Maximum length of the forecast output sequence.
        batch_size (int): Batch size for the dataloader. Default is 64.
        time_varying_unknown_reals (list): List of unknown real features.

    Returns:
        train_dataloader (DataLoader): Dataloader for the training data.
    """
    # Create a time index column
    data = data.copy()
    data["time_idx"] = (data["Date"] - data["Date"].min()).dt.total_seconds() // 3600
    data["time_idx"] = data["time_idx"].astype(int)

    # Add a dummy group_id column for single time series
    data["group_id"] = 0  # All rows belong to group 0

    # Convert categorical features to strings or categorical types
    data["month"] = data["month"].astype(str)
    data["weekday"] = data["weekday"].astype(str)
    data["is_weekend"] = data["is_weekend"].astype(str)
    data["hour"] = data["hour"].astype(str)

    # Check if all columns in time_varying_unknown_reals exist in the DataFrame
    time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in data.columns]
    missing_cols = [col for col in time_varying_unknown_reals if col not in data.columns]
    if missing_cols:
        print(f"Warning: The following columns are missing and will be removed from time_varying_unknown_reals: {missing_cols}")

    # Specify which columns are categorical and which are real
    static_categoricals = []  # Update with any actual static categorical features
    static_reals = []  # Update with any actual static real features
    time_varying_known_categoricals = ["month", "weekday", "is_weekend", "hour"]  # Convert these to categorical
    time_varying_known_reals = []  # Update with known real features if any

    # Create TimeSeriesDataSet for training
    training = TimeSeriesDataSet(
        data,
        time_idx="time_idx",  # Now using the integer time index
        target="PriceSK",  # Column representing the target variable
        group_ids=["group_id"],  # Use the newly created group_id column
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,  # Correct static categorical features
        static_reals=static_reals,  # Correct static real features
        time_varying_known_categoricals=time_varying_known_categoricals,  # Correct known categorical features
        time_varying_known_reals=time_varying_known_reals,  # Correct known real features
        time_varying_unknown_categoricals=[],  # List of unknown categorical features (if any)
        time_varying_unknown_reals=time_varying_unknown_reals,  # Filtered dynamic unknown real features
        add_relative_time_idx=True,  # Adds a relative time index
        add_target_scales=True,  # Scale target variable
        add_encoder_length=True,  # Adds encoder length to dataset
    )

    # Create the dataloader
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)

    return train_dataloader




