import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

pl.seed_everything(42, workers=True)
import random

random.seed(42)
import numpy as np

np.random.seed(42)

import pandas as pd

pd.set_option("display.max_columns", None)
pd.options.mode.chained_assignment = None
import warnings


# Pytorch Modelling
class UnscaledMSELoss(torch.nn.Module):
    def __init__(self, target_scaler):
        super(UnscaledMSELoss, self).__init__()
        self.target_scaler = target_scaler
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Extract median and IQR as tensors
        median = torch.tensor(
            self.target_scaler.center_, dtype=torch.float32, device=y_pred.device
        )
        iqr = torch.tensor(
            self.target_scaler.scale_, dtype=torch.float32, device=y_pred.device
        )

        # Reverse the scaling on y_pred and y_true using median and IQR from RobustScaler
        y_pred_unscaled = y_pred * iqr + median
        y_true_unscaled = y_true * iqr + median

        # Compute the MSE loss on the unscaled values
        loss = self.mse_loss(y_pred_unscaled, y_true_unscaled)
        return loss


class SimpleLSTMModel(pl.LightningModule):
    def __init__(
        self,
        input_size,
        t_plus1_dim,
        hidden_size,
        hidden_size1,
        output_size,
        num_layers,
        learning_rate,
        target_scaler,
    ):
        super(SimpleLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size + t_plus1_dim, hidden_size1)
        self.fc1 = nn.Linear(hidden_size1, output_size)
        self.loss_fn = UnscaledMSELoss(target_scaler)

        (
            self.input_size,
            self.hidden_size,
            self.hidden_size1,
            self.output_size,
            self.num_layers,
        ) = (input_size, hidden_size, hidden_size1, output_size, num_layers)
        self.learning_rate = learning_rate
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer

    def forward(self, x, t_plus1_features):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        combined_input = torch.cat(
            (lstm_out, t_plus1_features), dim=1
        )  # combined_input: (batch_size, hidden_dim + t_plus1_dim)

        out = self.fc(combined_input)  # Use the output of the last time step
        out = self.dropout(out)
        out = self.fc1(out)
        return out

    def training_step(self, batch, batch_idx):
        x, t_plus1_features, y = batch
        y_hat = self(x, t_plus1_features)
        loss = self.loss_fn(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t_plus1_features, y = batch

        y_pred_scaled = self(x, t_plus1_features)
        loss = self.loss_fn(y_pred_scaled, y)

        # Optionally log scaled loss
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, t_plus1_features, y = batch
        y_hat = self(x, t_plus1_features)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Define the scheduler
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            ),
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

        y_pred = self(x)
        mse_loss = self.criterion(y_pred, y)
        rmse_loss = torch.sqrt(mse_loss)
        self.log("test_loss", rmse_loss, prog_bar=True)
        return rmse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
