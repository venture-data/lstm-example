import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau

pl.seed_everything(106, workers=True)

class UnscaledMSELoss(torch.nn.Module):
    def __init__(self, target_scaler):
        super(UnscaledMSELoss, self).__init__()
        self.target_scaler = target_scaler
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Extract median and IQR as tensors
        median = torch.tensor(self.target_scaler.center_, dtype=torch.float32, device=y_pred.device)
        iqr = torch.tensor(self.target_scaler.scale_, dtype=torch.float32, device=y_pred.device)

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
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size + t_plus1_dim, hidden_size1)
        self.fc1 = nn.Linear(hidden_size1, output_size)
        self.loss_fn = UnscaledMSELoss(target_scaler)

        self.learning_rate = learning_rate
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, t_plus1_features):
        lstm_out, _ = self.lstm(x)  # LSTM will handle hidden and cell states internally
        lstm_out = lstm_out[:, -1, :]  # Use output from the last time step
        combined_input = torch.cat((lstm_out, t_plus1_features), dim=1)

        out = self.fc(combined_input)
        out = self.dropout(out)
        out = self.fc1(out)
        return out

    def training_step(self, batch, batch_idx):
        x, t_plus1_features, y = batch
        y_hat = self(x, t_plus1_features)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t_plus1_features, y = batch
        y_pred_scaled = self(x, t_plus1_features)
        loss = self.loss_fn(y_pred_scaled, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, t_plus1_features, y = batch
        y_hat = self(x, t_plus1_features)
        loss = self.loss_fn(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

# Ensure GPU is available
if torch.cuda.is_available():
    print("GPU is available.")
else:
    print("GPU is not available. Training will fall back to CPU.")

# Load the best hyperparameters from the JSON file
with open("lstm_hyperparameters.json", "r") as f:
    best_params = json.load(f)

# Initialize the model with the best hyperparameters
model = SimpleLSTMModel(
    input_size=best_params["input_size"], 
    t_plus1_dim=best_params["t_plus1_dim"],
    hidden_size=best_params["hidden_size"], 
    hidden_size1=best_params["hidden_size1"], 
    output_size=best_params["output_size"], 
    num_layers=best_params["num_layers"], 
    learning_rate=best_params["learning_rate"], 
    target_scaler=your_target_scaler  # Replace with your actual scaler
)

# Configure the trainer to use GPU
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',  # Use 'gpu' accelerator
    devices=1,  # Number of GPUs to use; set to '1' or the number of GPUs available
)

# Start training
trainer.fit(model, train_loader, val_loader)
