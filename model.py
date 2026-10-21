"""
model.py - Fetal Distress Detection Model
==========================================
A 1D CNN-LSTM hybrid model for classifying CTG (Cardiotocography) recordings
as normal vs. distressed based on FHR and UC signals.

Architecture:
    - 1D CNN blocks extract local temporal patterns from the CTG signal
    - Bidirectional LSTM captures long-range temporal dependencies
    - Fully connected head outputs distress probability

Input:  (batch, 2, seq_len) — channel 0 = FHR, channel 1 = UC
Output: (batch, 1)          — sigmoid probability of fetal distress
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json


class CNNBlock(nn.Module):
    """Single 1D convolutional block with BatchNorm, ReLU, and Dropout."""

    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class FetalDistressModel(nn.Module):
    """
    CNN-LSTM model for fetal distress detection from CTG signals.

    Parameters
    ----------
    input_channels : int
        Number of input signal channels (default 2: FHR + UC).
    cnn_channels : list[int]
        Number of filters in each CNN block.
    lstm_hidden : int
        Hidden size for the bidirectional LSTM.
    lstm_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout probability used throughout.
    fc_hidden : int
        Hidden units in the fully-connected classifier head.
    """

    def __init__(
        self,
        input_channels: int = 2,
        cnn_channels: list = None,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        fc_hidden: int = 64,
    ):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        # Store config for serialisation
        self.config = {
            "input_channels": input_channels,
            "cnn_channels": cnn_channels,
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers,
            "dropout": dropout,
            "fc_hidden": fc_hidden,
        }

        # --- CNN feature extractor ---
        cnn_blocks = []
        in_ch = input_channels
        for out_ch in cnn_channels:
            cnn_blocks.append(CNNBlock(in_ch, out_ch, kernel_size=7, dropout=dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)

        # --- Bidirectional LSTM ---
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, fc_hidden),  # *2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, channels, seq_len). channels = 2 (FHR, UC).

        Returns
        -------
        torch.Tensor
            Shape (batch, 1) — raw logit (apply sigmoid for probability).
        """
        # CNN: (batch, channels, seq_len) -> (batch, cnn_out, reduced_len)
        features = self.cnn(x)

        # Transpose for LSTM: (batch, reduced_len, cnn_out)
        features = features.permute(0, 2, 1)

        # LSTM: (batch, reduced_len, lstm_hidden*2)
        lstm_out, _ = self.lstm(features)

        # Use the last time-step output
        last_hidden = lstm_out[:, -1, :]

        # Classifier
        logit = self.classifier(last_hidden)
        return logit

    def predict_proba(self, x):
        """Return distress probability in [0, 1]."""
        self.eval()
        with torch.no_grad():
            logit = self.forward(x)
            return torch.sigmoid(logit)

    def save(self, save_dir: str):
        """
        Save model weights and config to a directory.

        Creates:
            save_dir/model_weights.pth
            save_dir/model_config.json
        """
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "model_weights.pth"))
        with open(os.path.join(save_dir, "model_config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        print(f"Model saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str, device: str = "cpu"):
        """
        Load a saved model from a directory.

        Parameters
        ----------
        save_dir : str
            Path containing model_weights.pth and model_config.json.
        device : str
            Device to load the model onto.

        Returns
        -------
        FetalDistressModel
            The loaded model in eval mode.
        """
        with open(os.path.join(save_dir, "model_config.json"), "r") as f:
            config = json.load(f)

        model = cls(**config)
        state_dict = torch.load(
            os.path.join(save_dir, "model_weights.pth"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Model loaded from {save_dir}")
        return model


# ---------------------------------------------------------------------------
# Convenience: allow `python model.py` to print architecture summary
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = FetalDistressModel()
    print(model)
    # Quick shape test with a dummy input (batch=4, 2 channels, 4800 samples = 20 min @ 4 Hz)
    dummy = torch.randn(4, 2, 4800)
    out = model(dummy)
    print(f"\nInput shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    prob = torch.sigmoid(out)
    print(f"Sample probabilities: {prob.squeeze().tolist()}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
