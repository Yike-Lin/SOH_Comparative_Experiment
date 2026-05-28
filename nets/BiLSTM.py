import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """
    Unified BiLSTM baseline for SOH regression.
    Expected input shape: (N, C, 128)
    """

    def __init__(self, input_channels=4, hidden_size=128, num_layers=2):
        super().__init__()
        self.input_channels = input_channels
        self.net = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        embed, _ = self.net(x)
        out = embed[:, -1, :]
        return self.predictor(out)

