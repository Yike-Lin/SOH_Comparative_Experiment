import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        attn_scores = self.attention(x)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * x, dim=1)
        return context


class DualBranchBiLSTMAttn(nn.Module):
    """
    Dual-branch BiLSTM with temporal attention on both charge and discharge streams.
    Input shape: (N, 8, 128), where channels 0:4 are charge and 4:8 are discharge.
    """

    def __init__(self, input_channels=8, branch_channels=4, hidden_size=128, num_layers=2):
        super().__init__()
        if input_channels != branch_channels * 2:
            raise ValueError(
                f'DualBranchBiLSTMAttn expects {branch_channels * 2} input channels, got {input_channels}.'
            )

        bi_hidden_dim = hidden_size * 2
        self.branch_channels = branch_channels

        self.lstm_charge = nn.LSTM(
            input_size=branch_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_charge = TemporalAttention(hidden_dim=bi_hidden_dim)

        self.lstm_discharge = nn.LSTM(
            input_size=branch_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.attn_discharge = TemporalAttention(hidden_dim=bi_hidden_dim)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 4, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        charge_x = x[:, :self.branch_channels, :].transpose(1, 2)
        discharge_x = x[:, self.branch_channels:, :].transpose(1, 2)

        out_charge, _ = self.lstm_charge(charge_x)
        feat_charge = self.attn_charge(out_charge)

        out_discharge, _ = self.lstm_discharge(discharge_x)
        feat_discharge = self.attn_discharge(out_discharge)

        fused_features = torch.cat((feat_charge, feat_discharge), dim=1)
        pred = self.regressor(fused_features)
        return pred


if __name__ == '__main__':
    x = torch.rand(30, 8, 128)
    net = DualBranchBiLSTMAttn(input_channels=8)
    y = net(x)
    print(x.shape, y.shape)

    num_params = sum(param.numel() for param in net.parameters())
    print(num_params)
