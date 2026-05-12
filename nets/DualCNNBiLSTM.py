import torch
import torch.nn as nn


class ChargeBiLSTMBranch(nn.Module):
    """
    Charge branch adapted from the user's single_stream_bilstm design,
    but without temporal attention for a cleaner benchmark comparison.
    """

    def __init__(self, input_channels=4, hidden_size=128, num_layers=2, bottleneck_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_size * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        fused_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        feat = self.bottleneck(fused_hidden)
        return feat


class DischargeCNNBranch(nn.Module):
    """
    Discharge branch adapted from CNN1D_Discharge in the user's project.
    """

    def __init__(self, input_channels=4, out_dim=128):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        feat = self.conv_block(x).squeeze(-1)
        return feat


class DualCNNBiLSTM(nn.Module):
    """
    Dual-branch heterogeneous model for full charge/discharge input.
    Input shape: (N, 8, 128), where channels 0:4 are charge and 4:8 are discharge.
    """

    def __init__(self, input_channels=8, branch_channels=4, hidden_size=128, num_layers=2):
        super().__init__()
        if input_channels != branch_channels * 2:
            raise ValueError(
                f'DualCNNBiLSTM expects {branch_channels * 2} input channels, got {input_channels}.'
            )

        self.charge_branch = ChargeBiLSTMBranch(
            input_channels=branch_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bottleneck_dim=128,
        )
        self.discharge_branch = DischargeCNNBranch(
            input_channels=branch_channels,
            out_dim=128,
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        charge_x = x[:, :4, :]
        discharge_x = x[:, 4:, :]
        charge_feat = self.charge_branch(charge_x)
        discharge_feat = self.discharge_branch(discharge_x)
        fused_feat = torch.cat([charge_feat, discharge_feat], dim=1)
        pred = self.regressor(fused_feat)
        return pred


if __name__ == '__main__':
    x = torch.rand(30, 8, 128)
    net = DualCNNBiLSTM(input_channels=8)
    y = net(x)
    print(x.shape, y.shape)

    num_params = sum(param.numel() for param in net.parameters())
    print(num_params)
