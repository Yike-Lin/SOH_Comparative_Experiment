import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    '''
    input shape: (N,C,128)
    Adapted from the CNN-BiLSTM backbone in Yan et al. (2023),
    while keeping a single SOH regression head for this benchmark.
    '''

    def __init__(self, input_channels=4):
        super(CNNBiLSTM, self).__init__()
        self.input_channels = input_channels

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.temporal_model_1 = nn.LSTM(
            input_size=256,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_model_2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )

        self.predictor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        '''
        :param x: (N,C,128)
        :return:
        '''
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x, _ = self.temporal_model_1(x)
        x, _ = self.temporal_model_2(x)
        out = x[:, -1, :]
        pred = self.predictor(out)
        return pred


if __name__ == '__main__':
    x = torch.rand(30, 4, 128)

    net = CNNBiLSTM(input_channels=4)
    y = net(x)
    print(x.shape, y.shape)

    num_params = sum(param.numel() for param in net.parameters())
    print(num_params)
