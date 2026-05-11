import torch
import torch.nn as nn


class MLP(nn.Module):
    '''
    input shape: (N,C,128)
    '''

    def __init__(self, input_channels=4):
        super(MLP, self).__init__()
        self.input_channels = input_channels
        self.net = nn.Sequential(
            nn.Linear(128*input_channels,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256,128),
            nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        '''
        :param x: (N,C,128)
        :return:
        '''
        x = x.view(-1,self.input_channels*128)
        fea = self.net(x)
        out = self.predictor(fea)
        return out

if __name__ == '__main__':
    x = torch.rand(30,4,128)

    net = MLP(input_channels=4)
    y = net(x)
    print(x.shape,y.shape)

    num_params = sum(param.numel() for param in net.parameters())
    print(num_params)
