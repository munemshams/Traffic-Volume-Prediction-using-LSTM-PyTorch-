import torch
import torch.nn as nn

class TrafficVolume(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=66,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.relu = nn.LeakyReLU()

        self.fc1 = nn.Linear(64, 1)

    def forward(self, x):

        _, (h_0, _) = self.lstm(x)

        out = h_0[-1]

        return self.relu(self.fc1(out))
