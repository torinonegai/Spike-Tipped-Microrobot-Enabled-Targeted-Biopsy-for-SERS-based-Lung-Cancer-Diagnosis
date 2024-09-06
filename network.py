import torch
from torch import nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=15),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # (b x 256 x 6 x 6)
            nn.Conv1d(8, 64, 15),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # (b x 256 x 6 x 6)
            nn.Conv1d(64, 64, 15),  # (b x 256 x 27 x 27)
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16128, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=5),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x