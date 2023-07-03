import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


class ModelV1(AutoEncoder):
    def __init__(self, activation=nn.SELU(), seed=23):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.MaxPool2d(2),
            activation,
            nn.Conv2d(8, 8, kernel_size=3),
            nn.MaxPool2d(2, padding=1),
            activation,
            nn.Conv2d(8, 8, kernel_size=3),
            nn.MaxPool2d(2),
            activation,
            nn.Flatten(),
            nn.Linear(32, 3),
            activation,
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.Unflatten(1, (8, 2, 2)),
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=3),
            activation,
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),
            activation,
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),
            activation,
        )
