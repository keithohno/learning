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

    def name(self):
        return f"{self.__class__.__name__}"

    def spec(self):
        return "default"


class ModelV1(AutoEncoder):
    def __init__(self, latent_dim, activation=nn.SELU(), seed=23):
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
            nn.Linear(32, latent_dim),
            activation,
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Unflatten(-1, (8, 2, 2)),
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=3),
            activation,
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),
            activation,
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),
            activation,
        )
        self.latent_dim = latent_dim
        self.activation = activation

    def name(self):
        return "V1"

    def spec(self):
        return f"{self.latent_dim}-{self.activation.__class__.__name__}"


class ModelV2(AutoEncoder):
    def __init__(self, latent_dim, activation=nn.SELU(), seed=23):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            activation,
            nn.Linear(128, latent_dim),
            activation,
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            activation,
            nn.Linear(128, 28 * 28),
            activation,
            nn.Unflatten(-1, (1, 28, 28)),
        )
        self.latent_dim = latent_dim
        self.activation = activation

    def name(self):
        return "V2"

    def spec(self):
        return f"{self.latent_dim}-{self.activation.__class__.__name__}"
