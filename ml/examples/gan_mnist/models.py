import torch
from torch import nn

from common.models import Model
from helpers import manual_seed


class Discriminator(Model):
    def __init__(self, seed=23):
        super().__init__()
        manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(72, 1),
        )

    def forward(self, x):
        return self.layers(x)

    def compile(self, loss_fn):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.loss_fn = loss_fn
        self.is_compiled = True

    def train_batch(self, x, y):
        if not self.is_compiled:
            raise RuntimeError("Model must be compiled before training")

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def id(self):
        return f"{self.__class__.__name__}"


class Generator(Model):
    def __init__(self, seed=23):
        super().__init__()
        manual_seed(seed)
        self.layers = nn.Sequential(
            nn.Linear(16, 72),
            nn.Unflatten(-1, (8, 3, 3)),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.noise_dim = 16

    def forward(self, x):
        return self.layers(x)

    def compile(self, loss_fn):
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_fn = loss_fn
        self.is_compiled = True

    def train_batch(self, z, y, discriminator):
        if not self.is_compiled:
            raise RuntimeError("Model must be compiled before training")

        x_hat = self(z)
        y_hat = discriminator(x_hat)
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def generate_data_from_noise(self, size, seed=23):
        manual_seed(seed)
        device = next(self.parameters()).device
        noise = torch.randn(size, self.noise_dim)
        return self(noise.to(device))

    def id(self):
        return f"{self.__class__.__name__}"
