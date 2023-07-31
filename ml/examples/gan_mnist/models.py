import torch
from torch import nn

from common.models import Model
from helpers import manual_seed


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


class Discriminator(Model):
    def __init__(self, seed=23):
        super().__init__()
        manual_seed(seed)
        self.layers = nn.Sequential(
            # 1 x 28 x 28
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # 32 x 14 x 14
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # 64 x 7 x 7
            nn.Conv2d(64, 128, 4, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # 128 x 4 x 4
            nn.Conv2d(128, 1, 4, bias=False),
            # 1 x 1 x 1
            nn.Flatten(),
        )

    def forward(self, x):
        return self.layers(x)

    def compile(self, loss_fn, seed=23):
        manual_seed(seed)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.loss_fn = loss_fn
        self.is_compiled = True
        self.apply(init_weights)

    def train_batch(self, x, y) -> float:
        if not self.is_compiled:
            raise RuntimeError("Model must be compiled before training")

        y_hat = self(x).squeeze()
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
        self.noise_dim = 96
        self.layers = nn.Sequential(
            nn.Unflatten(-1, (self.noise_dim, 1, 1)),
            nn.ConvTranspose2d(self.noise_dim, 128, 4, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, 4, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # 64 x 7 x 7
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # 32 x 14 x 14
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            # 1 x 28 x 28
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)

    def compile(self, loss_fn, seed=23):
        manual_seed(seed)
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.loss_fn = loss_fn
        self.is_compiled = True
        self.apply(init_weights)

    def train_batch(self, z, y, discriminator) -> float:
        if not self.is_compiled:
            raise RuntimeError("Model must be compiled before training")

        x_hat = self(z)
        y_hat = discriminator(x_hat).squeeze()
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def id(self):
        return f"{self.__class__.__name__}"
