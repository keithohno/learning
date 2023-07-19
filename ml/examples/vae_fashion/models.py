import torch
from torch import nn

from common.models import Model


class VAE(Model):
    def __init__(self, beta):
        super().__init__()
        self.encoder = None
        self.encoder_to_mean = None
        self.encoder_to_std = None
        self.decoder = None
        self.beta = beta
        self.loss_fn = nn.BCELoss()
        self.latent_dim = None

    def encode(self, x):
        x = self.encoder(x)
        mean = self.encoder_to_mean(x)
        std = self.encoder_to_std(x)
        return mean, std

    def sample(self, mean, std):
        z = mean + std * torch.randn_like(std)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.sample(*self.encode(x)))

    def loss(self, x_hat, x, mean, std):
        reconstruction_loss = self.loss_fn(x_hat, x)
        regularization_loss = torch.mean(0.5 * (mean**2 + std**2) - torch.log(std))
        return reconstruction_loss + self.beta * regularization_loss

    def tag(self):
        return str(self.beta)

    def id(self):
        return f"{self.__class__.__name__}-{self.tag()}"

    def compile(self):
        if (
            self.encoder
            and self.encoder_to_mean
            and self.encoder_to_std
            and self.decoder
            and self.latent_dim
        ):
            self.optimizer = torch.optim.Adam(self.parameters())
        else:
            raise NotImplementedError

    def train_batch(self, x):
        mean, std = self.encode(x)
        x_hat = self.decode(self.sample(mean, std))
        loss = self.loss(x_hat, x, mean, std)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def eval_batch(self, x):
        mean, std = self.encode(x)
        x_hat = self.decode(mean)  # no sampling
        loss = self.loss(x_hat, x, mean, std)
        return loss.item()


class VAEv1(VAE):
    def __init__(self, beta):
        super().__init__(beta)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encoder_to_mean = nn.Linear(144, 64)
        self.encoder_to_std = nn.Sequential(nn.Linear(144, 64), nn.Softplus())
        self.decoder = nn.Sequential(
            nn.Linear(64, 144),
            nn.Unflatten(-1, (16, 3, 3)),
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )
        self.latent_dim = 64


class VAEv2(VAE):
    def __init__(self, beta):
        super().__init__(beta)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encoder_to_mean = nn.Linear(144, 96)
        self.encoder_to_std = nn.Sequential(nn.Linear(144, 96), nn.Softplus())
        self.decoder = nn.Sequential(
            nn.Linear(96, 144),
            nn.Unflatten(-1, (16, 3, 3)),
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=1),
            nn.Sigmoid(),
        )
        self.latent_dim = 96
