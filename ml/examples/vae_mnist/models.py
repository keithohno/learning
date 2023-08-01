import torch
from torch import nn

from common.utils import manual_seed
from common.models import Model


class VAE(Model):
    def __init__(self, beta, seed=23):
        super().__init__()
        manual_seed(seed)
        self.encoder = nn.Identity()
        self.encoder_to_mean = nn.Identity()
        self.encoder_to_std = nn.Softplus()
        self.decoder = nn.Identity()
        self.loss_fn = nn.BCELoss()

        # weight assigned to regularization loss (KL divergence)
        self.beta = beta

    def encode(self, x):
        x = self.encoder(x)
        mean = self.encoder_to_mean(x)
        std = self.encoder_to_std(x)
        return mean, std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, std = self.encode(x)
        z = mean + std * torch.randn_like(std)
        x_hat = self.decode(z)

        return x_hat, mean, std

    def loss(self, x, x_hat, mean, std):
        reconstruction_loss = self.loss_fn(x_hat, x)
        regularization_loss = torch.mean(0.5 * (mean**2 + std**2) - torch.log(std))
        return reconstruction_loss + self.beta * regularization_loss

    def genus(self):
        raise NotImplementedError

    def species(self):
        raise NotImplementedError

    def id(self):
        return f"{self.genus()}-{self.species()}"

    def latent_dim(self):
        raise NotImplementedError


class VAEv1(VAE):
    def __init__(self, beta):
        super().__init__(beta)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
        )
        self.encoder_to_mean = nn.Linear(128, 16)
        self.encoder_to_std = nn.Sequential(nn.Linear(128, 16), nn.Softplus())
        self.decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Unflatten(-1, (1, 28, 28)),
            nn.Sigmoid(),
        )

    def genus(self):
        return "VAEv1"

    def species(self):
        return f"beta{self.beta}"

    def latent_dim(self):
        return 16


class VAEv2(VAE):
    def __init__(self, beta):
        super().__init__(beta)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encoder_to_mean = nn.Linear(72, 48)
        self.encoder_to_std = nn.Sequential(nn.Linear(72, 48), nn.Softplus())
        self.decoder = nn.Sequential(
            nn.Linear(48, 72),
            nn.Unflatten(-1, (8, 3, 3)),
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )

    def genus(self):
        return "VAEv2"

    def species(self):
        return f"beta{self.beta}"

    def latent_dim(self):
        return 48


class VAEv3(VAE):
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

    def genus(self):
        return "VAEv3"

    def species(self):
        return f"beta{self.beta}"

    def latent_dim(self):
        return 64


class VAEv4(VAE):
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

    def genus(self):
        return "VAEv4"

    def species(self):
        return f"beta{self.beta}"

    def latent_dim(self):
        return 96
