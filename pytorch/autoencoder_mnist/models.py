import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

        self.optimizer = None
        self.loss_fn = nn.MSELoss()

        self.name = f"{self.__class__.__name__}"
        self.spec = "default"

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


# First attempt at a convolutional auto-encoder
class ModelV1(AutoEncoder):
    def __init__(self, latent_dim, activation=nn.SELU(), seed=23):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = ModelV1.create_encoder(latent_dim, activation)
        self.decoder = ModelV1.create_decoder(latent_dim, activation)
        self.latent_dim = latent_dim
        self.activation = activation

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.name = "V1"
        self.spec = f"{self.latent_dim}-{self.activation.__class__.__name__}"

    @staticmethod
    def create_encoder(latent_dim, activation=nn.SELU()):
        return nn.Sequential(
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

    @staticmethod
    def create_decoder(latent_dim, activation=nn.SELU()):
        return nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Unflatten(-1, (8, 2, 2)),
            nn.ConvTranspose2d(8, 8, kernel_size=4, stride=3),
            activation,
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),
            activation,
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )


# Fully linear auto-encoder
class ModelV2(AutoEncoder):
    def __init__(self, latent_dim, activation=nn.SELU(), seed=23):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = ModelV2.create_encoder(latent_dim, activation)
        self.decoder = ModelV2.create_decoder(latent_dim, activation)
        self.latent_dim = latent_dim
        self.activation = activation

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.name = "V2"
        self.spec = f"{self.latent_dim}-{self.activation.__class__.__name__}"

    @staticmethod
    def create_encoder(latent_dim, activation=nn.SELU()):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            activation,
            nn.Linear(128, latent_dim),
            activation,
        )

    @staticmethod
    def create_decoder(latent_dim, activation=nn.SELU()):
        return nn.Sequential(
            nn.Linear(latent_dim, 128),
            activation,
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(-1, (1, 28, 28)),
        )


# Experimental model for comparing loss functions and optimizers
class ModelV3(AutoEncoder):
    def __init__(self, mse_loss=False, sgd_optim=False, seed=23):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.SELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 28 * 28),
            nn.SELU() if mse_loss else nn.Sigmoid(),
            nn.Unflatten(-1, (1, 28, 28)),
        )
        self.latent_dim = 32

        self.optimizer = (
            torch.optim.SGD(self.parameters(), lr=1e-1)
            if sgd_optim
            else torch.optim.Adam(self.parameters(), lr=1e-3)
        )
        self.loss_fn = nn.MSELoss() if mse_loss else nn.BCELoss()

        self.name = "V3"
        self.spec = (
            f"{self.loss_fn.__class__.__name__}-{self.optimizer.__class__.__name__}"
        )


# Hybrid model with convolutional encoder and linear decoder
class ModelV4(AutoEncoder):
    def __init__(self, latent_dim, seed=23):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = ModelV1.create_encoder(latent_dim)
        self.decoder = ModelV2.create_decoder(latent_dim)
        self.latent_dim = latent_dim

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.name = "V4"
        self.spec = f"{self.latent_dim}"


# Slightly larger convolutional model
class ModelV5(AutoEncoder):
    def __init__(self, latent_dim, seed=23):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.SELU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.MaxPool2d(2),
            nn.SELU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.SELU(),
            nn.Flatten(),
            nn.Linear(72, latent_dim),
            nn.SELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 72),
            nn.Unflatten(-1, (8, 3, 3)),
            nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),
            nn.SELU(),
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2),
            nn.SELU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2),
            nn.Sigmoid(),
        )
        self.latent_dim = latent_dim

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.name = "V5"
        self.spec = f"{self.latent_dim}"
