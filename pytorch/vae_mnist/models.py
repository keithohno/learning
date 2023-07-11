import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, seed=23):
        super().__init__()
        torch.manual_seed(seed)
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
        self.loss_fn = nn.BCELoss()

    def encode(self, x):
        x = self.encoder(x)
        mean = self.encoder_to_mean(x)
        std = self.encoder_to_std(x)
        return mean, std

    def forward(self, x):
        # encode
        mean, std = self.encode(x)

        # sample with reparameterization
        z = mean + std * torch.randn_like(std)

        # decode
        x_hat = self.decoder(z)

        return x_hat, mean, std

    def loss(self, x, x_hat, mean, std, beta=0.1):
        reconstruction_loss = self.loss_fn(x_hat, x)
        regularization_loss = torch.mean(0.5 * (mean**2 + std**2) - torch.log(std))
        return reconstruction_loss + beta * regularization_loss
