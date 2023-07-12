import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, regularization_weight, seed=23):
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
        self.regularization_weight = regularization_weight

    def encode(self, x):
        x = self.encoder(x)
        mean = self.encoder_to_mean(x)
        std = self.encoder_to_std(x)
        return mean, std

    def forward(self, x):
        mean, std = self.encode(x)
        z = mean + std * torch.randn_like(std)
        x_hat = self.decoder(z)

        return x_hat, mean, std

    def loss(self, x, x_hat, mean, std):
        reconstruction_loss = self.loss_fn(x_hat, x)
        regularization_loss = torch.mean(0.5 * (mean**2 + std**2) - torch.log(std))
        return reconstruction_loss + self.regularization_weight * regularization_loss

    def id(self):
        return f"beta{self.regularization_weight}"

    def load_from_disk(self, model_dir):
        self.load_state_dict(torch.load(f"{model_dir}/model-{self.id()}.pt"))
