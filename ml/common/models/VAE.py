import torch
from torch import Tensor

from . import Model


class VAE(Model):
    """
    Base class for variational autoencoder models.

    Requires the following to be set:
        - self.nz: latent space dimension
        - self.beta: regularization weight
        - self.loss_fn: reconstruction loss function
        - self.encoder: encoder network
        - self.encoder_mu: mean network
        - self.encoder_logvar: log-variance network
        - self.decoder: decoder network
    """

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encodes x (image space) into mean and log-variance (latent space)."""
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Samples from mean and log-variance using the reparameterization trick."""
        return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

    def decode(self, z: Tensor) -> Tensor:
        """Decodes z (latent space) into x (image space)."""
        return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass. Returns x_hat, mean, and log-variance."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def forward_no_reparam(self, x: Tensor) -> Tensor:
        """Forward pass without sampling."""
        mu, _ = self.encode(x)
        return self.decode(mu)

    def reg_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Regularization loss (KL-divergence * beta)."""
        return self.beta * torch.mean((mu**2 + torch.exp(logvar)) - logvar - 1)

    def loss(self, x_hat: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        """Total loss (reconstruction + regularization)."""
        return self.loss_fn(x_hat, x) + self.reg_loss(mu, logvar)
