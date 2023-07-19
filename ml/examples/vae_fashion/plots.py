import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from helpers import manual_seed
from common.plots import plot_image_grid


def generate_latent_lerp(model, output_dir, seed=23):
    manual_seed(seed)
    device = next(model.parameters()).device

    SAMPLES = 8
    INTERPOLATIONS = 10

    z_left = torch.randn(SAMPLES, model.latent_dim)
    z_right = torch.randn(SAMPLES, model.latent_dim)
    weights = torch.linspace(0, 1, INTERPOLATIONS)
    z_lerp = torch.lerp(
        z_left[:, None, :], z_right[:, None, :], weights[None, :, None]
    ).reshape(SAMPLES * INTERPOLATIONS, model.latent_dim)

    x_grid = (
        model.decode(z_lerp.to(device))
        .detach()
        .reshape(SAMPLES, INTERPOLATIONS, 28, 28)
        .cpu()
    )

    fig, _ = plot_image_grid(x_grid)
    fig.savefig(f"{output_dir}/lerp-latent/{model.id()}.png")
    plt.close(fig)


def generate_reconstruction_lerp(model, dataset, output_dir, seed=23):
    manual_seed(seed)
    device = next(model.parameters()).device

    SAMPLES = 8
    INTERPOLATIONS = 8

    dataloader = DataLoader(dataset, batch_size=SAMPLES, shuffle=True)
    x_left, _ = next(iter(dataloader))
    x_right, _ = next(iter(dataloader))
    z_left, _ = model.encode(x_left.to(device))
    z_right, _ = model.encode(x_right.to(device))
    weights = torch.linspace(0, 1, INTERPOLATIONS).to(device)
    z_lerp = torch.lerp(
        z_left[:, None, :], z_right[:, None, :], weights[None, :, None]
    ).reshape(SAMPLES * INTERPOLATIONS, model.latent_dim)

    x_grid = (
        model.decode(z_lerp.to(device))
        .reshape(SAMPLES, INTERPOLATIONS, 28, 28)
        .detach()
        .cpu()
    )
    x_grid = torch.cat([x_left, x_grid, x_right], dim=1)

    fig, _ = plot_image_grid(x_grid)
    fig.savefig(f"{output_dir}/lerp-reconstruction/{model.id()}.png")
    plt.close(fig)


def generate_latent_samples(model, output_dir, seed=23):
    manual_seed(seed)
    device = next(model.parameters()).device

    ROWS = 8
    COLS = 8

    z = torch.randn(ROWS * COLS, model.latent_dim).to(device)
    x_grid = model.decode(z).reshape(ROWS, COLS, 28, 28).detach().cpu()
    fig, _ = plot_image_grid(x_grid)
    fig.savefig(f"{output_dir}/sample-latent/{model.id()}.png")
    plt.close(fig)
