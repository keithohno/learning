import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from helpers import manual_seed
from common.plotting import plot_image_grid


def generate_mean_reconstructions(models, dataset, output_dir, seed=23):
    manual_seed(seed)
    device = next(models[0].parameters()).device

    DATA_SAMPLES = 8

    dataloader = DataLoader(dataset, batch_size=DATA_SAMPLES, shuffle=True)
    x, _ = next(iter(dataloader))
    x_grid = x.clone().unsqueeze(1)
    x = x.to(device)
    for i, model in enumerate(models):
        z, _ = model.encode(x)
        x_hat = model.decode(z).detach().unsqueeze(1)
        x_grid = torch.cat((x_grid, x_hat.cpu()), dim=1)
    x_grid = x_grid.squeeze()

    fig, axs = plot_image_grid(x_grid)
    for i, model in enumerate(models):
        axs[0, i + 1].set_title(model.id())
    axs[0, 0].set_title("Original")

    fig.savefig(f"{output_dir}/mean-reconstruction/{model.genus()}.png")
    plt.close()


def generate_sample_reconstructions(model, dataset, output_dir, seed=23):
    manual_seed(seed)
    device = next(model.parameters()).device

    SAMPLES = 8
    RECONSTRUCTIONS = 9

    dataloader = DataLoader(dataset, batch_size=SAMPLES, shuffle=True)
    x, _ = next(iter(dataloader))
    x_in = x.unsqueeze(1).repeat(1, RECONSTRUCTIONS, 1, 1, 1).reshape(-1, 1, 28, 28)
    x_hat, _, _ = model(x_in.to(device))
    x_hat = x_hat.reshape(SAMPLES, RECONSTRUCTIONS, 1, 28, 28)
    x_grid = torch.cat((x, x_hat.detach().squeeze().cpu()), dim=1)

    fig, axs = plot_image_grid(x_grid)
    axs[0, 0].set_title("Original")

    fig.savefig(f"{output_dir}/sample-reconstruction/{model.genus()}-{model.id()}.png")
    plt.close()


def generate_interpolation_reconstructions(model, dataset, output_dir, seed=23):
    manual_seed(seed)
    device = next(model.parameters()).device

    SAMPLES = 8
    INTERPOLATIONS = 8

    dataloader = DataLoader(dataset, batch_size=SAMPLES, shuffle=True)
    x_left, _ = next(iter(dataloader))
    x_right, _ = next(iter(dataloader))
    z_left, _ = model.encode(x_left.to(device))
    z_right, _ = model.encode(x_right.to(device))
    weights = torch.arange(0, 1, 1 / INTERPOLATIONS).to(device)
    z_grid = torch.lerp(
        z_left[:, None, :], z_right[:, None, :], weights[None, :, None]
    ).reshape(SAMPLES * INTERPOLATIONS, model.latent_dim())
    x_grid = model.decode(z_grid).detach().reshape(SAMPLES, INTERPOLATIONS, 28, 28)
    x_grid = torch.cat((x_left, x_grid.cpu(), x_right), dim=1)

    fig, _ = plot_image_grid(x_grid)
    fig.savefig(
        f"{output_dir}/interpolation-reconstruction/{model.genus()}-{model.id()}.png"
    )
    plt.close()


def generate_latent_space_constructions(model, output_dir, seed=23):
    manual_seed(seed)
    device = next(model.parameters()).device

    ROWS = 8
    COLS = 8

    samples = torch.randn(ROWS * COLS, model.latent_dim()).to(device)
    x_grid = model.decode(samples).detach().reshape(ROWS, COLS, 28, 28).cpu()

    fig, _ = plot_image_grid(x_grid)
    fig.savefig(f"{output_dir}/latent-construction/{model.genus()}-{model.id()}.png")
    plt.close()


def generate_latent_space_interpolations(model, output_dir, seed=23):
    manual_seed(seed)
    device = next(model.parameters()).device

    SAMPLES = 8
    INTERPOLATIONS = 8

    z_left = torch.randn(SAMPLES, model.latent_dim())
    z_right = torch.randn(SAMPLES, model.latent_dim())
    weights = torch.arange(0, 1, 1 / INTERPOLATIONS)
    z = (
        torch.lerp(z_left[:, None, :], z_right[:, None, :], weights[None, :, None])
        .reshape(SAMPLES * INTERPOLATIONS, model.latent_dim())
        .to(device)
    )
    x_grid = model.decode(z).detach().reshape(SAMPLES, INTERPOLATIONS, 28, 28).cpu()

    fig, _ = plot_image_grid(x_grid)
    fig.savefig(f"{output_dir}/latent-interpolation/{model.genus()}-{model.id()}.png")
    plt.close()


def plot_latent_space_parameters(models, dataset, output_dir, seed=23):
    manual_seed(seed)
    device = next(models[0].parameters()).device

    fig, axs = plt.subplots(
        len(models),
        2,
        figsize=(12, 3 * len(models)),
        sharex="col",
        sharey=True,
        squeeze=False,
    )

    dataloader = DataLoader(dataset, batch_size=1)
    with torch.inference_mode():
        for i, model in enumerate(models):
            model.eval()
            means = torch.tensor([]).to(device)
            stds = torch.tensor([]).to(device)
            for x, _ in dataloader:
                x = x.to(device)
                mean, std = model.encode(x)
                means = torch.cat((means, mean))
                stds = torch.cat((stds, std))
            means = means.view(-1).cpu()
            stds = stds.view(-1).cpu()
            axs[i, 0].hist(means, bins=100, range=(-3, 3))
            axs[i, 1].hist(stds, bins=100, range=(0.2, 1.2))
            axs[i, 0].set_yticklabels([])
            axs[i, 0].set_ylabel(model.id())
            axs[i, 1].get_yaxis().set_visible(False)

    axs[0, 0].set_title("means")
    axs[0, 1].set_title("stds")

    fig.savefig(f"{output_dir}/latent-shape/{model.genus()}.png")
    plt.close()


def plot_loss_history(models, loss_histories, output_dir):
    colors = pl.cm.viridis(torch.linspace(0, 1, len(models)))

    for i in range(len(loss_histories)):
        lmax = max(loss_histories[i])
        lmin = min(loss_histories[i])
        loss_histories[i] = [(l - lmin) / (lmax - lmin) for l in loss_histories[i]]

    plt.figure()
    for i, model in enumerate(models):
        plt.plot(loss_histories[i], label=model.id(), color=colors[i])
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("normalized loss")
    plt.savefig(f"{output_dir}/loss/{model.genus()}.png")
    plt.close()
