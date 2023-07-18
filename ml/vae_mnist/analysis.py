import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from helpers import manual_seed


def generate_mean_reconstructions(models, dataset, output_dir, seed=23):
    manual_seed(seed)
    device = next(models[0].parameters()).device

    DATA_SAMPLES = 8
    fig, axs = plt.subplots(
        DATA_SAMPLES, len(models) + 1, figsize=(2 + len(models) * 2, 16)
    )

    dataloader = DataLoader(dataset, batch_size=DATA_SAMPLES, shuffle=True)
    x, _ = next(iter(dataloader))
    x = x.to(device)

    for i, model in enumerate(models):
        z, _ = model.encode(x)
        x_hat = model.decode(z).detach().squeeze()
        for j in range(DATA_SAMPLES):
            axs[j, i + 1].imshow(x_hat[j].cpu(), cmap="gray")
            axs[j, i + 1].axis("off")
        axs[0, i + 1].set_title(model.id())

    for j in range(DATA_SAMPLES):
        axs[j, 0].imshow(x[j].squeeze().cpu(), cmap="gray")
        axs[j, 0].axis("off")

    axs[0, 0].set_title("Original")

    fig.savefig(f"{output_dir}/mean-reconstruction/{model.genus()}.png")
    plt.close()


def generate_sample_reconstructions(model, dataset, output_dir, seed=23):
    manual_seed(seed)
    device = next(model.parameters()).device

    ROWS = 8
    COLS = 9
    fig, axs = plt.subplots(ROWS, COLS, figsize=(1 + 2 * COLS, 2 * ROWS))

    dataloader = DataLoader(dataset, batch_size=ROWS, shuffle=True)
    x, _ = next(iter(dataloader))
    x = x.to(device)
    mean, std = model.encode(x)

    for i in range(COLS - 1):
        samples = mean + std * torch.randn_like(std)
        x_hat = model.decode(samples).detach().squeeze()
        for j in range(ROWS):
            axs[j, i + 1].imshow(x_hat[j].cpu(), cmap="gray")
            axs[j, i + 1].axis("off")
    for j in range(ROWS):
        axs[j, 0].imshow(x[j].squeeze().cpu(), cmap="gray")
        axs[j, 0].axis("off")
    axs[0, 0].set_title("Original")

    fig.savefig(f"{output_dir}/sample-reconstruction/{model.genus()}-{model.id()}.png")
    plt.close()


def generate_interpolation_reconstructions(model, dataset, output_dir, seed=23):
    manual_seed(seed)
    device = next(model.parameters()).device

    SAMPLES = 8
    INTERPOLATIONS = 8
    fig, axs = plt.subplots(
        SAMPLES,
        INTERPOLATIONS + 2,
        figsize=(
            2 * INTERPOLATIONS + 4,
            2 * SAMPLES,
        ),
    )

    dataloader = DataLoader(dataset, batch_size=SAMPLES, shuffle=True)
    x_left, _ = next(iter(dataloader))
    x_right, _ = next(iter(dataloader))
    z_left, _ = model.encode(x_left.to(device))
    z_right, _ = model.encode(x_right.to(device))
    weights = torch.arange(0, 1, 1 / INTERPOLATIONS).to(device)
    z = torch.lerp(
        z_left[:, None, :], z_right[:, None, :], weights[None, :, None]
    ).reshape(SAMPLES * INTERPOLATIONS, model.latent_dim())
    x_hat = model.decode(z).detach().squeeze()

    for i in range(SAMPLES):
        for j in range(INTERPOLATIONS):
            axs[i, j + 1].imshow(x_hat[i * INTERPOLATIONS + j].cpu(), cmap="gray")
            axs[i, j + 1].axis("off")
        axs[i, 0].imshow(x_left[i].squeeze().cpu(), cmap="gray")
        axs[i, 0].axis("off")
        axs[i, -1].imshow(x_right[i].squeeze().cpu(), cmap="gray")
        axs[i, -1].axis("off")

    fig.savefig(
        f"{output_dir}/interpolation-reconstruction/{model.genus()}-{model.id()}.png"
    )
    plt.close()


def generate_latent_space_constructions(model, output_dir, seed=23):
    manual_seed(seed)
    device = next(model.parameters()).device

    ROWS = 8
    COLS = 8
    fig, axs = plt.subplots(ROWS, COLS, figsize=(2 * COLS, 2 * ROWS))

    samples = torch.randn(ROWS * COLS, model.latent_dim()).to(device)
    x_hat = model.decode(samples).detach().squeeze()

    for i in range(ROWS):
        for j in range(COLS):
            axs[i, j].imshow(x_hat[i * COLS + j].cpu(), cmap="gray")
            axs[i, j].axis("off")

    fig.savefig(f"{output_dir}/latent-construction/{model.genus()}-{model.id()}.png")
    plt.close()


def generate_latent_space_interpolations(model, output_dir, seed=23):
    manual_seed(seed)
    device = next(model.parameters()).device

    SAMPLES = 8
    INTERPOLATIONS = 8
    fig, axs = plt.subplots(
        SAMPLES,
        INTERPOLATIONS,
        figsize=(2 * INTERPOLATIONS, 2 * SAMPLES),
    )

    z_left = torch.randn(SAMPLES, model.latent_dim())
    z_right = torch.randn(SAMPLES, model.latent_dim())
    weights = torch.arange(0, 1, 1 / INTERPOLATIONS)
    z = (
        torch.lerp(z_left[:, None, :], z_right[:, None, :], weights[None, :, None])
        .reshape(SAMPLES * INTERPOLATIONS, model.latent_dim())
        .to(device)
    )
    x_hat = model.decode(z).detach().squeeze()

    for i in range(SAMPLES):
        for j in range(INTERPOLATIONS):
            axs[i, j].imshow(x_hat[i * INTERPOLATIONS + j].cpu(), cmap="gray")
            axs[i, j].axis("off")

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
