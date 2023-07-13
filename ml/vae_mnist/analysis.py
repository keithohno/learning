import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.pylab as pl


def generate_sample_reconstructions(dataset, models, output_dir, seed=23):
    torch.manual_seed(seed)
    device = next(models[0].parameters()).device

    DATA_SAMPLES = 8

    dataloader = DataLoader(dataset, batch_size=DATA_SAMPLES, shuffle=True)
    x, _ = next(iter(dataloader))
    x = x.to(device)

    fig, axs = plt.subplots(
        DATA_SAMPLES, len(models) + 1, figsize=(2 + len(models) * 2, 16)
    )

    with torch.inference_mode():
        for i, model in enumerate(models):
            x_hat, _, _ = model(x)
            x_hat = x_hat.detach().squeeze()
            for j in range(DATA_SAMPLES):
                axs[j, i + 1].imshow(x_hat[j].cpu(), cmap="gray")
                axs[j, i + 1].axis("off")
            axs[0, i + 1].set_title(model.id())

    for j in range(DATA_SAMPLES):
        axs[j, 0].imshow(x[j].squeeze().cpu(), cmap="gray")
        axs[j, 0].axis("off")

    axs[0, 0].set_title("Original")

    fig.savefig(f"{output_dir}/reconstruction/{model.genus()}.png")


def generate_latent_space_constructions(model, output_dir, seed=23):
    torch.manual_seed(seed)
    device = next(model.parameters()).device

    ROWS = 8
    COLS = 8

    fig, axs = plt.subplots(ROWS, COLS, figsize=(2 * COLS, 2 * ROWS))
    samples = torch.randn(ROWS * COLS, model.latent_dim()).to(device)

    with torch.inference_mode():
        x_hat = model.decode(samples).detach().squeeze()
        for i in range(ROWS):
            for j in range(COLS):
                axs[i, j].imshow(x_hat[i * COLS + j].cpu(), cmap="gray")
                axs[i, j].axis("off")

    fig.suptitle(f"{model.genus()}-{model.id()}")
    fig.savefig(f"{output_dir}/z-construction/{model.genus()}-{model.id()}.png")


def plot_latent_space_parameters(dataset, models, output_dir, seed=23):
    torch.manual_seed(seed)
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

    fig.savefig(f"{output_dir}/z-shape/{model.genus()}.png")


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
