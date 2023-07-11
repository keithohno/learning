import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def plot_sample_reconstructions(dataset, model, output_dir, seed=23):
    torch.manual_seed(seed)
    device = next(model.parameters()).device

    DATA_SAMPLES = 8

    dataloader = DataLoader(dataset, batch_size=DATA_SAMPLES, shuffle=True)
    x, _ = next(iter(dataloader))
    x = x.to(device)

    fig, axs = plt.subplots(DATA_SAMPLES, 2, figsize=(4, 16))

    with torch.inference_mode():
        x_hat = model(x).detach().squeeze()
        for i in range(8):
            axs[i, 0].imshow(x[i].squeeze().cpu(), cmap="gray")
            axs[i, 1].imshow(x_hat[i].cpu(), cmap="gray")
            axs[i, 0].axis("off")
            axs[i, 1].axis("off")

    axs[0, 0].set_title("Original")
    axs[0, 1].set_title("Reconstruction")

    fig.savefig(f"{output_dir}/plots/samples.png")


def plot_latent_space_parameters(dataset, model, output_dir, seed=23):
    torch.manual_seed(seed)
    device = next(model.parameters()).device

    means = torch.tensor([]).to(device)
    stds = torch.tensor([]).to(device)

    dataloader = DataLoader(dataset, batch_size=1)
    with torch.inference_mode():
        for x, _ in dataloader:
            x = x.to(device)
            mean, std = model.encode(x)
            means = torch.cat((means, mean))
            stds = torch.cat((stds, std))

    means = means.permute(1, 0).cpu()
    stds = stds.permute(1, 0).cpu()

    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(4):
        for j in range(4):
            axs[i, j].hist(means[i * 4 + j], bins=50)
    fig.savefig(f"{output_dir}/plots/latent_means.png")

    fig, axs = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(4):
        for j in range(4):
            axs[i, j].hist(stds[i * 4 + j], bins=50)
    fig.savefig(f"{output_dir}/plots/latent_stds.png")
