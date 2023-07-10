import torch
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt


# plots loss charts
def plot_loss_history(models, loss_lists, output_dir):
    colors = [
        (x / len(models) / 2 + 0.5, x / len(models) / 2 + 0.5, 1.0)
        for x in range(len(models))
    ]

    plt.figure()
    for i in range(len(models)):
        plt.plot(loss_lists[i], color=colors[i], label=models[i].spec)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{output_dir}/results/{models[0].name}_loss.png")


# samples dataset and plots original and reconstructed images
def plot_sample_reconstruction(models, test_dataset, device, output_dir, seed=23):
    torch.manual_seed(seed)
    SAMPLES = 8
    fig, axs = plt.subplots(SAMPLES, len(models) + 1, figsize=(16, 16))
    x, _ = next(iter(DataLoader(test_dataset, batch_size=SAMPLES, shuffle=True)))
    x = x.to(device)

    for i, model in enumerate(models):
        model.eval()
        x_hat = model(x).detach()
        axs[0, i + 1].set_title(model.spec)
        for j in range(SAMPLES):
            img = x_hat[j].squeeze()
            axs[j, i + 1].imshow(img.cpu(), cmap="gray")
            axs[j, i + 1].axis("off")

    axs[0, 0].set_title("Original")
    for j in range(SAMPLES):
        img = x[j].squeeze()
        axs[j, 0].imshow(img.cpu(), cmap="gray")
        axs[j, 0].axis("off")

    fig.savefig(f"{output_dir}/results/{model.name}_samples.png")


def estimate_latent_distribution(model, dataloader, device):
    # calculate latent space means for each digit
    means = torch.zeros((10, model.latent_dim)).to(device)
    counts = torch.zeros(10).to(device)
    model.eval()
    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device)
            z = model.encoder(x)
            for digit, latent in zip(y, z):
                means[digit] += latent
                counts[digit] += 1
    means /= counts.reshape(-1, 1)

    # calculate latent space stds for each digit
    stds = torch.zeros((10, model.latent_dim)).to(device)
    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device)
            z = model.encoder(x)
            for digit, latent in zip(y, z):
                stds[digit] += (latent - means[digit]) ** 2
    stds /= counts.reshape(-1, 1)
    stds = torch.sqrt(stds)

    return means, stds


def plot_latent_reconstruction(models, test_dataset, device, output_dir, seed=23):
    torch.manual_seed(seed)

    fig1, axs1 = plt.subplots(10, len(models), figsize=(len(models) * 2, 20))
    fig2, axs2 = plt.subplots(10, len(models), figsize=(len(models) * 2, 20))

    dataloader = DataLoader(test_dataset, batch_size=32)
    for i, model in enumerate(models):
        means, stds = estimate_latent_distribution(model, dataloader, device)

        # sample from latent space and plot reconstructions
        for j in range(10):
            sample = torch.normal(means[j], stds[j])
            img = model.decoder(sample).detach().squeeze()
            axs1[j, i].imshow(img.cpu(), cmap="gray")
            axs1[j, i].axis("off")
        axs1[0, i].set_title(model.spec)

        # take means from latent space and plot reconstructions
        for j in range(10):
            img = model.decoder(means[j]).detach().squeeze()
            axs2[j, i].imshow(img.cpu(), cmap="gray")
            axs2[j, i].axis("off")
        axs2[0, i].set_title(model.spec)

    fig1.savefig(f"{output_dir}/results/{models[0].name}_latent_sample.png")
    fig2.savefig(f"{output_dir}/results/{models[0].name}_latent_mean.png")
