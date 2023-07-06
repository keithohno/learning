import torch
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt


# plots loss charts
def plot_loss_charts(models, loss_lists, output_dir):
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
