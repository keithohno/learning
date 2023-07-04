import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm

from .models import ModelV1


def run_training_pipeline(model, train_dataset, test_dataset, seed=23):
    torch.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    loss_list = []

    # training loop
    for _ in tqdm(range(50)):
        model.train()
        for x, _ in train_dataloader:
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.inference_mode():
            loss = 0
            for x, _ in test_dataloader:
                x_hat = model(x)
                loss += loss_fn(x_hat, x)
            loss_list.append(loss / len(test_dataloader))

    torch.save(model.state_dict(), f"autoencoder_mnist/data/{model.name()}.pt")

    return loss_list


def plot_loss_charts(models, loss_lists, colors):
    fig, ax = plt.subplots()
    for i in range(len(models)):
        ax.plot(loss_lists[i], color=colors[i], label=models[i].name())
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig("autoencoder_mnist/results/loss.png")


def plot_sample_reconstruction(model, test_dataset, seed=23):
    torch.manual_seed(seed)
    fig, axs = plt.subplots(4, 8)
    samples, _ = next(iter(DataLoader(test_dataset, batch_size=16, shuffle=True)))
    reconstructions = model(samples).detach().numpy()
    for row in range(4):
        for col in range(4):
            img = samples[row * 4 + col].squeeze()
            img_hat = reconstructions[row * 4 + col].squeeze()
            axs[row, col].imshow(img, cmap="gray")
            axs[row, col + 4].imshow(img_hat, cmap="gray")
            axs[row, col].axis("off")
            axs[row, col + 4].axis("off")

    fig.savefig(f"autoencoder_mnist/results/{model.name()}_sample.png")


def plot_latent_space(model, seed=23):
    torch.manual_seed(seed)
    fig, axs = plt.subplots(4, 12)
    for row in range(4):
        for col in range(4):
            vec1 = torch.tensor([0.3 * row, 0.3 * col, 0.0])
            vec2 = torch.tensor([0.3 * row, 0.0, 0.3 * col])
            vec3 = torch.tensor([0.0, 0.3 * row, 0.3 * col])

            pred1 = model.decoder(vec1).detach().numpy().squeeze()
            pred2 = model.decoder(vec2).detach().numpy().squeeze()
            pred3 = model.decoder(vec3).detach().numpy().squeeze()

            axs[row, col].imshow(pred1, cmap="gray")
            axs[row, col + 4].imshow(pred2, cmap="gray")
            axs[row, col + 8].imshow(pred3, cmap="gray")

    for ax in axs.flatten():
        ax.axis("off")

    fig.savefig(f"autoencoder_mnist/results/{model.name()}_latent.png")


def try_load_model(model):
    try:
        torch.load(f"autoencoder_mnist/data/{model.name()}.pt")
    except:
        return False
    return True


def run():
    train_dataset = MNIST("datasets", download=True, transform=ToTensor())
    test_dataset = MNIST("datasets", download=True, transform=ToTensor(), train=False)

    models = []
    latent_dims = [3, 5, 7, 10, 15]
    for latent_dim in latent_dims:
        models.append(ModelV1(latent_dim))

    loss_lists = []
    colors = [
        (x / len(models) / 2 + 0.5, x / len(models) / 2 + 0.5, 1.0)
        for x in range(len(models))
    ]
    for model in models:
        if not (try_load_model(model)):
            loss_lists.append(run_training_pipeline(model, train_dataset, test_dataset))

    plot_loss_charts(models, loss_lists, colors)
