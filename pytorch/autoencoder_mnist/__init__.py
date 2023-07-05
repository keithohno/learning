import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from .models import ModelV1

DIR = os.path.dirname(os.path.realpath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# trains/saves model and reports loss
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
            x = x.to(DEVICE)
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.inference_mode():
            loss = 0
            for x, _ in test_dataloader:
                x = x.to(DEVICE)
                x_hat = model(x)
                loss += loss_fn(x_hat, x).item()
            loss_list.append(loss / len(test_dataloader))

    torch.save(model.state_dict(), f"{DIR}/data/{model.name()}.pt")

    return loss_list


# plots loss charts
def plot_loss_charts(models, loss_lists, colors):
    fig, ax = plt.subplots()
    for i in range(len(models)):
        loss_list = [x.item() for x in loss_lists[i]]
        ax.plot(loss_list, color=colors[i], label=models[i].name())
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.savefig(f"{DIR}/results/loss.png")


# samples dataset and plots original and reconstructed images
def plot_sample_reconstruction(models, test_dataset, seed=23):
    SAMPLES = 8
    torch.manual_seed(seed)
    fig, axs = plt.subplots(SAMPLES, len(models) + 1, figsize=(16, 16))
    x, _ = next(iter(DataLoader(test_dataset, batch_size=SAMPLES, shuffle=True)))
    x = x.to(DEVICE)

    for i, model in enumerate(models):
        model.eval()
        x_hat = model(x).detach()
        axs[0, i + 1].set_title(model.name())
        for j in range(SAMPLES):
            img = x_hat[j].squeeze()
            axs[j, i + 1].imshow(img.cpu(), cmap="gray")
            axs[j, i + 1].axis("off")

    axs[0, 0].set_title("Original")
    for j in range(SAMPLES):
        img = x[j].squeeze()
        axs[j, 0].imshow(img.cpu(), cmap="gray")
        axs[j, 0].axis("off")

    fig.savefig(f"{DIR}/results/samples.png")


def try_load_model(model):
    try:
        state_dict = torch.load(f"{DIR}/data/{model.name()}.pt")
        model.load_state_dict(state_dict)
    except:
        return False
    return True


def run():
    train_dataset = MNIST("datasets", download=True, transform=ToTensor())
    test_dataset = MNIST("datasets", download=True, transform=ToTensor(), train=False)

    models = []
    latent_dims = [3, 5, 7, 10, 15]
    for latent_dim in latent_dims:
        models.append(ModelV1(latent_dim).to(DEVICE))

    loss_lists = []
    colors = [
        (x / len(models) / 2 + 0.5, x / len(models) / 2 + 0.5, 1.0)
        for x in range(len(models))
    ]
    for model in models:
        if not (try_load_model(model)):
            loss_lists.append(run_training_pipeline(model, train_dataset, test_dataset))

    if loss_lists:
        plot_loss_charts(models, loss_lists, colors)
    plot_sample_reconstruction(models, test_dataset)
