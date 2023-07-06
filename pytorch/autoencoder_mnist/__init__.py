import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from .models import ModelV1, ModelV2, ModelV3

DIR = os.path.dirname(os.path.realpath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50


# runs train -> plot pipeline for a list of models
def run_pipeline_for_models(models, train_dataset, test_dataset):
    loss_lists = []
    for model in models:
        if not (try_load_model(model)):
            loss_lists.append(train(model, train_dataset, test_dataset))

    if len(loss_lists) == len(models):
        plot_loss_charts(models, loss_lists)
    plot_sample_reconstruction(models, test_dataset)


# trains/saves model and reports loss
def train(model, train_dataset, test_dataset, seed=23):
    print("Training", model.name, model.spec)

    torch.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # training loop
    loss_list = []
    for _ in tqdm(range(EPOCHS)):
        model.train()
        for x, _ in train_dataloader:
            x = x.to(DEVICE)
            x_hat = model(x)
            loss = model.loss_fn(x_hat, x)
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

        model.eval()
        with torch.inference_mode():
            loss = 0
            for x, _ in test_dataloader:
                x = x.to(DEVICE)
                x_hat = model(x)
                loss += model.loss_fn(x_hat, x).item()
            loss_list.append(loss / len(test_dataloader))

    torch.save(model.state_dict(), f"{DIR}/data/{model.name}-{model.spec}.pt")

    return loss_list


# plots loss charts
def plot_loss_charts(models, loss_lists):
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
    plt.savefig(f"{DIR}/results/{models[0].name}_loss.png")


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

    fig.savefig(f"{DIR}/results/{model.name}_samples.png")


def try_load_model(model):
    try:
        state_dict = torch.load(f"{DIR}/data/{model.name}-{model.spec}.pt")
        model.load_state_dict(state_dict)
    except:
        return False
    return True


def run():
    train_dataset = MNIST("datasets", download=True, transform=ToTensor())
    test_dataset = MNIST("datasets", download=True, transform=ToTensor(), train=False)

    # ModelV1 block
    models = []
    latent_dims = [3, 5, 7, 10, 15]
    for latent_dim in latent_dims:
        models.append(ModelV1(latent_dim).to(DEVICE))
    run_pipeline_for_models(models, train_dataset, test_dataset)

    # ModelV2 block
    models = []
    latent_dims = [8, 16, 32]
    for latent_dim in latent_dims:
        models.append(ModelV2(latent_dim).to(DEVICE))
        models.append(ModelV2(latent_dim, activation=nn.ReLU()).to(DEVICE))
    run_pipeline_for_models(models, train_dataset, test_dataset)

    # ModelV3 block
    models = []
    for mse_loss in [False, True]:
        for sgd_optim in [False, True]:
            models.append(ModelV3(sgd_optim=sgd_optim, mse_loss=mse_loss).to(DEVICE))
    run_pipeline_for_models(models, train_dataset, test_dataset)
