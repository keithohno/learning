import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os

from .models import VAE
from .analysis import (
    plot_sample_reconstructions,
    plot_latent_space_parameters,
    plot_loss_history,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR = "vae_mnist"


def train(model, train_dataloader, test_dataloader):
    print(f"training {model.__class__.__name__}-{model.id()} ...")

    loss_history = []
    optimizer = torch.optim.Adam(model.parameters())

    # training loop
    for _ in tqdm(range(10)):
        model.train()
        for x, _ in train_dataloader:
            x = x.to(DEVICE)
            x_hat, mean, std = model(x)
            loss = model.loss(x, x_hat, mean, std)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # get test loss
        model.eval()
        with torch.inference_mode():
            loss = 0
            for x, _ in test_dataloader:
                x = x.to(DEVICE)
                x_hat, mean, std = model(x)
                loss += model.loss(x, x_hat, mean, std).item()
            loss /= len(test_dataloader)
            loss_history.append(loss)

    # save model
    torch.save(model.state_dict(), f"{DIR}/models/model-{model.id()}.pt")

    return loss_history


def can_load_models_from_disk(models):
    for model in models:
        if not os.path.isfile(f"{DIR}/models/model-{model.id()}.pt"):
            return False
    return True


def run():
    print("running vae_mnist...")

    # load data
    torch.manual_seed(23)
    train_dataset = MNIST(
        root="datasets",
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_dataset = MNIST(
        root="datasets",
        train=False,
        transform=ToTensor(),
        download=True,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model1 = VAE(0.1).to(DEVICE)
    model2 = VAE(0.5).to(DEVICE)
    model3 = VAE(1.0).to(DEVICE)
    models = [model1, model2, model3]

    # initialize and train/load model
    if can_load_models_from_disk(models):
        for model in models:
            model.load_from_disk(f"{DIR}/models")
    else:
        loss_histories = []
        for model in models:
            loss = train(model, train_dataloader, test_dataloader)
            loss_histories.append(loss)
        plot_loss_history(models, loss_histories, DIR)

    plot_sample_reconstructions(test_dataset, models, DIR)
    plot_latent_space_parameters(test_dataset, models, DIR)
