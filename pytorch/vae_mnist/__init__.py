import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from .models import VAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    print("running vae_mnist...")

    # load data
    dataset = MNIST(
        root="datasets",
        train=True,
        transform=ToTensor(),
        download=True,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # model test run
    model = VAE().to(DEVICE)
    x, _ = next(iter(dataloader))
    x_hat = model(x.to(DEVICE))

    assert x.shape == x_hat.shape
