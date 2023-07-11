from .models import VAE
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


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
    model = VAE()
    x, _ = next(iter(dataloader))
    x_hat = model(x)

    assert x.shape == x_hat.shape
