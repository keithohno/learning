import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_dataloaders():
    train_dataset = MNIST("datasets", download=True, transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = MNIST("datasets", download=True, transform=ToTensor(), train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader


def run():
    torch.manual_seed(23)

    train_dataloader, test_dataloader = get_dataloaders()

    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3),
        nn.MaxPool2d(2),
        nn.SELU(),
        nn.Conv2d(8, 8, kernel_size=3),
        nn.MaxPool2d(2, padding=1),
        nn.SELU(),
        nn.Conv2d(8, 8, kernel_size=3),
        nn.MaxPool2d(2),
        nn.SELU(),
        nn.Flatten(),
        nn.Linear(32, 3),
        nn.SELU(),
        nn.Linear(3, 32),
        nn.Unflatten(1, (8, 2, 2)),
        nn.ConvTranspose2d(8, 8, kernel_size=4, stride=3),
        nn.SELU(),
        nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2),
        nn.SELU(),
        nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2),
        nn.SELU(),
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    loss_list = []
    baseline_loss_list = []

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

            loss = 0
            for x, _ in test_dataloader:
                x_hat = model(x)
                loss += loss_fn(x_hat, x.flip(0))
            baseline_loss_list.append(loss / len(test_dataloader))

    plt.plot(loss_list, label="loss")
    plt.plot(baseline_loss_list, label="baseline")
    plt.legend()
    plt.savefig("autoencoder_mnist/loss.png")
