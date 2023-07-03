import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm

from .models import ModelV1


def get_dataloaders():
    train_dataset = MNIST("datasets", download=True, transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = MNIST("datasets", download=True, transform=ToTensor(), train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader


def run_training_pipeline(model, model_name, seed=23):
    torch.manual_seed(seed)

    train_dataloader, test_dataloader = get_dataloaders()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    loss_list = []
    baseline_loss_list = []

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

            loss = 0
            for x, _ in test_dataloader:
                x_hat = model(x)
                loss += loss_fn(x_hat, x.flip(0))
            baseline_loss_list.append(loss / len(test_dataloader))

    # save and plot
    torch.save(model.state_dict(), f"autoencoder_mnist/data/{model_name}.pt")

    plt.plot(loss_list, label="loss")
    plt.plot(baseline_loss_list, label="baseline")
    plt.legend()
    plt.savefig(f"autoencoder_mnist/results/{model_name}_loss.png")


def load_state_dict(model_name):
    try:
        return torch.load(f"autoencoder_mnist/data/{model_name}.pt")
    except:
        return None


def run():
    model = ModelV1()
    if (state_dict := load_state_dict("ModelV1")) is not None:
        model.load_state_dict(state_dict)
    else:
        run_training_pipeline(model, "ModelV1")
