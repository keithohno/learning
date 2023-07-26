import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.plots import plot_line_graphs, plot_image_grid
from helpers import get_device, get_dir, manual_seed
from .models import Discriminator, Generator

DEVICE = get_device()
DIR = get_dir(__file__)


def plot_generator_output(generator, label):
    manual_seed(23)
    noise = torch.randn(100, 16).to(DEVICE)
    x_grid = generator(noise).reshape(10, 10, 28, 28)
    fig, _ = plot_image_grid(x_grid.cpu().detach())
    fig.savefig(f"{DIR}/results/generated/{label}.png")
    plt.close(fig)


def run():
    manual_seed(23)
    d_model = Discriminator().to(DEVICE)
    g_model = Generator().to(DEVICE)

    manual_seed(23)
    train_mnist = MNIST("datasets", download=True, train=True, transform=ToTensor())
    train_mnist_dl = DataLoader(train_mnist, batch_size=32, shuffle=True)
    test_mnist = MNIST("datasets", download=True, train=False, transform=ToTensor())
    test_mnist_dl = DataLoader(test_mnist, batch_size=32)

    manual_seed(23)
    noise = torch.randn(2, len(train_mnist_dl), 32, 16).to(DEVICE)

    if d_model.can_load(f"{DIR}/models") and g_model.can_load(f"{DIR}/models"):
        d_model.load(f"{DIR}/models")
        g_model.load(f"{DIR}/models")

    else:
        d_model.compile(nn.BCEWithLogitsLoss())
        g_model.compile(nn.BCEWithLogitsLoss())

        d_loss_history = []
        d_acc_history = []
        g_loss_history = []

        for epoch in tqdm(range(20)):
            for i, (x, _) in enumerate(train_mnist_dl):
                # train discriminator on real data
                d_model.train()
                y = torch.ones(32).to(DEVICE)
                d_model.train_batch(x.to(DEVICE), y.to(DEVICE).float())

                # train discriminator on fake data
                x_hat = g_model(noise[0][i])
                y = torch.zeros(32).to(DEVICE)
                d_model.train_batch(x_hat, y)

                # train generator
                g_model.train()
                y = torch.ones(32).to(DEVICE)
                g_model.train_batch(noise[1][i], y, d_model)

            # evaluate discriminator loss/acc
            d_model.eval()
            with torch.inference_mode():
                av_loss = 0
                av_acc = 0
                for i, (x, _) in enumerate(test_mnist_dl):
                    x = x.to(DEVICE)
                    y = torch.ones(len(x)).to(DEVICE)
                    y_hat = d_model(x).squeeze()
                    av_loss += d_model.loss_fn(y_hat, y).item()
                    av_acc += y_hat.sigmoid().round().eq(y).float().mean().item()

                    z = noise[0][i]
                    x = g_model(z)
                    y_hat = d_model(x_hat).squeeze()
                    y = torch.zeros(32).to(DEVICE)
                    av_loss += d_model.loss_fn(y_hat, y).item()
                    av_acc += y_hat.sigmoid().round().eq(y).float().mean().item()

                av_loss /= 2 * len(test_mnist_dl)
                av_acc /= 2 * len(test_mnist_dl)
                d_loss_history.append(av_loss)
                d_acc_history.append(av_acc)

            # evaluate generator loss
            g_model.eval()
            with torch.inference_mode():
                av_loss = 0
                for z in noise[1]:
                    x_hat = g_model(z)
                    y_hat = d_model(x_hat).squeeze()
                    y = torch.ones(32).to(DEVICE)
                    av_loss += g_model.loss_fn(y_hat, y).item()
                g_loss_history.append(av_loss / len(noise[1]))

            # plot mid-training generator outputs
            plot_generator_output(g_model, f"epoch_{epoch+1}")

        d_model.save(f"{DIR}/models")
        g_model.save(f"{DIR}/models")

        # plot loss/acc
        fig = plot_line_graphs([d_loss_history, g_loss_history], ["disc", "gen"])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        fig.savefig(f"{DIR}/results/loss.png")
        plt.close(fig)

        fig = plot_line_graphs([d_acc_history], ["disc"])
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        fig.savefig(f"{DIR}/results/accuracy.png")
        plt.close(fig)
