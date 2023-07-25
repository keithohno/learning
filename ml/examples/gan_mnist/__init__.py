import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.plots import plot_line_graphs, plot_image_grid
from helpers import get_device, get_dir, manual_seed
from .models import Discriminator, Generator
from .datasets import NoiseDataset, MixedDataset

DEVICE = get_device()
DIR = get_dir(__file__)


def create_mixed_dataloader(mnist_data, generator, seed=23):
    manual_seed(seed)
    generator.eval()
    with torch.inference_mode():
        train_disc_dataset = MixedDataset(
            mnist_data.data.unsqueeze(1).float() / 255,
            generator.generate_data_from_noise(len(mnist_data.data)),
            DEVICE,
        )
        train_disc_dataloader = DataLoader(
            train_disc_dataset, batch_size=32, shuffle=True
        )

    return train_disc_dataloader


def create_noise_dataloader(size, dim, seed=23):
    manual_seed(seed)
    gen_dataset = NoiseDataset(size, dim, DEVICE)
    gen_dataloader = DataLoader(gen_dataset, batch_size=32, shuffle=True)
    return gen_dataloader


def plot_generator_output(generator, label):
    generator.eval()
    x_grid = generator.generate_data_from_noise(100).reshape(10, 10, 28, 28)
    fig, _ = plot_image_grid(x_grid.cpu().detach())
    fig.savefig(f"{DIR}/results/generated/{label}.png")
    plt.close(fig)


def run():
    manual_seed(23)
    train_mnist = MNIST("datasets", download=True, train=True)
    test_mnist = MNIST("datasets", download=True, train=False)

    d_model = Discriminator().to(DEVICE)
    g_model = Generator().to(DEVICE)

    if d_model.can_load(f"{DIR}/models") and g_model.can_load(f"{DIR}/models"):
        d_model.load(f"{DIR}/models")
        g_model.load(f"{DIR}/models")

    else:
        d_model.compile(nn.BCEWithLogitsLoss())
        g_model.compile(nn.BCEWithLogitsLoss())

        pre_gen_loss_history = []
        post_gen_loss_history = []
        pre_gen_acc_history = []
        post_gen_acc_history = []
        for epoch in tqdm(range(20)):
            # train discriminator
            train_disc_dataloader = create_mixed_dataloader(train_mnist, g_model)
            d_model.train()
            for x, y in train_disc_dataloader:
                d_model.train_batch(x, y)

            # evaluate discriminator (pre gen round)
            d_model.eval()
            with torch.inference_mode():
                test_disc_dataloader = create_mixed_dataloader(test_mnist, g_model)
                av_loss = 0
                av_acc = 0
                for x, y in test_disc_dataloader:
                    y_hat = d_model(x)
                    av_loss += d_model.loss_fn(y_hat, y).item()
                    av_acc += y_hat.sigmoid().round().eq(y).float().mean().item()
            pre_gen_loss_history.append(av_loss / len(test_disc_dataloader))
            pre_gen_acc_history.append(av_acc / len(test_disc_dataloader))

            # train generator
            g_model.train()
            gen_dataloader = create_noise_dataloader(
                len(train_mnist), g_model.noise_dim
            )
            for z, y in gen_dataloader:
                g_model.train_batch(z, y, d_model)

            # evaluate discriminator (post gen round)
            d_model.eval()
            with torch.inference_mode():
                test_disc_dataloader = create_mixed_dataloader(test_mnist, g_model)
                av_loss = 0
                av_acc = 0
                for x, y in test_disc_dataloader:
                    y_hat = d_model(x)
                    av_loss += d_model.loss_fn(y_hat, y).item()
                    av_acc += y_hat.sigmoid().round().eq(y).float().mean().item()
            post_gen_loss_history.append(av_loss / len(test_disc_dataloader))
            post_gen_acc_history.append(av_acc / len(test_disc_dataloader))

            # plot mid-training generator outputs
            plot_generator_output(g_model, f"epoch_{epoch+1}")

        d_model.save(f"{DIR}/models")
        g_model.save(f"{DIR}/models")

        # plot loss/acc
        fig = plot_line_graphs(
            [pre_gen_loss_history, post_gen_loss_history], ["pre", "post"]
        )
        plt.xlabel("epoch")
        plt.ylabel("loss")
        fig.savefig(f"{DIR}/results/loss.png")
        plt.close(fig)

        fig = plot_line_graphs(
            [pre_gen_acc_history, post_gen_acc_history], ["pre", "post"]
        )
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        fig.savefig(f"{DIR}/results/accuracy.png")
        plt.close(fig)
