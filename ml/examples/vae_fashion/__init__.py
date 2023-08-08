import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from common.plots import plot_normalized_loss_histories
from common.utils import get_dir, manual_seed

from . import experiment_beta
from .models import VAEv1, VAEv2
from .plots import (
    generate_latent_lerp,
    generate_latent_samples,
    generate_reconstruction_lerp,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIR = get_dir(__file__)


def train(model, train_dataloader, test_dataloader, epochs, seed=23):
    loss_history = []
    for _ in tqdm(range(epochs)):
        for x, _ in train_dataloader:
            model.train_batch(x.to(DEVICE))

        loss = 0
        for x, _ in test_dataloader:
            loss += model.eval_batch(x.to(DEVICE))
        loss /= len(test_dataloader)
        loss_history.append(loss)

    model.save(f"{DIR}/models")
    return loss_history


def run_model_group(models, train_dataset, test_dataset, epochs, seed=23):
    manual_seed(seed)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    if all([model.can_load(f"{DIR}/models") for model in models]):
        for model in models:
            model.load(f"{DIR}/models")
    else:
        loss_histories = []
        for model in models:
            loss_history = train(model, train_dataloader, test_dataloader, epochs)
            loss_histories.append(loss_history)
        betas = [model.beta for model in models]
        fig = plot_normalized_loss_histories(loss_histories, betas)
        fig.savefig(f"{DIR}/plots/loss/{models[0].__class__.__name__}.png")
        plt.close(fig)

    for model in models:
        generate_latent_lerp(model, f"{DIR}/plots")
        generate_reconstruction_lerp(model, test_dataset, f"{DIR}/plots")
        generate_latent_samples(model, f"{DIR}/plots")


def run():
    experiment_beta.run()

    train_dataset = FashionMNIST(
        root="datasets",
        train=True,
        transform=ToTensor(),
        download=True,
    )
    test_dataset = FashionMNIST(
        root="datasets",
        train=False,
        transform=ToTensor(),
        download=True,
    )

    betas = [0.1, 0.2, 0.5, 1.0, 1.5]
    models = [VAEv1(beta=beta).to(DEVICE) for beta in betas]
    for model in models:
        model.compile()
    run_model_group(models, train_dataset, test_dataset, epochs=50)

    betas = [0.1, 0.2, 0.5, 1.0, 1.5]
    models = [VAEv2(beta=beta).to(DEVICE) for beta in betas]
    for model in models:
        model.compile()
    run_model_group(models, train_dataset, test_dataset, epochs=50)
