import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from .models import VAEv1, VAEv2
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
    for _ in tqdm(range(25)):
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
    model.save_to_disk(f"{DIR}/models")

    return loss_history


def run_pipeline_for_models(models, train_dataset, test_dataset, seed=23):
    # create dataloaders
    torch.manual_seed(seed)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # train or load models
    if all([model.can_load_from_disk(f"{DIR}/models") for model in models]):
        for model in models:
            model.load_from_disk(f"{DIR}/models")
    else:
        loss_histories = []
        for model in models:
            loss = train(model, train_dataloader, test_dataloader)
            loss_histories.append(loss)
        plot_loss_history(models, loss_histories, f"{DIR}/plots")

    # plot results
    plot_sample_reconstructions(test_dataset, models, f"{DIR}/plots")
    plot_latent_space_parameters(test_dataset, models, f"{DIR}/plots")


def run():
    print("running vae_mnist...")

    # load data
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

    # v1 model block
    models = []
    for beta in [0.1, 0.5, 1.0]:
        models.append(VAEv1(beta).to(DEVICE))
    run_pipeline_for_models(models, train_dataset, test_dataset)

    # v2 model block
    models = []
    for beta in [0.1, 0.5, 1.0]:
        models.append(VAEv2(beta).to(DEVICE))
    run_pipeline_for_models(models, train_dataset, test_dataset)
