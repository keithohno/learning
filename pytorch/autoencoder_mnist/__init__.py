import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm

from .models import ModelV1


def run_training_pipeline(model, model_name, train_dataset, test_dataset, seed=23):
    torch.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

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


def plot_sample_reconstruction(model, model_name, test_dataset, seed=23):
    torch.manual_seed(seed)
    fig, axs = plt.subplots(4, 8)
    samples, _ = next(iter(DataLoader(test_dataset, batch_size=16, shuffle=True)))
    reconstructions = model(samples).detach().numpy()
    for row in range(4):
        for col in range(4):
            img = samples[row * 4 + col].squeeze()
            img_hat = reconstructions[row * 4 + col].squeeze()
            axs[row, col].imshow(img, cmap="gray")
            axs[row, col + 4].imshow(img_hat, cmap="gray")
            axs[row, col].axis("off")
            axs[row, col + 4].axis("off")

    fig.savefig(f"autoencoder_mnist/results/{model_name}_sample.png")


def plot_latent_space(model, model_name, test_dataset, seed=23):
    torch.manual_seed(seed)
    fig, axs = plt.subplots(4, 12)
    for row in range(4):
        for col in range(4):
            vec1 = torch.tensor([0.3 * row, 0.3 * col, 0.0])
            vec2 = torch.tensor([0.3 * row, 0.0, 0.3 * col])
            vec3 = torch.tensor([0.0, 0.3 * row, 0.3 * col])

            pred1 = model.decoder(vec1).detach().numpy().squeeze()
            pred2 = model.decoder(vec2).detach().numpy().squeeze()
            pred3 = model.decoder(vec3).detach().numpy().squeeze()

            axs[row, col].imshow(pred1, cmap="gray")
            axs[row, col + 4].imshow(pred2, cmap="gray")
            axs[row, col + 8].imshow(pred3, cmap="gray")

    for ax in axs.flatten():
        ax.axis("off")

    fig.savefig(f"autoencoder_mnist/results/{model_name}_latent.png")


def load_state_dict(model_name):
    try:
        return torch.load(f"autoencoder_mnist/data/{model_name}.pt")
    except:
        return None


def run():
    train_dataset = MNIST("datasets", download=True, transform=ToTensor())
    test_dataset = MNIST("datasets", download=True, transform=ToTensor(), train=False)

    model = ModelV1()
    if (state_dict := load_state_dict("ModelV1")) is not None:
        model.load_state_dict(state_dict)
    else:
        run_training_pipeline(model, "ModelV1", train_dataset, test_dataset)

    plot_sample_reconstruction(model, "ModelV1", test_dataset)
    plot_latent_space(model, "ModelV1", test_dataset)
