import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from .models import VAE
from .analysis import plot_sample_reconstructions, plot_latent_space_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR = "vae_mnist"


def train(model, train_dataloader, test_dataloader):
    print(f"training {model.__class__.__name__}...")

    # training statistics
    epoch_list = []
    loss_list = []

    # loss function and optimizer
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # training loop
    for epoch in tqdm(range(5)):
        model.train()
        for x, _ in train_dataloader:
            x = x.to(DEVICE)
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # get test loss
        model.eval()
        with torch.inference_mode():
            loss = 0
            for x, _ in test_dataloader:
                x = x.to(DEVICE)
                x_hat = model(x)
                loss += loss_fn(x_hat, x).item()
            loss /= len(test_dataloader)
            epoch_list.append(epoch)
            loss_list.append(loss)

    # save model
    torch.save(model.state_dict(), f"{DIR}/models/model.pt")

    return epoch_list, loss_list


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

    # initialize and train/load model
    model = VAE().to(DEVICE)
    try:
        state_dict = torch.load(f"{DIR}/models/model.pt")
        model.load_state_dict(state_dict)
    except:
        train(model, train_dataloader, test_dataloader)
    plot_sample_reconstructions(test_dataset, model, DIR)
    plot_latent_space_parameters(test_dataset, model, DIR)
