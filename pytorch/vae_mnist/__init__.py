import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from .models import VAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # model initialization
    model = VAE().to(DEVICE)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # training statistics
    epoch_list = []
    loss_list = []

    # training loop
    print("training...")
    for epoch in tqdm(range(5)):
        model.train()
        for x, _ in train_dataloader:
            x = x.to(DEVICE)
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # get test/train loss
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

    print("loss_list: ", loss_list)
