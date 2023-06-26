import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from helpers import eval_model


def get_dataloaders():
    train_dataset = MNIST("datasets", download=True, transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = MNIST("datasets", download=True, transform=ToTensor(), train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, test_dataloader


def train(model, loss_fn, optimizer, train_dataloader, test_dataloader):
    epoch_list = []
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in tqdm(range(10)):
        for x, y in train_dataloader:
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 2 == 0:
            train_loss, train_acc = eval_model(model, train_dataloader, loss_fn)

            test_loss, test_acc = eval_model(model, test_dataloader, loss_fn)

            epoch_list.append(epoch)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

    return epoch_list, train_loss_list, train_acc_list, test_loss_list, test_acc_list


def run():
    torch.manual_seed(23)
    train_dataloader, test_dataloader = get_dataloaders()

    model = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    train(model, loss_fn, optimizer, train_dataloader, test_dataloader)
