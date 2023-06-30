import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from .data import PointInClusterDataset


def train():
    # generate data
    train_dataset = PointInClusterDataset(train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = PointInClusterDataset(train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    num_clusters = train_dataset.num_clusters

    # define model
    model = nn.Sequential(
        nn.Linear(3, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_clusters),
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # data for plotting
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []
    epoch_list = []

    # training loop
    for epoch in tqdm(range(1001)):
        model.train()
        for x, y in test_dataloader:
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # evaluate loss/accuracy
        if epoch % 100 == 0:
            model.eval()
            with torch.inference_mode():
                train_loss = 0
                train_acc = 0
                for x, y in train_dataloader:
                    y_hat = model(x)
                    train_loss += loss_fn(y_hat, y)
                    train_acc += y_hat.argmax(dim=1).eq(y).float().mean().item()
                train_loss /= len(train_dataloader)
                train_acc /= len(train_dataloader)

                test_loss = 0
                test_acc = 0
                for x, y in test_dataloader:
                    y_hat = model(x)
                    test_loss += loss_fn(y_hat, y)
                    test_acc += y_hat.argmax(dim=1).eq(y).float().mean().item()
                test_loss /= len(test_dataloader)
                test_acc /= len(test_dataloader)

                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                epoch_list.append(epoch)

    # plotting
    plt.figure()
    plt.plot(epoch_list, train_loss_list, label="training")
    plt.plot(epoch_list, test_loss_list, label="testing")
    plt.legend()
    plt.savefig("classification_clusters/loss.png")

    plt.figure()
    plt.plot(epoch_list, train_acc_list, label="training")
    plt.plot(epoch_list, test_acc_list, label="testing")
    plt.legend()
    plt.savefig("classification_clusters/accuracy.png")


def run():
    torch.manual_seed(23)
    train()
