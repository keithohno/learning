from data import gen_clusters
from torch import nn
from sklearn.model_selection import train_test_split
import torch


def train():
    # geenerate data
    points, labels = gen_clusters()
    num_classes = int(labels[-1].item())
    xs_train, xs_test, ys_train, ys_test = train_test_split(
        points, labels, train_size=0.2, random_state=23
    )

    # define model
    model = nn.Sequential(
        nn.Linear(3, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes),
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # training loop
    for epoch in range(200):
        for x, y in zip(xs_train, ys_train):
            y_hat = model(x)
            y_true = torch.tensor([1.0 if i == y else 0.0 for i in range(num_classes)])
            loss = loss_fn(y_hat, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    torch.manual_seed(23)
    train()
