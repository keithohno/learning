from data import gen_clusters
from torch import nn
from sklearn.model_selection import train_test_split
import torch


def train():
    # geenerate data
    points, labels = gen_clusters()
    num_classes = int(labels.shape[1])
    xs_train, xs_test, ys_train, ys_test = train_test_split(
        points, labels, test_size=0.2, random_state=23
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
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # evaluate loss/accuracy
        if epoch % 50 == 0:
            ys_train_hat = model(xs_train)
            ys_test_hat = model(xs_test)

            train_loss = loss_fn(ys_train_hat, ys_train).item()
            test_loss = loss_fn(ys_test_hat, ys_test).item()

            train_accuracy = (
                ys_train_hat.argmax(dim=1)
                .eq(ys_train.argmax(dim=1))
                .float()
                .mean()
                .item()
            )
            test_accuracy = (
                ys_test_hat.argmax(dim=1)
                .eq(ys_test.argmax(dim=1))
                .float()
                .mean()
                .item()
            )

            print(
                f"Train: loss={train_loss:.2f}, acc={train_accuracy:.2f}",
                f" |  Test: train={test_loss:.2f}, test={test_accuracy:.2f}",
            )


if __name__ == "__main__":
    torch.manual_seed(23)
    train()
