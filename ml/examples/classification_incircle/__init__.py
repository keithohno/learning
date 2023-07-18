import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

RADIUS = 0.8


def is_in_circle(x, y):
    return x**2 + y**2 < RADIUS**2


def random_point():
    return torch.rand(2) * 2 - 1


def run_incircle_model():
    # generate train/test data
    points = torch.stack([random_point() for _ in range(10000)])
    predictions = (points.norm(dim=1) < RADIUS).float().view(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(
        points, predictions, test_size=0.2, random_state=23
    )

    # create model & training objects
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # accuracy data for plotting
    epoch_data = []
    test_accuracy_data = []
    train_accuracy_data = []

    # training loop
    EPOCHS = 10
    for epoch in range(1, EPOCHS + 1):
        # train model
        model.train()
        for x, y in zip(x_train, y_train):
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # evaluate model
        model.eval()
        with torch.inference_mode():
            y_test_hat = model(x_test).sigmoid().round()
            test_correct = (y_test_hat).eq(y_test).sum().item()
            y_train_hat = model(x_train).sigmoid().round()
            train_correct = (y_train_hat).eq(y_train).sum().item()

            epoch_data.append(epoch)
            test_accuracy_data.append(test_correct / 2000)
            train_accuracy_data.append(train_correct / 8000)

    return epoch_data, test_accuracy_data, train_accuracy_data


def run():
    torch.manual_seed(23)
    epoch_data, test_accuracy_data, train_accuracy_data = run_incircle_model()
    plt.plot(epoch_data, test_accuracy_data, label="test")
    plt.plot(epoch_data, train_accuracy_data, label="train")
    plt.legend()
    plt.savefig("examples/classification_incircle/out.png")
