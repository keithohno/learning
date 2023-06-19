import torch
import matplotlib.pyplot as plt

RADIUS = 0.8


def is_in_circle(x, y):
    return x**2 + y**2 < RADIUS**2


def random_point():
    return torch.rand(2) * 2 - 1


class InCircleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(2, 10)
        self.layer2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


def run_incircle_model():
    # generate train/test data
    points = torch.stack([random_point() for _ in range(10000)])
    predictions = (points.norm(dim=1) < RADIUS).float().view(-1, 1)
    x_train, y_train = points[:8000], predictions[:8000]
    x_test, y_test = points[8000:], predictions[8000:]

    # initialize model & optimizer
    model = InCircleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()

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
            y_test_hat = model(x_test)
            test_correct = (y_test_hat.gt(0.5)).eq(y_test.gt(0.5)).sum().item()
            y_train_hat = model(x_train)
            train_correct = (y_train_hat.gt(0.5)).eq(y_train.gt(0.5)).sum().item()

            epoch_data.append(epoch)
            test_accuracy_data.append(test_correct / 2000)
            train_accuracy_data.append(train_correct / 8000)

    return epoch_data, test_accuracy_data, train_accuracy_data


if __name__ == "__main__":
    torch.manual_seed(23)
    epoch_data, test_accuracy_data, train_accuracy_data = run_incircle_model()
    plt.plot(epoch_data, test_accuracy_data, label="test")
    plt.plot(epoch_data, train_accuracy_data, label="train")
    plt.legend()
    plt.savefig("out.png")
