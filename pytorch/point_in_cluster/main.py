from data import gen_clusters
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch


def train():
    # geenerate data
    points, labels = gen_clusters()
    num_clusters = int(labels.shape[1])
    xs_train, xs_test, ys_train, ys_test = train_test_split(
        points, labels, test_size=0.2, random_state=23
    )

    # define model
    model = nn.Sequential(
        nn.Linear(3, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, num_clusters),
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # data for plotting
    train_loss_list = []
    test_loss_list = []
    train_accuracy_list = []
    test_accuracy_list = []
    epoch_list = []

    # training loop
    for epoch in range(501):
        model.train()
        for x, y in zip(xs_train, ys_train):
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # evaluate loss/accuracy
        if epoch % 50 == 0:
            model.eval()
            with torch.inference_mode():
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

                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                train_accuracy_list.append(train_accuracy)
                test_accuracy_list.append(test_accuracy)
                epoch_list.append(epoch)

    # plotting
    plt.figure()
    plt.plot(epoch_list, train_loss_list, label="training")
    plt.plot(epoch_list, test_loss_list, label="testing")
    plt.legend()
    plt.savefig("loss.png")

    plt.figure()
    plt.plot(epoch_list, train_accuracy_list, label="training")
    plt.plot(epoch_list, test_accuracy_list, label="testing")
    plt.legend()
    plt.savefig("accuracy.png")


if __name__ == "__main__":
    torch.manual_seed(23)
    train()
