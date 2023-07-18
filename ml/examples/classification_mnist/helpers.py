import matplotlib.pyplot as plt


def eval_model(model, dataloader, loss_fn):
    loss = 0
    acc = 0
    for x, y in dataloader:
        y_hat = model(x)
        loss += loss_fn(y_hat, y)
        acc += y_hat.argmax(dim=1).eq(y).float().mean()
    loss /= len(dataloader)
    acc /= len(dataloader)
    return loss, acc


def plot_loss_accuracy(
    train_loss_list, test_loss_list, train_acc_list, test_acc_list, epoch_list, path
):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(epoch_list, train_loss_list, label="training")
    ax1.plot(epoch_list, test_loss_list, label="testing")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(epoch_list, train_acc_list, label="training")
    ax2.plot(epoch_list, test_acc_list, label="testing")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.savefig(path)
