import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import torch


def plot_image_grid(tensor):
    """
    Plots and returns a grid of images. Expects input tensor to have shape (rows, cols, im_height, im_width)
    """
    rows = tensor.shape[0]
    cols = tensor.shape[1]
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(tensor[i, j])
            axs[i, j].axis("off")
    return fig, axs


def plot_normalized_loss_histories(loss_histories, labels):
    """
    Plots normalized loss charts for multiple models
    """
    epochs = len(loss_histories[0])
    colors = pl.cm.viridis(torch.linspace(0, 1, len(loss_histories)))
    fig = plt.figure(figsize=(10, 6))
    for i, loss_history in enumerate(loss_histories):
        max_loss = max(loss_history)
        min_loss = min(loss_history)
        normalized_loss = [
            (loss - min_loss) / (max_loss - min_loss) for loss in loss_history
        ]
        plt.plot(range(epochs), normalized_loss, label=labels[i], color=colors[i])
    plt.legend(labels)
    plt.xlabel("epoch")
    plt.ylabel("loss (normalized)")
    return fig


def plot_loss_histories(loss_histories, labels) -> plt.Figure:
    """
    Plot loss charts for multiple models
    """
    epochs = len(loss_histories[0])
    colors = pl.cm.viridis(torch.linspace(0, 1, len(loss_histories)))
    fig = plt.figure(figsize=(10, 6))
    for i, loss_history in enumerate(loss_histories):
        plt.plot(range(epochs), loss_history, label=labels[i], color=colors[i])
    plt.legend(labels)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    return fig


def plot_line_graphs(data, labels):
    colors = pl.cm.viridis(torch.linspace(0, 1, len(data)))
    fig = plt.figure(figsize=(10, 6))
    for i, x in enumerate(data):
        plt.plot(range(len(x)), x, label=labels[i], color=colors[i])
    fig.legend(labels)
    return fig
