import matplotlib.pyplot as plt


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
