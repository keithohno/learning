import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from common.models.VAE import VAE as BaseVAE
from common.plots import plot_image_grid, plot_loss_histories
from common.utils import get_device, get_dir, manual_seed

DEVICE = get_device()
DIR = get_dir(__file__)
MODEL_DIR = f"{DIR}/models"
PLOT_DIR = f"{DIR}/results"


class VAE(BaseVAE):
    def __init__(self, beta):
        super().__init__()
        self.nz = 64
        self.beta = beta
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.encoder = nn.Sequential(
            # 1 x 28 x 28
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # 32 x 14 x 14
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # 64 x 7 x 7
            nn.Conv2d(64, 128, 4, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # 128 x 4 x 4
        )

        self.encoder_mu = nn.Sequential(
            nn.Conv2d(128, self.nz, 4, bias=False),
            nn.Flatten(),
        )
        self.encoder_logvar = nn.Sequential(
            nn.Conv2d(128, self.nz, 4, bias=False),
            nn.Flatten(),
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(-1, (self.nz, 1, 1)),
            nn.ConvTranspose2d(self.nz, 128, 4, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            # 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, 4, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # 64 x 7 x 7
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # 32 x 14 x 14
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            # 1 x 28 x 28
        )

    def id(self):
        return f"exp1_{self.beta}"

    def compile(self):
        self.optimizer = torch.optim.Adam(self.parameters())

    def train_batch(self, x):
        x_hat, mean, std = self.forward(x)
        loss = self.loss(x_hat, x, mean, std)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # compute regulatization and reconstruction loss separately
        reg_loss = self.reg_loss(mean, std).item()
        rec_loss = self.loss_fn(x_hat, x).item()
        return loss.item(), reg_loss, rec_loss

    def eval_batch(self, x):
        x_hat, mean, std = self(x)
        loss = self.loss(x_hat, x, mean, std)
        return loss


def run():
    dataset = FashionMNIST("datasets", download=True, train=True, transform=ToTensor())

    manual_seed(23)
    dl = DataLoader(dataset, batch_size=128, shuffle=True)

    betas = [round(0.1 * i, 1) for i in range(21)]
    models: list[VAE] = []
    labels = []
    for beta in betas:
        model = VAE(beta=beta).to(DEVICE)
        model.compile()
        models.append(model)
        labels.append(f"beta={beta}")

    # for plotting
    loss_histories = []
    reg_loss_histories = []
    rec_loss_histories = []

    for model in models:
        # load model
        if model.can_load(MODEL_DIR):
            model.load(MODEL_DIR)
            print(f"loading beta={model.beta} from disk")
            continue
        print(f"training beta={model.beta}")

        # for plotting
        loss_history = []
        reg_loss_history = []
        rec_loss_history = []

        # training loop
        model.train()
        for _ in tqdm(range(10)):
            av_loss = 0
            av_reg_loss = 0
            av_rec_loss = 0
            for x, _ in dl:
                loss, reg_loss, rec_loss = model.train_batch(x.to(DEVICE))
                av_loss += loss
                av_reg_loss += reg_loss
                av_rec_loss += rec_loss
            loss_history.append(av_loss / len(dl))
            reg_loss_history.append(av_reg_loss / len(dl))
            rec_loss_history.append(av_rec_loss / len(dl))

        loss_histories.append(loss_history)
        reg_loss_histories.append(reg_loss_history)
        rec_loss_histories.append(rec_loss_history)

        # save model
        model.save(MODEL_DIR)

    # loss plotting
    if len(models) == len(loss_histories):
        fig = plot_loss_histories(loss_histories, labels)
        fig.savefig(f"{PLOT_DIR}/loss.png")
        plt.close(fig)
        fig = plot_loss_histories(reg_loss_histories, labels)
        fig.savefig(f"{PLOT_DIR}/reg_loss.png")
        plt.close(fig)
        fig = plot_loss_histories(rec_loss_histories, labels)
        fig.savefig(f"{PLOT_DIR}/rec_loss.png")
        plt.close(fig)

    # generate images
    manual_seed(23)
    dl = DataLoader(dataset, batch_size=16, shuffle=True)
    samples = next(iter(dl))[0].to(DEVICE)
    x_grid = []
    for model in models:
        model.eval()
        x_grid.append(model.forward_no_reparam(samples).detach().cpu().squeeze())
    x_grid = torch.stack(x_grid).permute(1, 0, 2, 3)
    fig, axs = plot_image_grid(x_grid)
    for i, label in enumerate(labels):
        axs[0, i].set_title(label, fontsize=16)
    fig.savefig(f"{PLOT_DIR}/generated.png", bbox_inches="tight", pad_inches=1)
