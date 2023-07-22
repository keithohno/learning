import torch
from torch.utils.data import Dataset

from helpers import manual_seed


class NoiseDataset(Dataset):
    def __init__(self, size, dim, device, seed=23):
        manual_seed(seed)
        self.noise_data = torch.randn(size, dim).to(device)
        self.device = device

    def __len__(self):
        return self.noise_data.shape[0]

    def __getitem__(self, idx):
        return (self.noise_data[idx], torch.ones(1).to(self.device))


class MixedDataset(Dataset):
    def __init__(self, real_data, fake_data, device):
        self.real_data = real_data.to(device)
        self.fake_data = fake_data.to(device)
        self.device = device

    def __len__(self):
        return self.fake_data.shape[0] + self.real_data.shape[0]

    def __getitem__(self, idx):
        if idx < self.real_data.shape[0]:
            return (self.real_data[idx], torch.ones(1).to(self.device))
        else:
            return (
                self.fake_data[idx - self.real_data.shape[0]],
                torch.zeros(1).to(self.device),
            )
