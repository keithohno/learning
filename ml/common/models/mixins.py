import torch
import os


class SaveLoadMixin:
    def save(self, model_dir):
        torch.save(self.state_dict(), f"{model_dir}/{self.id()}.pt")

    def load(self, model_dir):
        self.load_state_dict(torch.load(f"{model_dir}/{self.id()}.pt"))

    def can_load(self, model_dir):
        return os.path.isfile(f"{model_dir}/{self.id()}.pt")
