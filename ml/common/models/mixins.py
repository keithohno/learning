import torch
import os


class SaveLoadMixin:
    def save_to_disk(self, model_dir):
        torch.save(self.state_dict(), f"{model_dir}/{self.id()}.pt")

    def load_from_disk(self, model_dir):
        self.load_state_dict(torch.load(f"{model_dir}/{self.id()}.pt"))

    def can_load_from_disk(self, model_dir):
        return os.path.isfile(f"{model_dir}/{self.id()}.pt")
