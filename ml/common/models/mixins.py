import torch
import os


class SaveLoadMixin:
    def save(self, model_dir):
        """Save model to model_dir. Model must implement id() method."""
        torch.save(self.state_dict(), f"{model_dir}/{self.id()}.pt")

    def load(self, model_dir):
        """Load model from model_dir. Model must implement id() method."""
        self.load_state_dict(torch.load(f"{model_dir}/{self.id()}.pt"))

    def can_load(self, model_dir):
        """Return true if saved .pt exists in model_dir. Model must implement id() method."""
        return os.path.isfile(f"{model_dir}/{self.id()}.pt")
