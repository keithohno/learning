import torch
import os


def manual_seed(seed):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    """Get device for training (cuda/mps if available)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_dir(file):
    """
    Get directory of file.

    Usage:
        get_dir(__file__) returns the directory of the file from which it is called.
    """
    return os.path.dirname(os.path.abspath(file))
