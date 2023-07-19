import torch
import os


def manual_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_dir(file):
    return os.path.dirname(os.path.abspath(file))
