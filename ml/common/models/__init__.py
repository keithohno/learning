import torch

from common.models.mixins import SaveLoadMixin
from common.utils import manual_seed


class Model(torch.nn.Module, SaveLoadMixin):
    def __init__(self, seed=23):
        super().__init__()
        manual_seed(seed)

    def id(self):
        raise NotImplementedError
