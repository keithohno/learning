import torch

from common.models.mixins import SaveLoadMixin


class Model(torch.nn.Module, SaveLoadMixin):
    def __init__(self):
        super().__init__()

    def id(self):
        raise NotImplementedError
