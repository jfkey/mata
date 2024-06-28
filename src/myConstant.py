import torch
from typing import Any, Optional

from torch_geometric.utils import degree
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

class MyConstanct(BaseTransform):

    def __init__(self, value: float = 1.0, cat: bool = True):
        self.value = value
        self.cat = cat

    def __call__(self, data):
        val = torch.full([data.num_nodes], self.value)
        val = torch.t(val.unsqueeze(0))
        if hasattr(data, 'x') and self.cat:
            pass
        else:
            data.x = val

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value})'
