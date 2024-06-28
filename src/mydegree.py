import torch
from typing import Any, Optional

from torch_geometric.utils import degree
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

class MyDegree(BaseTransform):
    r"""Adds the node degree as one hot encodings to the node features.

    Args:
        max_degree (int): Maximum degree.
        in_degree (bool, optional): If set to :obj:`True`, will compute the
            in-degree of nodes instead of the out-degree.
            (default: :obj:`False`)
        cat (bool, optional): Concat node degrees to node features instead
            of replacing them. (default: :obj:`True`)
    """
    def __init__(self, max_degree, attr_name: Optional[str] = 'cent_pe',):
        self.max_degree = max_degree-1
        self.attr_name = attr_name

    def __call__(self, data):
        idx, num_nodes = data.edge_index[1], data.num_nodes
        deg = degree(idx, data.num_nodes, dtype=torch.long)
        my_max = torch.full([num_nodes], self.max_degree)
        deg = torch.where(deg > self.max_degree, my_max, deg)
        deg = torch.t(deg.unsqueeze(0))

        data = add_node_attr(data, deg, attr_name=self.attr_name)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.max_degree})'
