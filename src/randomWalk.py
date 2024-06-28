from typing import Any, Optional, Union

import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from src.my_get_self_loop_attr import get_self_loop_attr
from typing import Optional, Tuple
from torch_geometric.utils.num_nodes import maybe_num_nodes

import torch
from torch import Tensor

def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask

def add_random_edge(edge_index, p: float, force_undirected: bool = False,
                    num_nodes: Optional[Union[Tuple[int], int]] = None,
                    training: bool = True) -> Tuple[Tensor, Tensor]:

    if p < 0. or p > 1.:
        raise ValueError(f'Ratio of added edges has to be between 0 and 1 '
                         f'(got {p}')
    if force_undirected and isinstance(num_nodes, (tuple, list)):
        raise RuntimeError('`force_undirected` is not supported for'
                           ' heterogeneous graphs')

    device = edge_index.device
    if not training or p == 0.0:
        edge_index_to_add = torch.tensor([[], []], device=device)
        return edge_index, edge_index_to_add

    if not isinstance(num_nodes, (tuple, list)):
        num_nodes = (num_nodes, num_nodes)
    num_src_nodes = maybe_num_nodes(edge_index, num_nodes[0])
    num_dst_nodes = maybe_num_nodes(edge_index, num_nodes[1])

    num_edges_to_add = round(edge_index.size(1) * p)
    row = torch.randint(0, num_src_nodes, size=(num_edges_to_add, ))
    col = torch.randint(0, num_dst_nodes, size=(num_edges_to_add, ))

    if force_undirected:
        mask = row < col
        row, col = row[mask], col[mask]
        row, col = torch.cat([row, col]), torch.cat([col, row])
    edge_index_to_add = torch.stack([row, col], dim=0).to(device)
    edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)
    return edge_index, edge_index_to_add


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

# @functional_transform('add_random_walk_pe')
class AddRandomWalkPE(BaseTransform):
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'rw_pe',
    ):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        edge_index, edge_weight = data.edge_index, data.edge_weight
        edge_index_remove, edge_mask = dropout_edge(edge_index, p=0.1)
        edge_index_add, edge_add = add_random_edge(edge_index, p=0.1, force_undirected=True)
        adj = SparseTensor.from_edge_index(edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes))
        adj_remove = SparseTensor.from_edge_index(edge_index_remove, edge_weight, sparse_sizes=(num_nodes, num_nodes))
        adj_add = SparseTensor.from_edge_index(edge_index_add, edge_weight, sparse_sizes=(num_nodes, num_nodes))
        adj_list = [adj, adj_remove, adj_add]


        noisy_embedding = []
        # Compute D^{-1} A:
        for adj in adj_list:
            deg_inv = 1.0 / adj.sum(dim=1)
            deg_inv[deg_inv == float('inf')] = 0
            adj = adj * deg_inv.view(-1, 1)
            out = adj
            row, col, value = out.coo()
            pe_list = [get_self_loop_attr((row, col), value, num_nodes)]
            for _ in range(self.walk_length - 1):
                out = out @ adj
                row, col, value = out.coo()
                pe_list.append(get_self_loop_attr((row, col), value, num_nodes))
            pe = torch.stack(pe_list, dim=-1)
            noisy_embedding.append(pe)

        if  len(noisy_embedding) == 1:
            pe = noisy_embedding[0]
        else:
            pe = 0.6*noisy_embedding[0] +0.2*noisy_embedding[0]+0.2*noisy_embedding[0]
        data = add_node_attr(data, pe, attr_name=self.attr_name)
        return data
