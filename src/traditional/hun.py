import time

import torch
import scipy.optimize as opt
from lapjv import lapjv
import numpy as np
import networkx as nx
import os.path as osp
import operator
import random
import os
import glob
from multiprocessing import Pool
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree
from torch_geometric.transforms import OneHotDegree, ToUndirected
import torch_geometric as pyg
from src.utils import *
from torch_geometric.data import Data

types_aids = [
    'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
    'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
    'Sb', 'Se', 'Ni', 'Te'
]
types_cancer = ['Pt', 'P', 'N', 'Cu', 'Mo', 'Pb', 'F', 'Se', 'As', 'Cl',
            'Ru', 'Ni', 'Sn', 'C', 'Nd', 'Fe', 'Te', 'O', 'B', 'S', 'Co',
            'Zr', 'Na', 'Zn', 'I', 'Er', 'Ti', 'Si', 'Br', 'K']

def cost_matrix_construction(G1, G2, dname:str):
    INF = G1.number_of_nodes() + G1.number_of_edges() + G2.number_of_nodes() + G2.number_of_edges() + 1
    assert (isinstance(G1, nx.Graph))
    assert (isinstance(G2, nx.Graph))
    ns1 = G1.number_of_nodes()
    ns2 = G2.number_of_nodes()
    cost_matrix = np.zeros((ns1 + ns2, ns1 + ns2), dtype=float)

    if dname in ['AIDS700nef', 'CANCER']:
        node_label = {i: types_aids.index(G1.node[i]['type']) for i in G1.node}
        node_label = sorted(node_label.items(), key=operator.itemgetter(0))
        g1_labels = np.array([k[1] for k in node_label])

        node_label = {i: types_aids.index(G2.node[i]['type']) for i in G2.node}
        node_label = sorted(node_label.items(), key=operator.itemgetter(0))
        g2_labels = np.array([k[1] for k in node_label])

        g1_labels = np.expand_dims(g1_labels, axis=1)
        g2_labels = np.expand_dims(g2_labels, axis=0)
        label_substitution_cost = np.abs(g1_labels - g2_labels)
        label_substitution_cost[np.nonzero(label_substitution_cost)] = 1
        cost_matrix[0:ns1, 0:ns2] = label_substitution_cost

    cost_matrix[0:ns1, ns2:ns1+ns2] = np.array([1 if i == j else INF for i in range(ns1) for j in range(ns1) ]).reshape(ns1, ns1)
    cost_matrix[ns1:ns1+ns2, 0:ns2] = np.array([1 if i == j else INF for i in range(ns2) for j in range(ns2) ]).reshape(ns2, ns2)


    # do not consider node and edge labels, i.e., the cost of edge Eui equals to the degree difference
    g1_degree = np.array([G1.degree(n) for n in range(ns1)], dtype=int)
    g2_degree = np.array([G2.degree(n) for n in range(ns2)], dtype=int)
    g1_degree = np.expand_dims(g1_degree, axis=1)
    g2_degree = np.expand_dims(g2_degree, axis=0)
    degree_substitution_cost = np.abs(g1_degree - g2_degree)
    cost_matrix[0:ns1, 0:ns2] += degree_substitution_cost
    return cost_matrix

def comp_ged(_x, _k):
    if len(_x.shape) == 3 and len(_k.shape) == 3:
        _batch = _x.shape[0]
        return torch.bmm(torch.bmm(_x.reshape(_batch, 1, -1), _k), _x.reshape(_batch, -1, 1)).view(_batch)
    elif len(_x.shape) == 2 and len(_k.shape) == 2:
        return torch.mm(torch.mm(_x.reshape(1, -1), _k), _x.reshape( -1, 1)).view(1)
    else:
        raise ValueError('Input dimensions not supported.')

def bipartite_for_cost_matrix(G1, G2, cost_matrix, alg_type:str, dname:str):
    if alg_type == 'hungarian':
        row, col = opt.linear_sum_assignment(cost_matrix)
    elif alg_type == 'vj':
        row, col, _ = lapjv(cost_matrix)
    node_match = {}
    cost = 0
    common = 0
    for i, n in enumerate(row):
        if n < G1.number_of_nodes():
            if col[i] < G2.number_of_nodes():
                node_match[n] = col[i]
                if G1.node[n]['type'] != G2.node[col[i]]['type'] and dname in ['AIDS700nef', 'CANCER']:
                    cost += 1
            else:
                node_match[n] = None
                cost +=1
    for n in G2.node:
        if n not in node_match.values(): cost += 1

    for edge in G1.edges():
        (p, q) = (node_match[edge[0]], node_match[edge[1]])
        if (p, q) in G2.edges():
            common += 1
    cost = cost + G1.number_of_edges() + G2.number_of_edges() - 2 * common
    return cost


def load_nx_list(path):
    ids = []
    names = glob.glob(osp.join(path, '*.gexf'))
    # Get sorted graph IDs given filename: 123.gexf -> 123
    ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))
    data_list = []
    for i, idx in enumerate(ids[-1]):
        i = i if len(ids) == 1 else i + len(ids[0])
        # Reading the raw `*.gexf` graph:
        G = nx.read_gexf(osp.join(path, f'{idx}.gexf'))
        # Mapping of nodes in `G` to a contiguous number:
        mapping = {name: j for j, name in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        data_list.append(G)
    return data_list

def nor_ged(g1_nodes, g2_nodes, ged):
    return np.exp(-1 * (2 * ged/ ( g1_nodes+ g2_nodes)))


if __name__ == '__main__':
    dname = "AIDS700nef"
    # dname = "IMDBMulti"
    path_train = "D:/workspace/GED/ourGED/datasets/{}/raw/{}/train/".format(dname, dname)
    path_test = "D:/workspace/GED/ourGED/datasets/{}/raw/{}/test/".format(dname, dname)

    # path = "C:/Users/liujf/Desktop/test/"
    # G1 = nx.read_gexf(osp.join(path, f'{4}.gexf'))
    # G2 = nx.read_gexf(osp.join(path, f'{21}.gexf'))

    g1_list = load_nx_list(path_train)
    g2_list = load_nx_list(path_test)
    from torch_geometric.datasets import GEDDataset

    ori_train = GEDDataset('D:/workspace/GED/ourGED/datasets/{}'.format(dname), dname, train=True)
    ori_test = GEDDataset('D:/workspace/GED/ourGED/datasets/{}'.format(dname), dname, train=False)

    max_degree = 100


    gt_ged = ori_train.ged
    alg_type = 'hungarian'
    l1_list = []
    i, j = 0, 0
    g1_list = g1_list[0:100]
    g2_list = g2_list[0:10]
    t = time.process_time()
    for g1 in g1_list:
        i += 1
        j = 0
        for g2 in g2_list:
            j += 1
            cost_mat = cost_matrix_construction(g1, g2, dname)
            pre_ged = bipartite_for_cost_matrix(g1, g2, cost_mat, alg_type, dname)
            v1 = nor_ged(g1.number_of_nodes(), g2.number_of_nodes(), pre_ged)
            v2 = nor_ged(g1.number_of_nodes(), g2.number_of_nodes(), gt_ged[i-1][j-1 + 560].item())
            l1_list.append( np.abs(v1-v2) )

    print("MAE loss:{}, in {} sec".format( np.mean(np.array(l1_list)), time.process_time()-t ))



