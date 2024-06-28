import os.path as osp
import random

import networkx
import numpy as np
import math
from texttable import Texttable
import networkx as nx
import torch.nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
import ctypes
import numpy as np


INT = ctypes.c_int
PINT = ctypes.POINTER(ctypes.c_int)
PPINT = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))


class myGraph2:
    edge_index = torch.tensor(0)
    x = torch.tensor(0)
    num_nodes = 0
    def __init__(self):     # do nothing.
        num_nodes = 0

class myGraph:
    G = nx.Graph()
    edge_index = torch.tensor(0)
    x = torch.tensor(0)
    num_nodes = 0
    def __init__(self):     # do nothing.
        num_nodes = 0


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

types = ['Ac', 'Ag', 'Al', 'As', 'Au', 'B', 'Bi', 'Br', 'C', 'Ca', 'Cd',
         'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Er', 'F', 'Fe', 'Ga', 'Gd', 'Ge',
         'Hg', 'Ho', 'I', 'Ir', 'K', 'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na',
         'Nb', 'Nd', 'Ni', 'O', 'Os', 'P', 'Pb', 'Pd', 'Pr', 'Pt', 'Re',
         'Rh', 'Ru', 'S', 'Sb', 'Se', 'Si', 'Sm', 'Sn', 'Tb', 'Te', 'Ti',
         'Tl', 'U', 'V', 'W', 'Yb', 'Zn', 'Zr']

if __name__ == '__main__':
    types.sort()
    print(types)

def read_gexf_data(graphname, new_types):
    # new_types = set()
    G = nx.read_gexf(graphname)
    # TODO: Mapping of nodes in `G` to a contiguous number: AIDS数据集已经确定是连续的整数了，这个去掉。
    # mapping = {name: j for j, name in enumerate(G.nodes())}
    # G = nx.relabel_nodes(G, mapping)

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=G.number_of_nodes())

    x = torch.zeros(G.number_of_nodes(), dtype=torch.long)
    for node, info in G.nodes(data=True):
         x[int(node)] = types.index(info['type'])
    x = F.one_hot(x, num_classes=len(types)).to(torch.float)

    return edge_index, x, G.number_of_nodes()

def get_from_all_graphs(all_graphs, id):
    G = all_graphs[id]
    # Mapping of nodes in `G` to a contiguous number:
    mapping = {name: j for j, name in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_index = to_undirected(edge_index, num_nodes=G.number_of_nodes())

    x = torch.zeros(G.number_of_nodes(), dtype=torch.long)
    for node, info in G.nodes(data=True):
        x[int(node)] = types.index(info['type'])

    x = F.one_hot(x, num_classes=len(types)).to(torch.float)
    return edge_index, x, G.number_of_nodes()


def denormalize_ged(g1_nodes, g2_nodes, sim_score):
    """
    Converts normalized similar into ged.
    """
    nged = -math.log(sim_score, math.e)
    return round(nged * (g1_nodes + g2_nodes) / 2)

def normalize_ged(g1_nodes, g2_nodes, ged):
    """
    Converts ged into normalized ged.
    """
    return torch.exp(-1 * torch.tensor (2 * ged/ ( g1_nodes+ g2_nodes)))

def calculate_loss(prediction, target):
    """
    Calculating the squared loss on the normalized GED.
    :param prediction: Predicted log value of GED.
    :param target: Factual log transofmed GED.
    :return score: Squared error.
    """
    prediction = -math.log(prediction)
    target = -math.log(target)
    score = (prediction-target)**2
    return score

def random_assign(row_num):
    res= [[],[]]
    res[0] = [i for i in range(row_num)]
    res[1] = [i for i in range(row_num)]
    random.shuffle(res[1])
    return res



def calculate_prec_at_k(k, prediction, groundtruth):
    """
    Calculating precision at k.
    """
    best_k_pred = prediction.argsort()[-k:]
    best_k_gt = groundtruth.argsort()[-k:]

    return len(set(best_k_pred).intersection(set(best_k_gt))) / k


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """
    temp = prediction.argsort()
    r_prediction = np.empty_like(temp)
    r_prediction[temp] = np.arange(len(prediction))

    temp = target.argsort()
    r_target = np.empty_like(temp)
    r_target[temp] = np.arange(len(target))

    return rank_corr_function(r_prediction, r_target).correlation

def int1ArrayToPointer(arr): #Converts a 2D numpy to ctypes 2D array.
    # Init needed data types
    ARR_DIMX = INT * arr.shape[0]
    arr_ptr = ARR_DIMX()
    for i, val in enumerate(arr):
        arr_ptr[i] = val
    return arr_ptr

def int2ArrayToPointer(arr): #Converts a 2D numpy to ctypes 2D array.
    # Init needed data types
    ARR_DIMX = INT * arr.shape[1]
    ARR_DIMY = PINT * arr.shape[0]
    # Init pointer
    arr_ptr = ARR_DIMY()
    # Fill the 2D ctypes array with values
    for i, row in enumerate(arr):
        arr_ptr[i] = ARR_DIMX()
        for j, val in enumerate(row):
            arr_ptr[i][j] = val
    return arr_ptr



def CT(input): # convert type
    ctypes_map = {int: ctypes.c_int, float: ctypes.c_double, str: ctypes.c_char_p}
    input_type = type(input)
    if input_type is list:
        length = len(input)
        if length == 0:
            print("convert type failed...input is " + input)
            return None
        else:
            arr = (ctypes_map[type(input[0])] * length)()
            for i in range(length):
                arr[i] = bytes(input[i], encoding="utf-8") if (type(input[0]) is str) else input[i]
            return arr
    else:
        if input_type in ctypes_map:
            return ctypes_map[input_type](bytes(input, encoding="utf-8") if type(input) is str else input)
        else:
            print("convert type failed...input is " + input)
            return None

def nx2txt(G: networkx.Graph, id:str, alg:str):  #
    if alg in ['AIDS700nef', 'CANCER']:
        line = "t " + "# " + id + "\n"
        for id, label in G.nodes(data=True):
            line += "v " + str(id) + " " + label['type'] + "\n"
        for (u, v) in G.edges():
            line += "e " + str(u) + " " + str(v) + " " + str(1) + "\n"
        return line
    elif alg in ['IMDBMulti']:
        line = "t " + "# " + id + "\n"
        for id, label in G.nodes(data=True):
            line += "v " + str(id) + " " + str(1) + "\n"
        for (u, v) in G.edges():
            line += "e " + str(u) + " " + str(v) + " " + str(1) + "\n"
        return line
    else:
        return ""

def get_beam_size(n1, n2, e1, e2, dataset:str):
    beam_size = -1
    if dataset in ['AIDS700nef']:
        beam_size = -1
    elif dataset in ['IMDBMulti']:
        if n1+e1 < 40 and n2+e2 < 40 and n1 <= 10 and n2 <= 10:     # 2kb
            beam_size = -1
        elif n1+e1 < 65 and n2+e2 < 65 and n1 <= 13 and n2 <= 13:   # 3kb
            beam_size = 1000
        elif n1+e1 < 200 and n2+e2 < 200 and n1 <= 21 and n2 <= 21: #10kb
            beam_size = 100
        elif e1 > 300 or e2 > 300:
            beam_size = 2
        else:
            beam_size = 10
    elif dataset in ['CANCER']:
        if n1 + e1 < 65 and n2 + e2 < 65 and n1 <= 25 and n2 <= 25:
            beam_size = 1000
        else:
            beam_size = 300

    return beam_size

if __name__ == '__main__':
    t = torch.Tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    c = torch.Tensor([[0.1002, 0.0000, 0.1016, 0.0990, 0.0998, 0.1035, 0.0999, 0.1018, 0.1042,
         0.0000],
        [0.1003, 0.0000, 0.1019, 0.0987, 0.0997, 0.1044, 0.1000, 0.1021, 0.1052,
         0.0000],
        [0.1002, 0.0000, 0.1016, 0.0990, 0.0998, 0.1037, 0.1000, 0.1018, 0.1044,
         0.0000],
        [0.1002, 0.0000, 0.1016, 0.0990, 0.0998, 0.1037, 0.1000, 0.1018, 0.1044,
         0.0000],
        [0.1004, 0.0000, 0.1019, 0.0985, 0.0998, 0.1043, 0.1000, 0.1020, 0.1047,
         0.0000],
        [0.1004, 0.0000, 0.1022, 0.0983, 0.0997, 0.1050, 0.1000, 0.1023, 0.1055,
         0.0000],
        [0.1003, 0.0000, 0.1020, 0.0987, 0.0997, 0.1042, 0.0999, 0.1021, 0.1049,
         0.0000],
        [0.1003, 0.0000, 0.1019, 0.0987, 0.0997, 0.1044, 0.1000, 0.1021, 0.1052,
         0.0000],
        [0.1000, 0.0000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
         0.0000],
        [0.1000, 0.0000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
         0.0000]])
    c.unsqueeze_(0)
    t.unsqueeze_(0)

    y = torch.sum(t, dim=2)
    print(y)