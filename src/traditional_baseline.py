# encoding=utf-8
import time

import os
import glob
import torch
import operator
import scipy.optimize as opt
from lapjv import lapjv
from src.utils import *
from scipy.stats import spearmanr, kendalltau

tm = 0
types_aids = [
    'O', 'S', 'C', 'N', 'Cl', 'Br', 'B', 'Si', 'Hg', 'I', 'Bi', 'P', 'F',
    'Cu', 'Ho', 'Pd', 'Ru', 'Pt', 'Sn', 'Li', 'Ga', 'Tb', 'As', 'Co', 'Pb',
    'Sb', 'Se', 'Ni', 'Te'
]
types_cancer = ['Pt', 'P', 'N', 'Cu', 'Mo', 'Pb', 'F', 'Se', 'As', 'Cl',
            'Ru', 'Ni', 'Sn', 'C', 'Nd', 'Fe', 'Te', 'O', 'B', 'S', 'Co',
            'Zr', 'Na', 'Zn', 'I', 'Er', 'Ti', 'Si', 'Br', 'K']

def cost_matrix_construction(G1, G2, dname:str, current_types:list):
    INF = G1.number_of_nodes() + G1.number_of_edges() + G2.number_of_nodes() + G2.number_of_edges() + 1
    assert (isinstance(G1, nx.Graph))
    assert (isinstance(G2, nx.Graph))
    ns1 = G1.number_of_nodes()
    ns2 = G2.number_of_nodes()
    cost_matrix = np.zeros((ns1 + ns2, ns1 + ns2), dtype=float)


    if dname == 'AIDS700nef':
        label_types = types_aids
    if dname == 'CANCER':
        label_types = types_cancer
    if current_types is not None and len(current_types) > 0:
        label_types = current_types

    if dname in ['AIDS700nef', 'CANCER']:
        node_label = {i: label_types.index(G1.node[i]['type']) for i in G1.node}
        node_label = sorted(node_label.items(), key=operator.itemgetter(0))
        g1_labels = np.array([k[1] for k in node_label])

        node_label = {i: label_types.index(G2.node[i]['type']) for i in G2.node}
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
    if G1.number_of_nodes() == G2.number_of_nodes():
        cost_matrix = cost_matrix[0:G1.number_of_nodes(), 0:G1.number_of_nodes()]
    mapping_str = ""
    can_used_for_AStar = True
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
                mapping_str += "{}|{} ".format(n, col[i])
            else:                       # 表示G1中删除了节点。由于G1为较小的图，此时的节点匹配无法作为A*Lsa的训练集
                node_match[n] = None
                cost +=1
                can_used_for_AStar = False

    for n in G2.node:
        if n not in node_match.values(): cost += 1

    for edge in G1.edges():
        (p, q) = (node_match[edge[0]], node_match[edge[1]])
        if (p, q) in G2.edges():
            common += 1
    cost = cost + G1.number_of_edges() + G2.number_of_edges() - 2 * common
    # generate mapping string
    return cost, can_used_for_AStar, mapping_str

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

def nx2txt(G, id:str, types_of_cancer:set()):  #
    line = "t " + "# " + id + "\n"
    for id, label in G.nodes(data=True):
        line += "v " + str(id) + " " + label['type'] + "\n"
        if types_of_cancer is not  None: types_of_cancer.add(label['type'])
    for (u, v) in G.edges():
        line += "e " + str(u) + " " + str(v) + " " + str(1) + "\n"

    return line

# load mata.so
curPath = os.path.dirname(os.path.dirname(__file__))
so_path = os.path.join(curPath, 'Astar', 'mata.so')
app_astar = ctypes.cdll.LoadLibrary(so_path)  # app_astar: approximate astar
app_astar.ged.restype = ctypes.c_char_p


# the ground truth of graph parir
# input:
#  g1: the small graph, g2: the large graph
# ged search_space time_cost q_id g_id node_matching
# 	e.g., 0 5 2 186 126 0|2 1|1 2|0 3|3 4|4
# output:
#   g1_index g2_index g1_id g2_id small_graph_id small_graph_id ged mapping_res
#   e.g., 0 2 126 186 186 126 0 0|2 1|1 2|0 3|3 4|4
def gen_ground_truth(g1:nx.Graph, g1_id:str, g2:nx.Graph, g2_id:str, beam_size:int, types_of_cancer:set()):
    assert(g1.number_of_nodes() <= g2.number_of_nodes()) # 不管是那种都要保证G1 < G2
    dname = "CANCER"
    res_str = ""
    app_ged = 10000
    execuated_alg = 1

    # the results of A*LSa-beam
    topk = -1
    matching_order = np.array([0, 0, 0])
    matching_nodes = np.array([[0, 0, 0], [0, 0, 0]])
    g1_str = nx2txt(g1, g1_id, types_of_cancer)
    g2_str = nx2txt(g2, g2_id, types_of_cancer)
    astar_out = app_astar.ged(CT(g1_str), CT(g2_str), int1ArrayToPointer(matching_order),
                              int1ArrayToPointer(matching_order), int2ArrayToPointer(matching_nodes),
                              CT(topk), CT(beam_size))
    astar_out = astar_out.decode('ascii').split()
    app_ged = int(astar_out[0])
    res_str = "{} {} {}".format(g1_id, g2_id, astar_out[0])
    for i, as_i in enumerate(astar_out):
        if i > 4: res_str += " " + as_i

    # the results of Hungarian
    alg_type = "hungarian"
    cost_mat = cost_matrix_construction(g1, g2, dname, list(types_of_cancer))
    pre_ged_hun, can_used_for_AStar, mapping_str = bipartite_for_cost_matrix(g1, g2, cost_mat, alg_type, dname)
    if pre_ged_hun < app_ged and can_used_for_AStar is True:
        app_ged = pre_ged_hun
        res_str = "{} {} {}".format(g1_id, g2_id, app_ged)
        res_str += " " + mapping_str
        execuated_alg = 2

    # the results of VJ
    alg_type = "vj"
    cost_mat = cost_matrix_construction(g1, g2, dname, list(types_of_cancer))
    pre_ged_vj, can_used_for_AStar, mapping_str = bipartite_for_cost_matrix(g1, g2, cost_mat, alg_type, dname)
    if pre_ged_vj < app_ged and can_used_for_AStar is True:
        app_ged = pre_ged_vj
        res_str = "{} {} {}".format(g1_id, g2_id, app_ged)
        res_str += " " + mapping_str
        execuated_alg = 3
    return res_str, execuated_alg


def beam(g1:nx.Graph, g1_id:int, g2:nx.Graph, g2_id:int):
    global tm
    g1_str = nx2txt(g1, str(g1_id), types_of_cancer=None)
    g2_str = nx2txt(g2, str(g2_id), types_of_cancer=None)
    topk = -1
    beam_size = -1
    matching_order = np.array([0, 0, 0])
    matching_nodes = np.array([[0, 0, 0], [0, 0, 0]])
    start_t = time.process_time()

    astar_out = app_astar.ged(CT(g1_str), CT(g2_str), int1ArrayToPointer(matching_order),
                                   int1ArrayToPointer(matching_order), int2ArrayToPointer(matching_nodes),
                                   CT(topk), CT(beam_size))
    tm += time.process_time() - start_t
    astar_out = astar_out.decode('utf-8').split()
    return [int(astar_out[0]), int(astar_out[1])]

# aids的三种传统算法求解
# g1_list: train graphs
# g2_list: test graphs
# no_test_index: 560 for aids dataset
def tradition_aids(g1_list:list, g2_list:list, no_test_index:int, alg_type:str):
    dname = "AIDS700nef"
    l1_list, l2_list = [], []
    rho_list, tau_list, prec_at_10_list, prec_at_20_list = [], [], [], []
    acc_num, fea_num, superior_num, cur_idx = 0, 0, 0, 0
    all_space, all_time = 0, 0

    for i, g1 in enumerate(g2_list):
        prediction_row, ground_truth_row = np.zeros(len(g1_list), dtype=float), np.zeros(len(g1_list), dtype=float)
        for j, g2 in enumerate(g1_list):
            cur_idx += 1
            if g1.number_of_nodes() > g2.number_of_nodes():
                small_g = g2
                large_g = g1
            else:
                small_g = g1
                large_g = g2
            start_t = time.process_time()
            if alg_type in ["hungarian", "vj"]:
                cost_mat = cost_matrix_construction(small_g, large_g, dname, current_types=None)
                pre_ged, _, _ = bipartite_for_cost_matrix(small_g, large_g, cost_mat, alg_type, dname)
            elif alg_type in ["beam"]:
                pre_ged, ss = beam(small_g, i, large_g, j)
                all_space += ss
            all_time += time.process_time() - start_t

            v_pre = nor_ged(g1.number_of_nodes(), g2.number_of_nodes(), pre_ged)
            v_gt = nor_ged(g1.number_of_nodes(), g2.number_of_nodes(), gt_ged[j][i + no_test_index].item())
            prediction_row[j] = v_pre
            ground_truth_row[j] = v_gt

            if np.abs(v_pre - v_gt) < 1e-8: acc_num += 1
            if gt_ged[j][i + no_test_index].item() - pre_ged > 1e-8:
                superior_num += 1
            else:
                l1_list.append(np.abs(v_pre-v_gt))
                l2_list.append( (v_pre-v_gt)* (v_pre-v_gt))

        rho_list.append(calculate_ranking_correlation(spearmanr, prediction_row, ground_truth_row))
        tau_list.append(calculate_ranking_correlation(kendalltau, prediction_row, ground_truth_row))
        prec_at_10_list.append(calculate_prec_at_k(10, prediction_row, ground_truth_row))
        prec_at_20_list.append(calculate_prec_at_k(20, prediction_row, ground_truth_row))

    print("mae: " + str(round(np.mean(l1_list), 5)))
    print("mse: " + str(round(np.mean(l2_list), 5)))
    print("fea: " + str(round(1.0)))
    print("acc: " + str(round(acc_num * 1.0 / cur_idx, 5)))
    print("Spearman's_rho: " + str(round(np.nanmean(rho_list), 5)))
    print("Kendall's_tau: " + str(round(np.nanmean(tau_list), 5)))
    print("p@10: " + str(round(np.mean(prec_at_10_list), 5)))
    print("p@20: " + str(round(np.mean(prec_at_20_list), 5)))
    print("search_space: " + str(all_space / cur_idx))
    print("average_time: " + str(all_time / cur_idx))
    print("superior_num: " + str(round(superior_num / cur_idx, 5)))


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

    gt_ged = ori_train.ged
    alg_type = 'beam'
    # MAE loss:0.23941692358541386, in 1.78125 sec
    # alg_type = 'vj'
    # MAE loss:0.28972563661569, in 1.71875 sec

    tradition_aids(g1_list, g2_list, no_test_index=560, alg_type= alg_type)




