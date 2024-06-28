# Implementations of A* Algorithm including A*-Beamsearch and A*-Pathlength, with different lower bounds.
# Reference:
# "Fast Suboptimal Algorithms for the Computation of Graph Edit Distance"
# "Efficient Graph Similarity Search Over Large Graph Databases"
# "Efficient Graph Edit Distance Computation and Verification via Anchor-aware Lower Bound Estimation"
# "Speeding Up Graph Edit Distance Computation through Fast Bipartite Matching"(VJ Algorithm)
# "Approximate graph edit distance computation by means of bipartite graph matching"(Hungarian Algorithm)
# Author: Lei Yang
import sys
from os.path import basename
import networkx as nx
import xml.etree.ElementTree as ET
import torch
import math
import random
import time
import pickle
import numpy as np
from munkres import Munkres
import glob
import os.path as osp
import os
from scipy.stats import spearmanr, kendalltau
from src.utils import calculate_ranking_correlation, calculate_prec_at_k
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx
# from src.traditional.trad_ged import graph_edit_distance as myGED
from random import sample
# Calculate the cost of edit path
def cost_edit_path(edit_path, u, v, lower_bound):
    cost = 0

    source_nodes = []
    target_nodes = []
    nodes_dict = {}
    for operation in edit_path:
        if operation[0] == None:
            cost += 1
            target_nodes.append(operation[1])
        elif operation[1] == None:
            cost += 1
            source_nodes.append(operation[0])
        else:
            if u.nodes[operation[0]]['label'] != v.nodes[operation[1]]['label']:
                cost += 1
            source_nodes.append(operation[0])
            target_nodes.append(operation[1])
        nodes_dict[operation[0]] = operation[1]

    edge_source = u.subgraph(source_nodes).edges()
    edge_target = v.subgraph(target_nodes).edges()

    sum = 0
    for edge in list(edge_source):
        (p, q) = (nodes_dict[edge[0]], nodes_dict[edge[1]])
        if (p, q) in edge_target:
            sum += 1
    cost = cost + len(edge_source) + len(edge_target) - 2 * sum

    if len(lower_bound) == 3 and lower_bound[2] == 'a':
        # Anchor
        anchor_cost = 0
        cross_edge_source = []
        cross_edge_target = []
        cross_edge_source_tmp = set(u.edges(source_nodes))
        for edge in cross_edge_source_tmp:
            if edge[0] not in source_nodes or edge[1] not in source_nodes:
                cross_edge_source.append(edge)
        cross_edge_target_tmp = set(v.edges(target_nodes))
        for edge in cross_edge_target_tmp:
            if edge[0] not in target_nodes or edge[1] not in target_nodes:
                cross_edge_target.append(edge)

        for edge in cross_edge_source:
            (p, q) = (nodes_dict[edge[0]], edge[1])
            if (p, q) in cross_edge_target:
                anchor_cost += 1

        return cost + anchor_cost
    else:
        return cost


# Check unprocessed nodes in graph u and v
def check_unprocessed(u, v, path):
    processed_u = []
    processed_v = []

    for operation in path:
        if operation[0] != None:
            processed_u.append(operation[0])

        if operation[1] != None:
            processed_v.append(operation[1])
    # print(processed_u, processed_v)
    unprocessed_u = set(u.nodes()) - set(processed_u)
    unprocessed_v = set(v.nodes()) - set(processed_v)
    return list(unprocessed_u), list(unprocessed_v)


def list_unprocessed_label(unprocessed_node, u):
    unprocessed_label = []
    for node in unprocessed_node:
        unprocessed_label.append(u.nodes[node]['label'])
    unprocessed_label.sort()
    return unprocessed_label


def transfer_to_torch(unprocessed_u, unprocessed_v, u, v):
    """
    Transferring the data to torch and creating a hash table with the indices, features and target.
    :param data: Data dictionary.
    :return new_data: Dictionary of Torch Tensors.
    """
    global_labels_file = open('AIDS.pkl', 'rb')
    global_labels = pickle.load(global_labels_file)
    superLabel = str(len(global_labels) - 1)
    new_data = dict()
    g1 = u.subgraph(unprocessed_u)
    g2 = v.subgraph(unprocessed_v)
    reorder_u = {val: str(index) for index, val in enumerate(unprocessed_u)}
    g1_tmp = nx.Graph()
    for (val, index) in reorder_u.items():
        g1_tmp.add_node(index, label=val)
    count = len(g1)
    g1_tmp.add_node(str(count), label=superLabel)
    for (i, j) in list(g1.edges()):
        g1_tmp.add_edge(reorder_u[i], reorder_u[j])
    for node in reorder_u.values():
        g1_tmp.add_edge(str(count), node)
    g1 = g1_tmp

    reorder_v = {val: str(index) for index, val in enumerate(unprocessed_v)}
    g2_tmp = nx.Graph()
    for (val, index) in reorder_v.items():
        g2_tmp.add_node(index, label=val)
    count = len(g2)
    g2_tmp.add_node(str(count), label=superLabel)
    for (i, j) in list(g2.edges()):
        g2_tmp.add_edge(reorder_v[i], reorder_v[j])
    for node in reorder_v.values():
        g2_tmp.add_edge(str(count), node)
    g2 = g2_tmp

    edges_1 = [[edge[0], edge[1]] for edge in g1.edges()] + [[edge[1], edge[0]] for edge in g1.edges()]
    edges_2 = [[edge[0], edge[1]] for edge in g2.edges()] + [[edge[1], edge[0]] for edge in g2.edges()]
    edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
    edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)
    label_1 = [str(g1.nodes[node]['label']) for node in g1.nodes()]
    label_2 = [str(g2.nodes[node]['label']) for node in g2.nodes()]
    features_1 = torch.FloatTensor(np.array(
        [[1.0 if global_labels[node] == label_index else 0 for label_index in global_labels.values()] for node in
         label_1]))
    features_2 = torch.FloatTensor(np.array(
        [[1.0 if global_labels[node] == label_index else 0 for label_index in global_labels.values()] for node in
         label_2]))
    new_data["edge_index_1"] = edges_1.cuda()
    new_data["edge_index_2"] = edges_2.cuda()
    new_data["features_1"] = features_1.cuda()
    new_data["features_2"] = features_2.cuda()

    ged = 4  # randomly
    normalized_ged = ged / (0.5 * (len(label_1) + len(label_2)))
    new_data["target"] = torch.from_numpy(np.exp(-normalized_ged).reshape(1, 1)).view(-1).float()
    return new_data


def star_cost(p, q):
    cost = 0
    if p == None:
        cost += 2 * len(q) - 1
        return cost
    if q == None:
        cost += 2 * len(p) - 1
        return cost
    if p[0] != q[0]:
        cost += 1
    if len(p) > 1 and len(q) > 1:
        p[1:].sort()
        q[1:].sort()
        i = 1
        j = 1
        cross_node = 0
        while (i < len(p) and j < len(q)):
            if p[i] == q[j]:
                cross_node += 1
                i += 1
                j += 1
            elif p[i] < q[j]:
                i += 1
            else:
                j += 1
        cost = cost + max(len(p), len(q)) - 1 - cross_node
    cost += abs(len(q) - len(p))
    return cost


def unprocessed_cost(unprocessed_u_set, unprocessed_v_set, u, v):
    global lower_bound
    # print (lower_bound)
    if lower_bound == 'heuristic':
        # heuristic
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            inter_node = set(unprocessed_u).intersection(set(unprocessed_v))
            cost = max(len(unprocessed_u), len(unprocessed_v)) - len(inter_node)
            cost_set.append(cost)
        return cost_set
    elif lower_bound[0:2] == 'LS':
        # LS
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            cross_node = 0
            u_label = list_unprocessed_label(unprocessed_u, u)
            v_label = list_unprocessed_label(unprocessed_v, v)

            i = 0
            j = 0
            while (i < len(u_label) and j < len(v_label)):
                if u_label[i] == v_label[j]:
                    cross_node += 1
                    i += 1
                    j += 1
                elif u_label[i] < v_label[j]:
                    i += 1
                else:
                    j += 1

            node_cost = max(len(unprocessed_u), len(unprocessed_v)) - cross_node
            edge_u = u.subgraph(unprocessed_u).edges()
            edge_v = v.subgraph(unprocessed_v).edges()
            inter_edge = set(edge_u).intersection(set(edge_v))
            edge_cost = max(len(edge_u), len(edge_v)) - len(inter_edge)
            cost = node_cost + edge_cost
            cost_set.append(cost)
        return cost_set
    elif lower_bound == 'Noah':  # and min(len(unprocessed_u),len(unprocessed_v)) > 1: # add terminate condition
        # args = parameter_parser()
        # trainer = GPNTrainer(args)
        # print(trainer.number_of_labels)
        # trainer.model = torch.load('model.pkl')
        # print(trainer.number_of_labels)
        # print(trainer.model.number_labels)
        # trainer.model.eval()
        # model = torch.load('model.pkl', map_location=lambda storage, loc: storage)
        model = torch.load('model.pkl')
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            if unprocessed_u and unprocessed_v:
                data = transfer_to_torch(unprocessed_u, unprocessed_v, u, v)
                prediction = model(data)
                cost = prediction * (
                            max(len(unprocessed_u), len(unprocessed_v)) + max(len(u.subgraph(unprocessed_u).edges()),
                                                                              len(v.subgraph(unprocessed_v).edges())))
                if cost < 3:
                    lower_bound = 'BM'
                cost_set.append(int(cost))
            else:
                cost = max(len(unprocessed_u), len(unprocessed_v))
                cost_set.append(cost)
        # print (cost_set)
        return cost_set

    elif lower_bound == 'BM':
        # BM
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            cost = 0
            u_label = list_unprocessed_label(unprocessed_u, u)
            v_label = list_unprocessed_label(unprocessed_v, v)
            i = 0
            j = 0
            while (i < len(u_label) and j < len(v_label)):
                if u_label[i] == v_label[j] and u.edges(unprocessed_u[i]) == v.edges(unprocessed_v[j]):
                    u_label.pop(i)
                    v_label.pop(j)
                    i += 1
                    j += 1
                elif u_label[i] < v_label[j]:
                    i += 1
                else:
                    j += 1
            i = 0
            j = 0
            while (i < len(u_label) and j < len(v_label)):
                if u_label[i] == v_label[j]:
                    cost += 0.5
                    u_label.pop(i)
                    v_label.pop(j)
                    i += 1
                    j += 1
                elif u_label[i] < v_label[j]:
                    i += 1
                else:
                    j += 1
            cost = cost + max(len(u_label), len(v_label))
            cost_set.append(cost)
        return cost_set
    else:
        # SM
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            stars_u = []
            temp_u = u.subgraph(unprocessed_u)
            for node in unprocessed_u:
                node_list = []
                node_list.append(node)
                for k in temp_u.neighbors(node):
                    node_list.append(k)
                stars_u.append(node_list)

            stars_v = []
            temp_v = v.subgraph(unprocessed_v)
            for node in unprocessed_v:
                node_list = []
                node_list.append(node)
                for k in temp_v.neighbors(node):
                    node_list.append(k)
                stars_v.append(node_list)

            max_degree = 0
            for i in stars_u:
                if len(i) > max_degree:
                    max_degree = len(i)

            for i in stars_v:
                if len(i) > max_degree:
                    max_degree = len(i)
            # Initial cost matrix
            if len(stars_u) > len(stars_v):
                for i in range(len(stars_u) - len(stars_v)):
                    stars_v.append(None)
            if len(stars_u) < len(stars_v):
                for i in range(len(stars_v) - len(stars_u)):
                    stars_u.append(None)
            cost_matrix = []
            for star1 in stars_u:
                cost_tmp = []
                for star2 in stars_v:
                    cost_tmp.append(star_cost(star1, star2))
                cost_matrix.append(cost_tmp)
            if cost_matrix == []:
                cost_set.append(0)
            else:
                m = Munkres()
                indexes = m.compute(cost_matrix)
                cost = 0
                for row, column in indexes:
                    value = cost_matrix[row][column]
                    cost += value
                cost = cost / max(4, max_degree)
                cost_set.append(cost)
        return cost_set


def graph_edit_distance(u, v, lower_bound, beam_size, start_node=None):
    # Partial edit path
    open_set = []
    cost_open_set = []
    partial_cost_set = []
    path_idx_list = []
    time_count = 0.0
    # For each node w in V2, insert the substitution {u1 -> w} into OPEN
    if start_node == None or start_node not in list(u.nodes()):
        u1 = list(u.nodes())[0]  # randomly access a node
    else:
        u1 = start_node
    call_count = 0
    unprocessed_u_set = []
    unprocessed_v_set = []
    for w in list(v.nodes()):
        edit_path = []
        edit_path.append((u1, w))
        unprocessed_u, unprocessed_v = check_unprocessed(u, v, edit_path)
        new_cost = cost_edit_path(edit_path, u, v, lower_bound)
        cost_list = [new_cost]
        unprocessed_u_set.append(unprocessed_u)
        unprocessed_v_set.append(unprocessed_v)
        # new_cost += unprocessed_cost(unprocessed_u, unprocessed_v, u, v)
        call_count += 1
        open_set.append(edit_path)
        partial_cost_set.append(cost_list)
    unprocessed_cost_set = unprocessed_cost(unprocessed_u_set, unprocessed_v_set, u, v)
    start = time.process_time()
    for i in range(len(unprocessed_cost_set)):
        new_cost = unprocessed_cost_set[i] + partial_cost_set[i][0]
        cost_open_set.append(new_cost)
    end = time.process_time()
    time_count = time_count + end - start

    # Insert the deletion {u1 -> none} into OPEN
    edit_path = []
    edit_path.append((u1, None))
    unprocessed_u, unprocessed_v = check_unprocessed(u, v, edit_path)
    new_cost = cost_edit_path(edit_path, u, v, lower_bound)
    cost_list = [new_cost]
    start = time.process_time()
    new_cost_set = unprocessed_cost([unprocessed_u], [unprocessed_v], u, v)
    new_cost += new_cost_set[0]
    end = time.process_time()
    time_count = time_count + end - start
    call_count += 1
    open_set.append(edit_path)
    cost_open_set.append(new_cost)
    partial_cost_set.append(cost_list)

    while cost_open_set:
        if beam_size:
            # BeamSearch
            tmp_path_set = []
            tmp_cost_set = []
            tmp_partial_cost_set = []
            if len(cost_open_set) > beam_size:
                zipped = zip(open_set, cost_open_set, partial_cost_set)
                sort_zipped = sorted(zipped, key=lambda x: x[1])
                result = zip(*sort_zipped)
                open_set, cost_open_set, partial_cost_set = [list(x)[0:beam_size] for x in result]
                # for i in range(beam_size):
                #     path_idx = cost_open_set.index(min(cost_open_set))
                #     if idx_flag == 0:
                #         path_idx_list.append(path_idx)
                #         idx_flag = 1
                #     print (cost_open_set, path_idx)
                #     tmp_path_set.append(open_set.pop(path_idx))
                #     tmp_cost_set.append(cost_open_set.pop(path_idx))
                #     tmp_partial_cost_set.append(partial_cost_set.pop(path_idx))

                # open_set = tmp_path_set
                # cost_open_set = tmp_cost_set
                # partial_cost_set = tmp_partial_cost_set

        # Retrieve minimum-cost partial edit path pmin from OPEN
        # print (cost_open_set)
        path_idx = cost_open_set.index(min(cost_open_set))
        path_idx_list.append(path_idx)
        min_path = open_set.pop(path_idx)
        cost = cost_open_set.pop(path_idx)
        cost_list = partial_cost_set.pop(path_idx)

        # print(len(open_set))
        # Check p_min is a complete edit path
        unprocessed_u, unprocessed_v = check_unprocessed(u, v, min_path)

        # Return if p_min is a complete edit path
        if not unprocessed_u and not unprocessed_v:
            return min_path, cost, cost_list, call_count, time_count, path_idx_list

        else:
            if unprocessed_u:
                u_next = unprocessed_u.pop()
                unprocessed_u_set = []
                unprocessed_v_set = []
                for v_next in unprocessed_v:
                    new_path = min_path.copy()
                    new_path.append((u_next, v_next))
                    unprocessed_u, unprocessed_v = check_unprocessed(u, v, new_path)
                    new_cost = cost_edit_path(new_path, u, v, lower_bound)
                    new_cost_list = cost_list.copy()
                    new_cost_list.append(new_cost)
                    unprocessed_u_set.append(unprocessed_u)
                    unprocessed_v_set.append(unprocessed_v)
                    # new_cost += unprocessed_cost(unprocessed_u, unprocessed_v, u, v)
                    call_count += 1
                    open_set.append(new_path)
                    # cost_open_set.append(new_cost)
                    partial_cost_set.append(new_cost_list)
                start = time.process_time()
                new_cost_set = unprocessed_cost(unprocessed_u_set, unprocessed_v_set, u, v)
                for i in range(len(new_cost_set)):
                    new_cost = new_cost_set[i] + partial_cost_set[i - len(new_cost_set)][-1]
                    cost_open_set.append(new_cost)
                end = time.process_time()
                time_count = time_count + end - start

                new_path = new_path = min_path.copy()
                new_path.append((u_next, None))
                unprocessed_u, unprocessed_v = check_unprocessed(u, v, new_path)
                new_cost = cost_edit_path(new_path, u, v, lower_bound)
                new_cost_list = cost_list.copy()
                new_cost_list.append(new_cost)
                start = time.process_time()
                new_cost_set = unprocessed_cost([unprocessed_u], [unprocessed_v], u, v)
                new_cost += new_cost_set[0]
                end = time.process_time()
                time_count = time_count + end - start
                call_count += 1
                open_set.append(new_path)
                cost_open_set.append(new_cost)
                partial_cost_set.append(new_cost_list)


            else:
                # All nodes in u have been processed, all nodes in v should be Added.
                unprocessed_u_set = []
                unprocessed_v_set = []
                for v_next in unprocessed_v:
                    new_path = min_path.copy()
                    new_path.append((None, v_next))
                    new_cost = cost_edit_path(new_path, u, v, lower_bound)
                    new_cost_list = cost_list.copy()
                    new_cost_list.append(new_cost)
                    unprocessed_u_set.append(unprocessed_u)
                    unprocessed_v_set.append(unprocessed_v)
                    call_count += 1
                    open_set.append(new_path)
                    # cost_open_set.append(new_cost)
                    partial_cost_set.append(new_cost_list)
                start = time.process_time()
                new_cost_set = unprocessed_cost(unprocessed_u_set, unprocessed_v_set, u, v)
                for i in range(len(new_cost_set)):
                    new_cost = new_cost_set[i] + partial_cost_set[i - len(new_cost_set)][-1]
                    cost_open_set.append(new_cost)
                end = time.process_time()
                time_count = time_count + end - start
    return None, None, None, None, None, None


def VJ(g1, g2):
    edit_path = []
    g1_nodes = []
    g2_nodes = []
    for node in g1.nodes():
        g1_nodes.append((node, g1.nodes[node]['label']))
    for node in g2.nodes():
        g2_nodes.append((node, g2.nodes[node]['label']))
    g1_nodes = sorted(g1_nodes, key=lambda x: (x[1], x[0]))
    g2_nodes = sorted(g2_nodes, key=lambda x: (x[1], x[0]))
    g1_nodes_tmp = g1_nodes.copy()
    g2_nodes_tmp = g2_nodes.copy()
    i = 0
    j = 0
    while (i < len(g1_nodes) and j < len(g2_nodes)):
        if g1_nodes[i][1] == g2_nodes[j][1]:
            edit_path.append((g1_nodes[i][0], g2_nodes[j][0]))
            i += 1
            j += 1
            del g1_nodes_tmp[i - len(g1_nodes) - 1]
            del g2_nodes_tmp[j - len(g2_nodes) - 1]
        elif g1_nodes[i][1] > g2_nodes[j][1]:
            j += 1
        else:
            i += 1

    if (len(g1_nodes_tmp) == len(g2_nodes_tmp)):
        for k in range(len(g1_nodes_tmp)):
            edit_path.append((g1_nodes_tmp[k][0], g2_nodes_tmp[k][0]))
    if (len(g1_nodes_tmp) > len(g2_nodes_tmp)):
        for k in range(len(g2_nodes_tmp)):
            edit_path.append((g1_nodes_tmp[k][0], g2_nodes_tmp[k][0]))
        for k in range(len(g2_nodes_tmp), len(g1_nodes_tmp)):
            edit_path.append((g1_nodes_tmp[k][0], None))
    if (len(g1_nodes_tmp) < len(g2_nodes_tmp)):
        for k in range(len(g1_nodes_tmp)):
            edit_path.append((g1_nodes_tmp[k][0], g2_nodes_tmp[k][0]))
        for k in range(len(g1_nodes_tmp), len(g2_nodes_tmp)):
            edit_path.append((None, g2_nodes_tmp[k][0]))
    edit_path_tmp = []
    cost_list = []
    for i in edit_path:
        edit_path_tmp.append(i)
        cost_list.append(cost_edit_path(edit_path_tmp, g1, g2, 'VJ'))
    return edit_path, cost_edit_path(edit_path, g1, g2, 'VJ'), cost_list

class DFS_hungary():
    def __init__(self, g1, g2):
        self.g1, self.g2 = g1, g2
        self.nx = list(self.g1.nodes())
        self.ny = list(self.g2.nodes())
        self.edge = {}
        for node1 in self.g1:
            edge_tmp = {}
            for node2 in self.g2:
                if self.g1.nodes[node1]['label'] == self.g2.nodes[node2]['label']:
                    edge_tmp[node2] = 0
                else:
                    edge_tmp[node2] = 1
            self.edge[node1] = edge_tmp
        self.cx = {}
        for node in self.g1:
            self.cx[node] = -1
        self.cy = {}
        self.visited = {}
        for node in self.g2:
            self.cy[node] = -1
            self.visited[node] = 0
        self.edit_path = []

    def min_cost(self):
        res = 0
        for i in self.nx:
            if self.cx[i] == -1:
                for key in self.ny:
                    self.visited[key] = 0
                res += self.path(i)
        return res, self.edit_path

    def path(self, u):
        for v in self.ny:
            if not (self.edge[u][v]) and (not self.visited[v]):
                self.visited[v] = 1
                if self.cy[v] == -1:
                    self.cx[u] = v
                    self.cy[v] = u
                    self.edit_path.append((u, v))
                    return 0
                else:
                    if (self.cy[v], v) in self.edit_path: self.edit_path.remove((self.cy[v], v)) # modify by junfeng.
                    # self.edit_path.remove((self.cy[v], v))
                    if not (self.path(self.cy[v])):
                        self.cx[u] = v
                        self.cy[v] = u
                        self.edit_path.append((u, v))
                        return 0
        self.edit_path.append((u, None))
        return 1


def Hungarian(g1, g2):
    cost, edit_path = DFS_hungary(g1, g2).min_cost()
    if len(g1.nodes()) < len(g2.nodes()):
        processed = [v[1] for v in edit_path]
        for node in g2.nodes():
            if node not in processed:
                edit_path.append((None, node))
    edit_path_tmp = []
    cost_list = []
    for i in edit_path:
        edit_path_tmp.append(i)
        cost_list.append(cost_edit_path(edit_path_tmp, g1, g2, 'Hungarian'))
    return edit_path, cost_edit_path(edit_path, g1, g2, 'Hungarian'), cost_list


# Load graph from .gxl files
def loadGXL(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    index = 0
    g = nx.Graph(filename=basename(filename), name=root[0].attrib['id'])
    dic = {}  # used to retrieve incident nodes of edges
    for node in root.iter('node'):
        dic[node.attrib['id']] = index
        labels = {}
        for attr in node.iter('attr'):
            labels[attr.attrib['name']] = attr[0].text
        labels['label'] = node.attrib['id']
        g.add_node(index, **labels)
        index += 1

    for edge in root.iter('edge'):
        labels = {}
        for attr in edge.iter('attr'):
            labels[attr.attrib['name']] = attr[0].text
        g.add_edge(dic[edge.attrib['from']], dic[edge.attrib['to']], **labels)
    return g


def main():
    lower_bound_list = ['heuristic', 'LS', 'BM', 'LSa', 'BMa', 'SM', 'Noah']
    # Open beamsearch
    beam_size = 0
    gen_flag = 0
    # global lower_bound_type
    global lower_bound
    # Load data and compute GED
    if len(sys.argv) < 4:
        print("Usage: python ged.py GRAPH_1 GRAPH_2 TYPE BEAMSIZE GEN_FLAG.\nDefault lower bound type is LS")
        # sys.exit(0)
        g1 = nx.read_gexf('2692.gexf')
        g2 = nx.read_gexf('2693.gexf')
        lower_bound = 'LS'
    else:
        # Load data from .gxl files
        # g1 = loadGXL(sys.argv[1])
        # g2 = loadGXL(sys.argv[2])
        # Load data from .gexf files
        g1 = nx.read_gexf(sys.argv[1])
        g2 = nx.read_gexf(sys.argv[2])
        # Choose lower bound type
        lower_bound = sys.argv[3]
        graph_pair = []
        graph_pair.append(sys.argv[1])
        graph_pair.append(sys.argv[2])
        if lower_bound not in lower_bound_list:
            print('No such lower bound, use default lower bound type LS.')
            lower_bound = 'LS'
        if len(sys.argv) == 5:
            beam_size = int(sys.argv[4])
        if len(sys.argv) == 6:
            beam_size = int(sys.argv[4])
            if sys.argv[5] != '0':
                gen_flag = 1

    min_path1, cost1, cost_list1, call_count, time_count, path_idx_list = graph_edit_distance(g1, g2, lower_bound, beam_size)
    t = cost_edit_path(min_path1, g1, g2, lower_bound)
    print("cost:{}, time:{}".format(t , time_count))



    min_path2, cost2, cost_list2 = VJ(g1, g2)
    min_path3, cost3, cost_list3 = Hungarian(g1, g2)
    #min_path3, cost3, cost_list3 = VJ(g1, g2)
    if (cost1 <= cost2 and cost1 <= cost2):
        cost = cost1
        min_path = min_path1
        cost_list = cost_list1
    elif (cost2 < cost1 and cost2 <= cost3):
        cost = cost2
        min_path = min_path2
        cost_list = cost_list2
    else:
        cost = cost3
        min_path = min_path3
        cost_list = cost_list3
    print(cost)
    # print (call_count)
    if gen_flag == 1:
        print(min_path, cost_list)
        # print (call_count)
        # print (path_idx_list)
    # print (cost2)
    # print (min_path2)
    # print (call_count)
    # print ('%.3f' %time_count)
    # print("Minimize path is:", min_path)
    # print("Partial cost list is:", cost_list)
    # print("GED is:", cost)


def load_ground_truth(path, dataset : str):
    print("\nload ground truth.\n")
    if dataset in ['AIDS700nef', 'LINUX', 'ALKANE', 'IMDBMulti']: # torch_geometric datasets
        from torch_geometric.datasets import GEDDataset
        dataset_path = os.path.dirname(os.path.dirname(path))
        ori_train = GEDDataset(dataset_path, dataset, train=True)
        training_graphs = ori_train[:len(ori_train) // 4 * 3]          # // 除 向下取整。 3: 1 : 1
        val_graphs = ori_train[len(ori_train) // 4 * 3:]
        testing_graphs = GEDDataset(dataset_path, dataset, train=False)
    else:
        raise ValueError('Unknown dataset name {}'.format(dataset))

    nged_matrix = training_graphs.norm_ged
    nged_matrix = torch.exp(-nged_matrix)
    return training_graphs, val_graphs, testing_graphs,   nged_matrix


def load_graph_from_raw(path):
    ids = []
    names = glob.glob(osp.join(path, '*.gexf'))
    ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))
    graph_list= []
    for i, idx in enumerate(ids[-1]):
        g = nx.read_gexf(osp.join(path, f'{idx}.gexf'))
        # g = dgl.from_networkx(g)
        graph_list.append(g)
    return graph_list

# len( batch_graph1) == len(batch_graph2)
# batch_graph1， batch_graph2 是list的networkx，
# ground_truth：norm 之后的ged，700x700的矩阵，
# val_size:
# start_idx:表示上一次

def normalize_ged(g1_nodes, g2_nodes, ged):
    return np.exp(-1 * (2 * ged/ ( g1_nodes+ g2_nodes)))

def denormalize_ged(g1_nodes, g2_nodes, sim_score):
    nged = -math.log(sim_score, math.e)
    return (nged * (g1_nodes + g2_nodes) / 2)


def calc_metric_batch_pair(batch_graph1, batch_graph2, gt_list):
    assert len(batch_graph1) == len(batch_graph2)
    res_list = []
    res_metric = dict()
    mae, mse = 0.0, 0.0
    norm_mae, norm_mse = 0.0, 0.0
    acc = 0
    for i in range(len(batch_graph1)):
        t = time.process_time()
        g1 = batch_graph1[i].to_undirected()
        g2 = batch_graph2[i].to_undirected()
        # min_path, cost, cost_list = VJ(g1, g2)

        # hausdorff
        # bipartite

        g1 = dgl.from_networkx(g1)
        g2 = dgl.from_networkx(g2)
        cost, node_mapping, edge_mapping = myGED(g1, g2, algorithm='bipartite' )
        g1size = g1.number_of_nodes()
        g2size = g2.number_of_nodes()
        # g1size = len(g1)
        # g2size = len(g2)
        # min_path1, cost, cost_list1, call_count, time_count, path_idx_list = graph_edit_distance(g1, g2, lower_bound,
        #                                                                                           10)

        norm_pred = normalize_ged(g1size, g2size, cost)
        norm_mae = norm_mae + np.abs( gt_list[i] - norm_pred)
        norm_mse = norm_mse + (gt_list[i] - norm_pred)*(gt_list[i] - norm_pred)
        denorm_gt = denormalize_ged(g1size, g2size, gt_list[i])
        mae = mae + np.abs(cost - denorm_gt)
        mse = mse + (cost - denorm_gt)*(cost - denorm_gt)
        if abs(norm_pred - gt_list[i]) < 1e-2:
            acc = acc + 1
        res_list.append(norm_pred)
        if denorm_gt - 1 < cost:
            pass
        else:
            print("$$$$$$$$$$$$$$$$$$$$$$error$$$$$$$$$$$$$$$$$$$$$$")
        # if norm_pred > gt_list[i]:
            # print("$$$$$$$$$$$$$$$$$$$$$$error$$$$$$$$$$$$$$$$$$$$$$")

    res_list = np.array(res_list)
    gt_list = np.array(gt_list)
    rho = calculate_ranking_correlation(spearmanr, res_list, gt_list)
    tau = calculate_ranking_correlation(kendalltau, res_list, gt_list)
    pat10 = calculate_prec_at_k(10, res_list, gt_list)
    pat20 = calculate_prec_at_k(20, res_list, gt_list)

    acc = acc * 1.0 / len(batch_graph2)

    res_metric['mae'] = mae
    res_metric['mse'] = mse
    res_metric['norm_mae'] = norm_mae
    res_metric['norm_mse'] = norm_mse
    res_metric['acc'] = acc
    res_metric['rho'] = rho
    res_metric['tau'] = tau
    res_metric['pat10'] = pat10
    res_metric['pat20'] = pat20

    return res_metric

def calc_metric_batch_pair2(batch_graph1, batch_graph2, gt_list):



    res_list = []
    res_metric = dict()
    mae, mse = 0.0, 0.0
    norm_mae, norm_mse = 0.0, 0.0
    acc = 0
    for i in range(batch_graph1.num_graphs):
        g1 = to_networkx(batch_graph1[i])
        g2 = to_networkx(batch_graph2[i])

        min_path, cost, cost_list = VJ(g1, g2)

        # hausdorff
        # bipartite
        #cost, node_mapping, edge_mapping = myGED(g1, g2, algorithm='bipartite' )
        # g1size = g1.number_of_nodes()
        # g2size = g2.number_of_nodes()
        g1size = g1.num_nodes
        g2size = g2.num_nodes
        min_path1, cost, cost_list1, call_count, time_count, path_idx_list = graph_edit_distance(g1, g2, lower_bound,
                                                                                                  10)

        norm_pred = normalize_ged(g1size, g2size, cost)
        norm_mae = norm_mae + np.abs( gt_list[i] - norm_pred)
        norm_mse = norm_mse + (gt_list[i] - norm_pred)*(gt_list[i] - norm_pred)
        denorm_gt = denormalize_ged(g1size, g2size, gt_list[i])
        mae = mae + np.abs( cost - denorm_gt )
        mse = mse + (cost - denorm_gt)*(cost - denorm_gt)
        if abs(norm_pred - gt_list[i]) < 1e-2:
            acc = acc + 1
        res_list.append(norm_pred)
        if denorm_gt - 1 < cost:
            pass
        else:

            print("$$$$$$$$$$$$$$$$$$$$$$error$$$$$$$$$$$$$$$$$$$$$$")
        # if norm_pred > gt_list[i]:
            # print("$$$$$$$$$$$$$$$$$$$$$$error$$$$$$$$$$$$$$$$$$$$$$")

    res_list = np.array(res_list)
    gt_list = np.array(gt_list)
    rho = calculate_ranking_correlation(spearmanr, res_list, gt_list)
    tau = calculate_ranking_correlation(kendalltau, res_list, gt_list)
    pat10 = calculate_prec_at_k(10, res_list, gt_list)
    pat20 = calculate_prec_at_k(20, res_list, gt_list)

    acc = acc * 1.0 / len(batch_graph2)

    res_metric['mae'] = mae
    res_metric['mse'] = mse
    res_metric['norm_mae'] = norm_mae
    res_metric['norm_mse'] = norm_mse
    res_metric['acc'] = acc
    res_metric['rho'] = rho
    res_metric['tau'] = tau
    res_metric['pat10'] = pat10
    res_metric['pat20'] = pat20

    return res_metric


def score(train_path:str, test_path:str, dataset:str, batch_size:int):
    training_graphs = load_graph_from_raw(train_path)       # networkx of graphs
    testing_graphs = load_graph_from_raw(test_path)         # networkx of graphs
    pyg_train, pyg_val, pyg_test, ground_truth = load_ground_truth(os.path.dirname(train_path), dataset)
    mae, mse, norm_mae, norm_mse, acc = 0.0, 0.0, 0.0, 0.0, 0.0
    rho, tau, pat10, pat20 = 0.0, 0.0, 0.0, 0.0
    itera_times = 0
    for i, g1 in enumerate(pyg_test):
        if len(testing_graphs[i]) != g1.num_nodes:
            print("#########error#########")

    for i, g1 in enumerate(pyg_train):
        if len(training_graphs[i]) != g1.num_nodes:
            print("#########error#########")


    start_t = time.process_time()
    for i, g1 in enumerate(pyg_test):
        data_list_1 = list([g1]* len(pyg_train))
        nx_data_list1 = list([testing_graphs[i]] * len(pyg_train))

        for start_idx in range(0, len(pyg_train), batch_size):
            graph_list1 = data_list_1[start_idx:start_idx+batch_size]
            graph_list2 = pyg_train[start_idx: start_idx+batch_size]
            source_batch = Batch.from_data_list(graph_list1)  # 将一批图看做是一个大图的对象。
            target_batch = Batch.from_data_list(graph_list2)
            nor_ged = ground_truth[source_batch["i"].reshape(-1).tolist(), target_batch["i"].reshape(-1).tolist()].tolist()
            nor_ged = torch.from_numpy(np.array(nor_ged)).view(-1).float()

            nx_graph_list1 = nx_data_list1[start_idx:start_idx+batch_size]
            nx_graph_list2 = training_graphs[start_idx:start_idx+batch_size]
            # gt_list = ground_truth[i + train_size, start_idx:start_idx + len(batch_graph1)]
            res_metric = calc_metric_batch_pair(nx_graph_list1, nx_graph_list2, nor_ged)
            mae += res_metric['mae']
            mse += res_metric['mse']
            norm_mae += res_metric['norm_mae']
            norm_mse += res_metric['norm_mse']
            acc += res_metric['acc']
            rho += res_metric['rho']
            tau += res_metric['tau']
            pat10 += res_metric['pat10']
            pat20 += res_metric['pat20']
            itera_times += 1

    time_cost = time.process_time() - start_t
    print("mae: " + str(round(mae / itera_times, 5)))
    print("mse: " + str(round(mse /itera_times, 5)))
    print("norm_mae: " + str(round(norm_mae / itera_times, 5)))
    print("norm_mse: " + str(round(norm_mse /itera_times, 5)))
    print("fea: " + str(round(1.0)))
    print("acc: " + str(round( acc / itera_times, 5)))
    print("Spearman's rho: " + str(round(rho /itera_times, 5)))
    print("Kendall's tau: " + str(round(tau /itera_times, 5)))
    print("p@10: " + str(round(pat10/itera_times, 5 )))
    print("p@20: " + str(round(pat20/itera_times, 5)))
    print("search space:" + str(0))
    print("average time(sec.):" + str( round(time_cost/ (len(training_graphs) * len(testing_graphs)), 5)))


if __name__ == "__main__":
    global lower_bound
    lower_bound = "BM"
    # path = "D:/datasets/GED/AIDS700nef/test/"
    # lower_bound = "heuristic"
    # beam_size = 10
    # names = glob.glob(osp.join(path, '*.gexf'))
    # all_time = 0
    # for i in range(100):
    #     idlist = sample(names, 2)
    #     g1 = nx.read_gexf(osp.join(path, f'{idlist[0]}'))
    #     g2 = nx.read_gexf(osp.join(path, f'{idlist[1]}'))
    #     t = time.process_time()
    #     min_path2, cost2, cost_list2 = VJ(g1, g2)
    #     min_path1, cost1, cost_list1, call_count, time_count, path_idx_list = graph_edit_distance(g1, g2, lower_bound, beam_size)
    #     all_time += time.process_time() - t
    #     cost1 = cost_edit_path(min_path1, g1, g2, lower_bound)
    #     min_path2, cost2, cost_list2 = VJ(g1, g2)
    #     min_path3, cost3, cost_list3 = Hungarian(g1, g2)
    # print(all_time)


    path = "D:/workspace/GED/ourGED/datasets/{}/raw/{}/"
    dataset= "AIDS700nef"
    path = path.format(dataset, dataset)

    batch_size=128
    score(os.path.join(path, "train"), os.path.join(path, "test"), dataset, batch_size)

    # g1 = nx.read_gexf("D:/datasets/GED/Syn/train/0_1.gexf")
    # g2 = nx.read_gexf("D:/datasets/GED/Syn/train/0_2.gexf")
    # # min_path2, cost2, cost_list2 = Hungarian(g1.to_directed(), g2.to_directed())
    # G1 = dgl.from_networkx(g1)
    # G2 = dgl.from_networkx(g2)
    # a,b,c = myGED(G1, G2)
    # print(a)

""" 
LSa, LS 算法存在bug。 
"""