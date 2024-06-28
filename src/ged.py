# Running example: python src/ged.py D:/datasets/GED/Syn/test/0_1.gexf D:/datasets/GED/Syn/test/0_2.gexf  BM 10 0
# Implementations of A* Algorithm including A*-Beamsearch and A*-Pathlength, with different lower bounds.
# Reference:
# "Fast Suboptimal Algorithms for the Computation of Graph Edit Distance"
# "Efficient Graph Similarity Search Over Large Graph Databases" (15 TKDE, filter. )
# "Efficient Graph Edit Distance Computation and Verification via Anchor-aware Lower Bound Estimation" (17 arxiv, LS, LSa, BM, BMa)
# "Speeding Up Graph Edit Distance Computation through Fast Bipartite Matching"(VJ Algorithm)
# "Approximate graph edit distance computation by means of bipartite graph matching"(Hungarian Algorithm)
# Author: Lei Yang
import sys
import os
from os.path import basename
import networkx as nx
import xml.etree.ElementTree as ET
import torch
import math
import random
import time
import pickle

import numpy as np
# from torch_geometric.nn import GCNConv
# from parser import parameter_parser
# from gpn import GPNTrainer
from munkres import Munkres

check_unprocessed_size = 0

# Calculate the cost of edit path
'''
给定已匹配的部分节点 edit_path，源图u 和目标图v，A star算法的lower_bound，
返回 已匹配子图的编辑代价（节点，和边）
1. 计算已匹配节点的编辑代价，
2. 计算已匹配节点构建出的induce graph中边的编辑代价。
'''
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
'''
根据A star执行过程中已有的节点匹配对，返回源图和目标图未匹配的节点列表。
'''


def check_unprocessed(u, v, path):
    global check_unprocessed_size
    check_unprocessed_size += 1
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


'''
输入：
unprocessed_u_set: 源图未匹配节点的集合。 
unprocessed_v_set：目标图未匹配节点的集合
返回：
cost_set：源图和目标图未匹配子图的 预计编辑代价的集合，即h(p)集合
'''


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
        # model = torch.load('model.pkl')
        cost_set = []
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            if unprocessed_u and unprocessed_v:
                data = transfer_to_torch(unprocessed_u, unprocessed_v, u, v)
                model = torch.load('model.pkl')
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
    open_set = []  # 存放在A star执行过程中，所有的部分编辑路径。
    cost_open_set = []  # 存放open_set 中对应的编辑代价。即每个编辑路径的g(p)+h(p)
    partial_cost_set = []  # 存放open_set 中对应的已匹配部分的编辑代价，即g(p)
    path_idx_list = []
    time_count = 0.0
    # For each node w in V2, insert the substitution {u1 -> w} into OPEN
    if start_node == None or start_node not in list(u.nodes()):
        u1 = list(u.nodes())[0]  # randomly access a node
    else:
        u1 = start_node
    call_count = 0
    unprocessed_u_set = []  # 存放在A star执行过程中，源图中所有分支的未匹配的节点
    unprocessed_v_set = []  # 存放在A star执行过程中，目标图中所有分支的未匹配的节点
    # 先从源图中选择一个节点u，遍历所有可能的编辑操作（替换操作）
    for w in list(v.nodes()):
        edit_path = []  # 存放A star算法执行过程中，源图和目标图节点匹配对的序列。
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
    start = time.perf_counter()
    for i in range(len(unprocessed_cost_set)):
        new_cost = unprocessed_cost_set[i] + partial_cost_set[i][0]
        cost_open_set.append(new_cost)
    end = time.perf_counter()
    time_count = time_count + end - start

    # Insert the deletion {u1 -> none} into OPEN
    #  从源图中选择一个节点u，考虑节点u的删除操作。
    edit_path = []
    edit_path.append((u1, None))
    unprocessed_u, unprocessed_v = check_unprocessed(u, v, edit_path)
    new_cost = cost_edit_path(edit_path, u, v, lower_bound)
    cost_list = [new_cost]
    start = time.perf_counter()
    new_cost_set = unprocessed_cost([unprocessed_u], [unprocessed_v], u, v)
    new_cost += new_cost_set[0]
    end = time.perf_counter()
    time_count = time_count + end - start
    call_count += 1
    open_set.append(edit_path)
    cost_open_set.append(new_cost)
    partial_cost_set.append(cost_list)

    while cost_open_set:  # 考虑所有的候选集
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
        # 找到最小的编辑代价，并将该编辑代价从open_set中删除
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
            if unprocessed_u:  # 源图未匹配的节点不为空，按照前面类似的逻辑。
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
                start = time.perf_counter()
                new_cost_set = unprocessed_cost(unprocessed_u_set, unprocessed_v_set, u, v)
                for i in range(len(new_cost_set)):
                    new_cost = new_cost_set[i] + partial_cost_set[i - len(new_cost_set)][-1]
                    cost_open_set.append(new_cost)
                end = time.perf_counter()
                time_count = time_count + end - start

                new_path = new_path = min_path.copy()
                new_path.append((u_next, None))
                unprocessed_u, unprocessed_v = check_unprocessed(u, v, new_path)
                new_cost = cost_edit_path(new_path, u, v, lower_bound)
                new_cost_list = cost_list.copy()
                new_cost_list.append(new_cost)
                start = time.perf_counter()
                new_cost_set = unprocessed_cost([unprocessed_u], [unprocessed_v], u, v)
                new_cost += new_cost_set[0]
                end = time.perf_counter()
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
                start = time.perf_counter()
                new_cost_set = unprocessed_cost(unprocessed_u_set, unprocessed_v_set, u, v)
                for i in range(len(new_cost_set)):
                    new_cost = new_cost_set[i] + partial_cost_set[i - len(new_cost_set)][-1]
                    cost_open_set.append(new_cost)
                end = time.perf_counter()
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
                    self.edit_path.remove((self.cy[v], v))
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
    start_time = time.process_time()
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

    cost = cost_edit_path(min_path1, g1, g2, lower_bound)
    print("real cost:", cost)

    min_path2, cost2, cost_list2 = VJ(g1, g2)
    # min_path3, cost3, cost_list3 = Hungarian(g1,g2)
    min_path3, cost3, cost_list3 = VJ(g1, g2)
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
    print(time.process_time() - start_time)



if __name__ == "__main__":
    main()
    print("(check_unprocessed_size) explore size: ", check_unprocessed_size)
