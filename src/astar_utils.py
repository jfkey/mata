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
from torch_geometric.utils import subgraph
from src import global_var as glo_dict

import numpy as np
# from torch_geometric.nn import GCNConv
# from parser import parameter_parser
# from gpn import GPNTrainer
from munkres import Munkres

check_unprocessed_size = 0
glo_dict._init()
glo_dict.set_value('cost_edit_path', 0.0)
glo_dict.set_value('check_unprocessed', 0.0)
glo_dict.set_value('unprocessed_cost', 0.0)
glo_dict.set_value('net_prediction', 0.0)

# Calculate the cost of edit path
'''
给定已匹配的部分节点 edit_path，源图u 和目标图v，A star算法的lower_bound，
返回 已匹配子图的编辑代价（节点，和边）
1. 计算已匹配节点的编辑代价，
2. 计算已匹配节点构建出的induce graph中边的编辑代价。
'''


def cost_edit_path(edit_path, u, v, lower_bound):
    start = time.perf_counter()
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
            if u.nodes[operation[0]]['type'] != v.nodes[operation[1]]['type']:
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

    tmp = glo_dict.get_value('cost_edit_path')
    glo_dict.set_value('cost_edit_path', tmp + time.perf_counter() - start)
    return cost






# Check unprocessed nodes in graph u and v
'''
根据A star执行过程中已有的节点匹配对，返回源图和目标图未匹配的节点列表。
'''
def check_unprocessed(u, v, path):
    start = time.perf_counter()
    global check_unprocessed_size
    check_unprocessed_size += 1
    processed_u = []
    processed_v = []

    for operation in path:
        if operation[0] != None:
            processed_u.append(operation[0])

        if operation[1] != None:
            processed_v.append(operation[1])
    unprocessed_u = set(u.nodes()) - set(processed_u)
    unprocessed_v = set(v.nodes()) - set(processed_v)
    # 更新当前函数的使用时间
    tmp = glo_dict.get_value('check_unprocessed')
    glo_dict.set_value('check_unprocessed', tmp + time.perf_counter() - start)
    return list(unprocessed_u), list(unprocessed_v)


def list_unprocessed_label(unprocessed_node, u):
    unprocessed_label = []
    for node in unprocessed_node:
        unprocessed_label.append(u.nodes[node]['type'])
    unprocessed_label.sort()
    return unprocessed_label


'''
输入：
unprocessed_u_set: 源图未匹配节点的集合。 
unprocessed_v_set：目标图未匹配节点的集合
返回：
cost_set：源图和目标图未匹配子图的 预计编辑代价的集合，即h(p)集合
'''
def unprocessed_cost(lower_bound, net_prediction, edge_index_1, edge_index_2, feature_1, feature_2, unprocessed_u_set, unprocessed_v_set, u, v):
    start = time.perf_counter()
    cost_set = []
    if lower_bound == 'Noah':
        for i in range(len(unprocessed_u_set)):
            unprocessed_u = unprocessed_u_set[i]
            unprocessed_v = unprocessed_v_set[i]
            if unprocessed_u and unprocessed_v:
                edge_attr_1, edge_attr_2 = None, None
                tmp_edge_index_1, edge_attr_1 = subgraph(unprocessed_u, edge_index_1, edge_attr_1, relabel_nodes=True, num_nodes=feature_1.shape[0])
                tmp_edge_index_2, edge_attr_2 = subgraph(unprocessed_v, edge_index_2, edge_attr_2, relabel_nodes=True, num_nodes=feature_2.shape[0])
                tmp_feature_1 = feature_1[unprocessed_u]
                tmp_feature_2 = feature_2[unprocessed_v]

                cost = net_prediction(tmp_edge_index_1, tmp_edge_index_2, tmp_feature_1, tmp_feature_2)
                cost_set.append(cost)
            else:
                cost = max(len(unprocessed_u), len(unprocessed_v))
                cost_set.append(cost)
    elif lower_bound == 'LS':
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
    tmp = glo_dict.get_value('unprocessed_cost')
    glo_dict.set_value('unprocessed_cost', tmp + time.perf_counter() - start)
    return cost_set


def graph_edit_distance(u, v, matching_nodes, net_prediction, edge_index_1, edge_index_2, feature_1, feature_2,
                        lower_bound='Noah', beam_size=100, start_node=None):
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
    # for w in list(v.nodes()):
    for w in matching_nodes[u1]:
        edit_path = []  # 存放A star算法执行过程中，源图和目标图节点匹配对的序列。
        edit_path.append((u1, w))
        unprocessed_u, unprocessed_v = check_unprocessed(u, v, edit_path)
        new_cost = cost_edit_path(edit_path, u, v, lower_bound)
        cost_list = [new_cost]
        unprocessed_u_set.append(unprocessed_u)
        unprocessed_v_set.append(unprocessed_v)
        call_count += 1
        open_set.append(edit_path)
        partial_cost_set.append(cost_list)
    unprocessed_cost_set = unprocessed_cost(lower_bound, net_prediction, edge_index_1, edge_index_2, feature_1, feature_2, unprocessed_u_set, unprocessed_v_set, u, v)
    start = time.perf_counter()
    for i in range(len(unprocessed_cost_set)):
        new_cost = unprocessed_cost_set[i] + partial_cost_set[i][0]
        cost_open_set.append(new_cost)
    end = time.perf_counter()
    time_count = time_count + end - start



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
                for v_next in unprocessed_v :   # TODO 增加 matching pairs
                    if v_next in matching_nodes[u_next]:
                        new_path = min_path.copy()
                        new_path.append((u_next, v_next))
                        unprocessed_u, unprocessed_v = check_unprocessed(u, v, new_path)
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
                new_cost_set = unprocessed_cost(lower_bound, net_prediction, edge_index_1, edge_index_2, feature_1, feature_2, unprocessed_u_set, unprocessed_v_set, u, v)
                for i in range(len(new_cost_set)):
                    new_cost = new_cost_set[i] + partial_cost_set[i - len(new_cost_set)][-1]
                    cost_open_set.append(new_cost)
                end = time.perf_counter()
                time_count = time_count + end - start

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
                new_cost_set = unprocessed_cost(lower_bound, net_prediction, edge_index_1, edge_index_2, feature_1, feature_2, unprocessed_u_set, unprocessed_v_set, u, v)
                for i in range(len(new_cost_set)):
                    new_cost = new_cost_set[i] + partial_cost_set[i - len(new_cost_set)][-1]
                    cost_open_set.append(new_cost)
                end = time.perf_counter()
                time_count = time_count + end - start
    return None, None, None, None, None, None



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

    # min_path1, cost1, cost_list1, call_count, time_count, path_idx_list = graph_edit_distance(g1, g2, lower_bound, beam_size, str(start_node[0]))
    min_path1, cost1, cost_list1, call_count, time_count, path_idx_list = graph_edit_distance(g1, g2, lower_bound,
                                                                                              beam_size)



