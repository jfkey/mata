# coding=utf8
import os
import operator
import re
from tqdm import tqdm
from src.utils import *
from src.traditional_baseline import gen_ground_truth
import glob
import torch

#################注意 这里和aids，imdb生成的network不同，这里不需要对节点进行relabel。################
def read_all_nx_graphs(r_path):
    import networkx as nx
    ids, Ns = [], []
    names = glob.glob(osp.join(r_path, '*.gexf'))
    ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))
    data_list = []
    for i, idx in enumerate(ids[-1]):
        i = i if len(ids) == 1 else i + len(ids[0])
        # Reading the raw `*.gexf` graph:
        G = nx.read_gexf(osp.join(r_path, f'{idx}.gexf'))           # 由于Cancer中的图中的节点已经是连续的了，所以不需要进行relabel操作
        data_list.append(G)
    return data_list
#################注意 这里和aids，imdb生成的network不同，这里不需要对节点进行relabel。################
def read_all_nx_graphs_imdb(r_path):
    import networkx as nx
    ids, Ns = [], []
    names = glob.glob(osp.join(r_path, '*.gexf'))
    ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))
    data_list = []
    for i, idx in enumerate(ids[-1]):
        i = i if len(ids) == 1 else i + len(ids[0])
        # Reading the raw `*.gexf` graph:
        G = nx.read_gexf(osp.join(r_path, f'{idx}.gexf'))
        mapping = {name: j for j, name in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        Ns.append(G.number_of_nodes())

        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(G)
    return data_list


def load_labels(label_path):
    f = open(label_path, 'r')
    line = f.readline()
    label_list = []
    while line is not None and line != '':
        item = dict()
        line = line.split()
        item['rowid'] = int(line[0])
        item['colid'] = int(line[1])
        item['rgid'] = int(line[2])
        item['cgid'] = int(line[3])
        item['g1'] = int(line[4])
        item['g2'] = int(line[5])
        item['ged'] = int(line[6])
        # map = dict()  # key:g1 nodes, value: g2 nodes
        int_map = []
        for str in line:
            str_arr = str.split("|")
            if (len(str_arr)) > 1 and str_arr[0] != '-1':
                int_map.append(int(str_arr[1]))
        item['int_map'] = int_map
        label_list.append(item)
        line = f.readline()
    f.close()
    return label_list

def load_labels_imdb(label_path):
    f = open(label_path, 'r')
    line = f.readline()
    label_list = []
    while line is not None and line != '':
        item = dict()
        line = line.split()
        item['rowid'] = int(line[0])
        item['colid'] = int(line[1])
        item['ged'] = int(line[2])
        # map = dict()  # key:g1 nodes, value: g2 nodes
        int_map = []
        for str in line:
            str_arr = str.split("|")
            if (len(str_arr)) > 1 and str_arr[0] != '-1':
                int_map.append(int(str_arr[1]))
        item['int_map'] = int_map
        label_list.append(item)
        line = f.readline()
    f.close()
    return label_list


def nx2txt(G, id:str, types_of_cancer:set()):  #
    line = "t " + "# " + id + "\n"
    for id, label in G.nodes(data=True):
        line += "v " + str(id) + " " + label['type'] + "\n"
        if types_of_cancer is not  None: types_of_cancer.add(label['type'])
    for (u, v) in G.edges():
        line += "e " + str(u) + " " + str(v) + " " + str(1) + "\n"
    return line

def nx2txt_imdb(G, id:str ):  #
    line = "t " + "# " + id + "\n"
    for id, label in G.nodes(data=True):
        line += "v " + str(id) + " " + str(1) + "\n"
    for (u, v) in G.edges():
        line += "e " + str(u) + " " + str(v) + " " + str(1) + "\n"
    return line


# load mata.so
curPath = os.path.dirname(os.path.dirname(__file__))
so_path = os.path.join(curPath, 'Astar', 'mata.so')
app_astar = ctypes.cdll.LoadLibrary(so_path)  # app_astar: approximate astar
app_astar.mapping_ed.restype = ctypes.c_int
app_astar.map_operations.restype = ctypes.c_char_p #



def assert_mapping_cancer(graphs:list, label_list:list):
    label_list = label_list[0:1000]
    for item in label_list:
        types_of_cancer = set()
        g1 = graphs[item['rowid']]
        g2 = graphs[item['colid']]
        g1_str = nx2txt(g1, str(item['rowid']), types_of_cancer)
        g2_str = nx2txt(g2, str(item['colid']), types_of_cancer)
        given_ged = item['ged']
        g2_order = np.array(item['int_map'])
        g1_order = list()
        for i in range (len(g2_order)): g1_order.append(i)
        g1_order = np.array(g1_order)
        # LIB bool assert_mapping(const char* q_str, const char* g_str, int* q_order_nodes, int* g_order_nodes, int given_ged ) {
        astar_out = app_astar.mapping_ed(CT(g1_str), CT(g2_str), int1ArrayToPointer(g1_order),
                                  int1ArrayToPointer(g2_order))

        if (astar_out != given_ged ):
            print("astar_out:{}, given ged:{}".format(astar_out, given_ged))
            print("rowid:{}".format(item['rowid']))
            print("colid:{}".format(item['colid']))
            # print("g1_str:{}".format(g1_str))
            # print("g2_str:{}".format(g2_str))

def map_operations_cancer(graphs:list, label_list:list):
    label_list = label_list[0:1000]
    insert_node, insert_edge, remove_edge, relabel = np.full(len(label_list), 1e-10), np.full(len(label_list), 1e-10), np.full(len(label_list), 1e-10), np.full(len(label_list), 1e-10)
    for j, item in enumerate(label_list):
        types_of_cancer = set()
        g1 = graphs[item['rowid']]
        g2 = graphs[item['colid']]
        g1_str = nx2txt(g1, str(item['rowid']), types_of_cancer)
        g2_str = nx2txt(g2, str(item['colid']), types_of_cancer)
        given_ged = item['ged']
        g2_order = np.array(item['int_map'])
        g1_order = list()
        for i in range(len(g2_order)): g1_order.append(i)
        g1_order = np.array(g1_order)
        # map operations: //ged; insert_node; insert_edge; remove edge, relabel;
        map_ops = app_astar.map_operations(CT(g1_str), CT(g2_str), int1ArrayToPointer(g1_order), int1ArrayToPointer(g2_order))
        map_ops = map_ops.decode('ascii').split()
        if int(map_ops[0]) == 0: continue
        insert_node[j] = float(map_ops[1]) / float(map_ops[0])
        insert_edge[j] = float(map_ops[2]) / float(map_ops[0])
        remove_edge[j] = float(map_ops[3]) / float(map_ops[0])
        relabel[j] = float(map_ops[4]) / float(map_ops[0])

    print("insert_node={:.4f}, insert_edge={:.4f}, remove_edge={:.4f},"
              " relabel={:.4f}".format( np.mean(insert_node), np.mean(insert_edge), np.mean(remove_edge), np.mean(relabel) ))


def map_operations_imdb(train_graphs: list, test_graphs: list, label_list: list):
    graphs = train_graphs + test_graphs
    label_list = label_list[2000:3000]
    insert_node, insert_edge, remove_edge, relabel = np.full(len(label_list), 1e-10), np.full(len(label_list),
                                                                                              1e-10), np.full(
        len(label_list), 1e-10), np.full(len(label_list), 1e-10)
    for j, item in enumerate(label_list):
        g1 = graphs[item['rowid']]
        g2 = graphs[item['colid']]
        types_of_cancer = set()
        # g1_str = nx2txt_imdb(g1, str(item['rowid']))
        # g2_str = nx2txt_imdb(g2, str(item['colid']))
        g1_str = nx2txt(g1, str(item['rowid']), types_of_cancer)
        g2_str = nx2txt(g2, str(item['colid']), types_of_cancer)
        given_ged = item['ged']
        g2_order = np.array(item['int_map'])
        g1_order = list()
        for i in range(len(g2_order)): g1_order.append(i)
        g1_order = np.array(g1_order)
        # map operations: //ged; insert_node; insert_edge; remove edge, relabel;
        map_ops = app_astar.map_operations(CT(g1_str), CT(g2_str), int1ArrayToPointer(g1_order),
                                           int1ArrayToPointer(g2_order))
        map_ops = map_ops.decode('ascii').split()
        if int(map_ops[0]) == 0: continue
        insert_node[j] = float(map_ops[1]) / float(map_ops[0])
        insert_edge[j] = float(map_ops[2]) / float(map_ops[0])
        remove_edge[j] = float(map_ops[3]) / float(map_ops[0])
        relabel[j] = float(map_ops[4]) / float(map_ops[0])

    print("insert_node={:.4f}, insert_edge={:.4f}, remove_edge={:.4f},"
          " relabel={:.4f}".format(np.mean(insert_node), np.mean(insert_edge), np.mean(remove_edge), np.mean(relabel)))


def assert_mapping_imdb(train_graphs:list, test_graphs:list, label_list:list):
    graphs = train_graphs + test_graphs
    label_list = label_list[0:1000]
    for item in label_list:
        g1 = graphs[item['rowid']]
        g2 = graphs[item['colid']]
        g1_str = nx2txt_imdb(g1, str(item['rowid']))
        g2_str = nx2txt_imdb(g2, str(item['colid']))
        given_ged = item['ged']
        g2_order = np.array(item['int_map'])
        g1_order = list()
        for i in range (len(g2_order)): g1_order.append(i)
        g1_order = np.array(g1_order)
        # LIB bool assert_mapping(const char* q_str, const char* g_str, int* q_order_nodes, int* g_order_nodes, int given_ged ) {
        astar_out = app_astar.mapping_ed(CT(g1_str), CT(g2_str), int1ArrayToPointer(g1_order),
                                  int1ArrayToPointer(g2_order))

        if (astar_out != given_ged ):
            print("astar_out:{}, given ged:{}".format(astar_out, given_ged))
            print("rowid:{}".format(item['rowid']))
            print("colid:{}".format(item['colid']))
            # print("g1_str:{}".format(g1_str))
            # print("g2_str:{}".format(g2_str))

""" test for assert_mapping. 
if __name__ == '__main__':
    ## assert_mapping for cancer dataset
    path = "D:/datasets/GED/Cancer/graphs"
    data_list = read_all_nx_graphs(path)
    labels = load_labels(osp.join(path, 'labels.txt'))
    assert_mapping_cancer(data_list, labels)
    ####################

    ## assert_mapping for IMDB dataset
    # path_train = "D:/workspace/GED/ourGED/datasets/IMDBMulti/raw/IMDBMulti/train/"
    # path_test = "D:/workspace/GED/ourGED/datasets/IMDBMulti/raw/IMDBMulti/test/"
    # path_process = "D:/workspace/GED/ourGED/datasets/IMDBMulti/processed/"
    # train_list = read_all_nx_graphs_imdb(path_train)
    # test_list = read_all_nx_graphs_imdb(path_test)
    # labels = load_labels_imdb( osp.join(path_process, 'pairwise_map.txt') )
    # assert_mapping_imdb(train_list, test_list, labels)
    ####################
    
    #################################### results
    # Cancer中有 5/1000 的mapping和GED没有对应，应该是Hungarian 算法和VJ算法的匹配存在问题，而且两者之间的差距非常小。
    # IMDB 数据中mapping 没有问题
    ####################################
"""


if __name__ == '__main__':
    ## operations for the mapping. CANCER
    # path = "D:/datasets/GED/Cancer/graphs"
    # data_list = read_all_nx_graphs(path)
    # labels = load_labels(osp.join(path, 'labels.txt'))
    # map_operations_cancer(data_list, labels)
    ####################

    ## operations for the mapping. IMDB
    # path_train = "D:/workspace/GED/ourGED/datasets/IMDBMulti/raw/IMDBMulti/train/"
    # path_test = "D:/workspace/GED/ourGED/datasets/IMDBMulti/raw/IMDBMulti/test/"
    # path_process = "D:/workspace/GED/ourGED/datasets/IMDBMulti/processed/"
    # train_list = read_all_nx_graphs_imdb(path_train)
    # test_list = read_all_nx_graphs_imdb(path_test)
    # labels = load_labels_imdb( osp.join(path_process, 'pairwise_map.txt') )
    # map_operations_imdb(train_list, test_list, labels)
    ####################

    # operations for the mapping. AIDS700nef
    path_train = "D:/workspace/GED/ourGED/datasets/AIDS700nef/raw/AIDS700nef/train/"
    path_test = "D:/workspace/GED/ourGED/datasets/AIDS700nef/raw/AIDS700nef/test/"
    path_process = "D:/workspace/GED/ourGED/datasets/AIDS700nef/processed/"
    train_list = read_all_nx_graphs_imdb(path_train)
    test_list = read_all_nx_graphs_imdb(path_test)
    labels = load_labels_imdb( osp.join(path_process, 'pairwise_map.txt') )
    map_operations_imdb(train_list, test_list, labels)
    ###################


    #################################### results
    # AIDS: 
    # IMDB:
    # Cancer:
    ####################################
