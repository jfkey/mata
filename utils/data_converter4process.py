import networkx as nx
import random
import pickle
import os
import re
import os.path as osp
import glob
import torch

path = "D:/datasets/GED/AIDS"
sdz_name = "AID2DA99.sdz"
pat = "[a-zA-Z]{1,2}"
all_graphs_pk = "AIDS_nx.pk"            # 所有图的gexf格式的数据
all_graphs_txt = "AIDS_txt.txt"            # 所有图的gexf格式的数据


def process(r_path):
    import networkx as nx

    ids, Ns = [], []
    # Iterating over paths for raw and processed data (train + test):
    # for r_path, p_path in zip(self.raw_paths, self.processed_paths):
        # Find the paths of all raw graphs:
    names = glob.glob(osp.join(r_path, '*.gexf'))
    # Get sorted graph IDs given filename: 123.gexf -> 123
    ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))

    data_list = []

    for i, idx in enumerate(ids[-1]):
        i = i if len(ids) == 1 else i + len(ids[0])
        # Reading the raw `*.gexf` graph:
        G = nx.read_gexf(osp.join(r_path, f'{idx}.gexf'))
        mapping = {name: j for j, name in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        Ns.append(G.number_of_nodes())

        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        data_list.append(G)
    return data_list


def allgraphs2txt(wfile, graphs):        #将所有的gexf pickle文件，转为txt文件
    f_txt = open(os.path.join(wfile), 'w')

    for gid, G in enumerate(graphs):
        f_txt.write("t " + "# " + str(gid) + "\n")
        for id, label in G.nodes(data=True):
            f_txt.write("v " + str(id) + " " + label['type'] + "\n")
        for (u, v)  in G.edges():
            f_txt.write("e " + str(u) + " " + str(v) + " " + str(1) + "\n")
    f_txt.close()


def gen_node_pair_pk (path, file_name):
    print("gen_node_pair_pk")
    f = open(os.path.join(path,file_name), 'r')
    line = f.readline()
    data = []
    while line is not None and line != '':
        item = dict()
        line = line.split()
        item['g1'] = line[0]
        item['g2'] = line[1]
        item['ged'] = int(line[2])
        map = dict()                        # key:g1 nodes, value: g2 nodes
        for str in line:
            str_arr = str.split("|")
            if (len(str_arr)) > 1 and str_arr[0] != '-1':
                map[str_arr[0]] = str_arr[1]
        item['map'] = map
        data.append(item)
        line = f.readline()
    f.close()

    with open( os.path.join(path, "pairwise_map.pk"), 'wb') as f:
        pickle.dump(data, f)
    # 可通过graph pair的i,j的索引 从pairwise_map中找到图对


if __name__ == '__main__':
    dataset = "AIDS700nef"
    split_index = 560
    exe_file = 'GED-ICDE.exe'

    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    rtrain_path = 'datasets/{}/raw/{}/train/'.format(dataset, dataset)
    rtest_path = 'datasets/{}/raw/{}/test/'.format(dataset, dataset)
    process_path = 'datasets/{}/processed/'.format(dataset)
    ####################  生成txt文件用来计算GED #######################
    train_graph = process(os.path.join(path,rtrain_path))
    test_graph = process(os.path.join(path, rtest_path))
    graphs = train_graph + test_graph
    allgraphs2txt( os.path.join(path, process_path, dataset+"_txt.txt"), graphs)

    ####################     调用exe 计算GED     #######################
    o = os.path.join(path, process_path, "pairwise_map.txt")
    d = os.path.join(path, process_path, dataset+"_txt.txt")
    para = " -m mypairwise -p astar -l LSa -g -s {} -o {}  -d {} -q {} ".format(split_index, o, d, d)
    # s 表示 split index
    # o 表示 输出的文件名
    # d, q表示要进行pair wise 计算GED的文件名
    exe_file_path = os.path.join(path, process_path, exe_file)
    os.system(exe_file_path + para)

    ####################将匹配的节点对构建pickle文件#######################
    gen_node_pair_pk(os.path.join(path, process_path), "pairwise_map.txt" )




