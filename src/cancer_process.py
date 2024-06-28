# coding=utf8
import os
import operator
import re
from tqdm import tqdm
from src.utils import *
from src.traditional_baseline import gen_ground_truth
import glob

"""
1. 随机选择800graph，节点大小位于20-100之间
2. 对(800*800)/2 的graph pairs中选择 100K图对
"""
pat = "[a-zA-Z]{1,2}"               # 用来匹配节点的label
all_line_count = 8000000            # the estimation of all line count of sdz file

def sdz2gexf(path, sdz_name):
    print("load sdz data: {}{}, and convert to gexf".format(path, sdz_name))
    tq = tqdm(all_line_count, ascii=True, position=0)
    update_tq_idx = 0

    types = set()
    new_graph = False
    all_graphs_nx = dict()
    with open(os.path.join(path, sdz_name) ) as f:
        new_graph = False
        graph_cnt = 0
        line = f.readline()
        while line:
            update_tq_idx = 0
            update_tq_idx = update_tq_idx+1
            line = line.rstrip()
            if (graph_cnt == 0):
                new_graph = True
                line_idx = 0
                nodes, edges = 0, 0
                while new_graph and line:  # create the  graph for the first line
                    line = line.rstrip()
                    if (line_idx == 0):
                        tmpG = nx.Graph(id=line)
                    if (line_idx == 3):
                        nodes = int(line[0:3].strip())
                        edges = int(line[3:6].strip())
                    if (line_idx > 3 and line_idx <= 3 + nodes):
                        # tmpG.add_node(line_idx - 4, type= line[30:35].strip() )
                        type = re.findall(pat, line)[0]
                        tmpG.add_node(line_idx - 4, type=type)
                        types.add(type)
                    elif (line_idx > 3 + nodes and line_idx <= 3 + nodes + edges):
                        tmpG.add_edge(int(line[0:3].strip()) - 1, int(line[3:6].strip()) - 1, valence=int(line[6:9].strip() ))
                    line_idx += 1

                    if (line.startswith("M")):
                        graph_cnt += 1
                        new_graph = False
                        # nx.write_gexf(tmpG, os.path.join(path, "gexf", tmpG.graph['id'] + ".gexf"))
                        all_graphs_nx[tmpG.graph['id']] = tmpG
                    line = f.readline()

            line = f.readline()

            if (line.startswith("$$$$")):
                new_graph = True
                line_idx = 0
                nodes, edges = 0, 0
                while new_graph and line:        # create the graph for the remaining file.
                    update_tq_idx = update_tq_idx + 1
                    line = f.readline().rstrip()
                    if (line_idx == 0):
                        tmpG = nx.Graph(id=line)
                    if (line_idx == 3):
                        nodes = int(line[0:3].strip())
                        edges = int(line[3:6].strip())
                    if (line_idx > 3 and line_idx <= 3 + nodes ):
                        type = re.findall(pat, line)[0]
                        tmpG.add_node(line_idx - 4, type=type)
                        types.add(type)

                    elif (line_idx > 3 + nodes and line_idx <= 3 + nodes + edges):
                        tmpG.add_edge( int( line[0:3].strip())-1, int(line[3:6].strip()) -1, valence = int( line[6:9].strip() ) )
                    line_idx += 1
                    if ( line.startswith("M")):
                        graph_cnt += 1
                        new_graph = False
                        # nx.write_gexf(tmpG, os.path.join(path, "gexf", tmpG.graph['id'] + ".gexf" ) )
                        all_graphs_nx[tmpG.graph['id']] = tmpG

            tq.update(update_tq_idx)
            tq.set_description("Processing...")
    return all_graphs_nx

def sample_from_nx(all_graph_nx, graph_num=800, min_nodes = 20, max_nodes= 100):
    print("sample {} from all graphs, with node size > {} and < {}".format(graph_num, min_nodes, max_nodes))
    sampled_nx = dict()
    while (len(sampled_nx) < graph_num ):
        id = random.choice(list(all_graph_nx))
        if id not in sampled_nx:
            nodesize = len(all_graph_nx[id])
            if nodesize > min_nodes and nodesize < max_nodes:
                sampled_nx[int(id)] = all_graph_nx[id]
    ## write to
    sampled_nx = sorted(sampled_nx.items(), key=operator.itemgetter(0))

    return sampled_nx

# sample_nx: list<id, nx.graph>
# A* output:
# 	ged search_space time_cost q_id g_id node_matching
# 	e.g., 0 5 2 186 126 0|2 1|1 2|0 3|3 4|4
# 	time_cost: microsecond.
# 	node_matching: u1|v1 u2|v2 ... um|vm ... -1|vn, where u_i is the node of graph q,  and v_i is the node of graph g.
def gen_cancer_pairs(sampled_nx, path, cancer_dataset_size = 100000):
    print("generate {}  pairs from the sample graphs".format(cancer_dataset_size))
    types_of_cancer = set()
    beam_size = 20
    id2node_gap = dict()
    if not os.path.exists(path):
        os.mkdir(path)
    for i, g_pair in enumerate(sampled_nx):
        nx.write_gexf(g_pair[1], os.path.join(path, "{}.gexf".format(g_pair[0])))

    #选择尽可能相似的pair作为Cancer数据集（即节点规模差距尽可能小的）
    for i in range(len(sampled_nx)):
        j = i
        while j < len(sampled_nx):
            id2node_gap[(i,j)] = np.abs(len(sampled_nx[i][1]) - len(sampled_nx[j][1]))
            j = j + 1

    id2node_gap = sorted(id2node_gap.items(), key=operator.itemgetter(1))

    id2node_gap = id2node_gap[0: cancer_dataset_size]
    all_lines = []
    beam_count, hungarian_count, vj_count = 0, 0, 0

    for item in id2node_gap:
        item = item[0]
        g1 = sampled_nx[item[0]][1]
        g2 = sampled_nx[item[1]][1]
        g1_id = str(sampled_nx[item[0]][0])
        g2_id = str(sampled_nx[item[1]][0])
        if g1.number_of_nodes() > g2.number_of_nodes():         # 在调用传统算法求解的时候，总是保证 |g1| < |g2|
            res_str, executed_alg = gen_ground_truth(g2, g2_id, g1, g1_id, beam_size, types_of_cancer)
            if executed_alg == 1: beam_count += 1
            elif executed_alg == 2: hungarian_count += 1
            elif executed_alg == 3: vj_count += 1
        else:
            res_str, executed_alg = gen_ground_truth(g1, g1_id, g2, g2_id, beam_size, types_of_cancer)
            if executed_alg == 1: beam_count += 1
            elif executed_alg == 2: hungarian_count += 1
            elif executed_alg == 3: vj_count += 1

        line = str(item[0]) + " " + str(item[1]) + " " + g1_id + " " + g2_id + " " + res_str
        all_lines.append(line)

    # random shuffle
    random.shuffle(all_lines)
    flabels = open(os.path.join(path, "labels.txt"), 'w')
    for line_item in all_lines:
        flabels.write(line_item + "\n")
    flabels.close()

    print("dataset size of CANCER:{}, \n The number of executions by beam is {}. "
          "\n The number of executions by hungarian is {}."
          "\n The number of executions by VJ is {}.".format(len(all_lines), beam_count, hungarian_count, vj_count))
    print("types of CANCER: {}".format(types_of_cancer))



#################注意 这里和aids，imdb生成的network不同。################
def read_all_nx_graphs(r_path):
    import networkx as nx
    ids, Ns = [], []
    names = glob.glob(osp.join(r_path, '*.gexf'))
    ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))
    data_list = []
    for i, idx in enumerate(ids[-1]):
        i = i if len(ids) == 1 else i + len(ids[0])
        # Reading the raw `*.gexf` graph:
        G = nx.read_gexf(osp.join(r_path, f'{idx}.gexf'))
        data_list.append(G)
    return data_list
#################注意 这里和aids，imdb生成的network不用。################



if __name__ == '__main__':
    """ 1. 随机选择800graph，节点大小位于20-100之间
        2. 对(800*800)/2 的graph pairs中选择 100K图对，图对之间的节点差异尽可能的小
        3. 计算100k个图对的标签，并记录标签集 
        三个传统的方法: A*LSa beam: beam size 20;  Hungarian;  VJ
        需要注意的是: Cancer数据集中的labels.txt中mapping的节点顺序和networkx中的graph是对应的 
    """
    path = "D:/datasets/GED/Cancer/"
    sdzname = "CAN2DA99.sdz"
    graph_num = 800
    cancer_dataset_size = 100000
    all_graphs_nx = sdz2gexf(path, sdzname)
    sampled_nx = sample_from_nx(all_graphs_nx, graph_num=graph_num)
    gen_cancer_pairs(sampled_nx, path=os.path.join(path, "graphs"), cancer_dataset_size=cancer_dataset_size)

'''
Output:
Processing...: : 7965009it [18:20, 8406.07it/s]dataset size of CANCER:100000, 
 The number of executions by beam is 99245. 
 The number of executions by hungarian is 755.
 The number of executions by VJ is 0.
types of CANCER: {'Ag', 'In', 'S', 'La', 'Pd', 'Br', 'Co', 'Fe', 'Ge', 'Zn', 'W', 'O', 'Ni', 'P', 'Rh', 'B', 'Na', 'U', 'Mo', 'N', 'Ti', 'F', 'Ir', 'Cl', 'Pt', 'As', 'Os', 'Sn', 'C', 'Au', 'Nd', 'Ga', 'Bi', 'I', 'Ru', 'Zr', 'Si'}
Processing...: : 7965009it [47:46, 2778.95it/s]
'''