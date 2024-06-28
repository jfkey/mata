import operator
import shutil
from annoy import AnnoyIndex
import pickle
import random
import numpy as np
import os
import os.path as osp
import glob
import pickle
import networkx as nx

# sort the query id by the graph size (increasing order) 最后返回都是小于node_num的图
# input:
# path: path of all .gexf file
# random_query_idx: a list of query index. with values 0, 3, xxx
# idx2fileID: map the index to the graph file name
def rank_query_ids_filter(path, random_query_idx, idx2fileID, node_num):
    print("load all gexf graphs from.{}".format(path))
    idx_graphnum = dict()
    for i in random_query_idx:
        fileID = idx2fileID[i]
        G = nx.read_gexf(osp.join(path, fileID))
        if G.number_of_nodes() <= node_num and G.number_of_nodes() > 5:     # 生成大于5小于16个节点的图，具有精确GED
            idx_graphnum[i] = G.number_of_nodes()
    idx_graphnum = sorted(idx_graphnum.items(),key=operator.itemgetter(1))
    return idx_graphnum

# 把一个networkx的graph转成txt格式，并写到文件f中
def gexf2txt(f, graphid, path):  #
    G = nx.read_gexf(path + graphid)
    f.write("t " + "# " + graphid.split(".")[0] + "\n")
    for id, label in G.nodes(data=True):
        f.write("v " + str(id) + " " + label['type'] + "\n")
    for (u, v) in G.edges():
        f.write("e " + str(u) + " " + str(v) + " " + str(1) + "\n")


# 只生成小于16个节点的图，作为训练集。
def small_graph_topKsearch(path, embed_name, query_num, topK,node_num):    # file_id 123.gexf
    id2embeding = dict()
    fileID2idx, idx2fileID = dict(), dict()
    # load embedding.
    f = open(embed_name, "rb")
    id2embeding = pickle.load(f)
    f.close()

    vector_len = 32  # Length of item vector that will be indexed
    annoy = AnnoyIndex(vector_len, 'euclidean')     #
    idx = 0
    for graph_name in id2embeding:
        fileID2idx[graph_name] = idx
        idx2fileID[idx] = graph_name

        val = np.reshape(np.array(id2embeding[graph_name]), -1)
        vec =[i for i in val]    # convert numpy array to vector
        annoy.add_item(idx, vec)
        idx += 1
    annoy.build(10)             # 10 trees

    all_idx = [v for v in range (len (id2embeding))]
    # all_idx = [v for v in range (100)]
    # rank the query graphs by node size.
    all_idx_graphnum = rank_query_ids_filter(path, all_idx, idx2fileID, node_num)
    random_query_idx = random.sample(range(0, len(all_idx_graphnum)), query_num)
    random_query_idx = sorted(random_query_idx)
    idx_graphnum = [all_idx_graphnum[i] for i in random_query_idx]

    query_topK_res = dict()
    for k in idx_graphnum:
        query_topK_res[k[0]] = annoy.get_nns_by_item(k[0], topK)

    print("generate query text")
    f1 = open(path + "query-first-" + str(query_num) +"-"+ str(topK)  + ".txt", 'w')
    f2 = open(path + "query-second-" + str(query_num) + "-" + str(topK) + ".txt", 'w')
    # write to file, with formate query id \t top_id \t top_id
    with open(path + "query" + str(query_num) +"-"+ str(topK) + ".txt", 'w') as f:
        for x, item in enumerate (query_topK_res):
            f.write( idx2fileID[item].split(".")[0] + "\t" + str(idx_graphnum[x][1]))
            for i in query_topK_res[item]:
                f.write("\t" + idx2fileID[i].split(".")[0])
                gexf2txt(f1, idx2fileID[item], path)
                gexf2txt(f2, idx2fileID[i], path)
            f.write("\n")
    f1.close()
    f2.close()


if __name__ == '__main__':
    # file_id = "633075.gexf"  一共有87个节点
    # file_id = "143437.gexf" # 一共62个节点
    # file_id = "675985.gexf" # 一共27个节点
    # file_id = "653319.gexf" # 一共35个节点
    # file_id = "634182.gexf"  # 一共15个节点
    # file_id = "639687.gexf"  # 一共20个节点
    # file_id = "57598.gexf"  # 9个节点
    path = "D:/datasets/GED/AIDS/gexf/"
    embed_name = "id2embeding.pickle"
    query_num = 4000
    topK = 100

    node_num = 16
