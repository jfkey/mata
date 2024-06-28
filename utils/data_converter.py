import networkx as nx
import random
import pickle
import os
import re


path = "D:/datasets/GED/AIDS"
sdz_name = "AID2DA99.sdz"
pat = "[a-zA-Z]{1,2}"
all_graphs_pk = "AIDS_nx.pk"            # 所有图的gexf格式的数据
all_graphs_txt = "AIDS_txt.txt"            # 所有图的gexf格式的数据


"""
# 把原始sdz文件转成一组gexf 文件，并用二进制的格式保存
# 原始文件名：AID2DA99.sdz
# 保存之后的文件名：
"""
def sdz2gexf(path, sdz_name):
    types = set()
    new_graph = False
    res = dict()

    with open(os.path.join(path,sdz_name) ) as f:
        new_graph = False
        graph_cnt = 0
        line = f.readline()
        while line:

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
                        res[tmpG.graph['id']] = tmpG
                    line = f.readline()


            line = f.readline()
            if (graph_cnt % 10000 == 0 ):
                print(graph_cnt)

            if (line.startswith("$$$$")):
                new_graph = True
                line_idx = 0
                nodes, edges = 0, 0
                while new_graph and line:        # create the graph for the remaining file.
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
                        res[tmpG.graph['id']] = tmpG

    # print(graph_cnt)
    res['types'] = types
    with open ( os.path.join(path, "AIDS_nx.pk") , 'wb') as f:
        pickle.dump(res, f)
    print(types)

def allgraphs2txt():        #将所有的gexf pickle文件，转为txt文件
    f_txt = open(os.path.join(path, all_graphs_txt), 'w')
    with open(os.path.join(path, all_graphs_pk), 'rb') as f:
        all_graphs = pickle.load(f)
    all_graphs.pop('types', None)
    for gid, G in all_graphs.items():
        f_txt.write("t " + "# " + gid + "\n")
        for id, label in G.nodes(data=True):
            f_txt.write("v " + str(id) + " " + label['type'] + "\n")
        for (u, v) in G.edges():
            f_txt.write("e " + str(u) + " " + str(v) + " " + str(1) + "\n")



# 把一个networkx的graph转成txt格式，并写到文件f中
def gexf2txt(f, graphid, all_graphs):  #
    # G = nx.read_gexf(path + graphid)
    G = all_graphs[graphid]
    f.write("t " + "# " + graphid.split(".")[0] + "\n")
    for id, label in G.nodes(data=True):
        f.write("v " + str(id) + " " + label['type'] + "\n")
    for (u, v) in G.edges():
        f.write("e " + str(u) + " " + str(v) + " " + str(1) + "\n")



"""
把生成的graph-> topK graphs，
生成exact-GED执行的多个graph_g.txt和graph_q.txt
"""
def split4runGED(file_name ,file_num):
    f = open(os.path.join(path, all_graphs_pk), 'rb')
    all_graphs = pickle.load(f)
    f.close()

    f1 = open(os.path.join(path, "query-first-4000-100-0.txt"), 'w')
    f2 = open(os.path.join(path, "query-second-4000-100-0.txt"), 'w')

    with open(file_name, 'r') as f:
        lines = f.readlines()
        num_graph_each_file = (int)(len(lines) / file_num)
        for idx, line in enumerate(lines):
            linearr = line.strip().split("\t")
            for tmp in linearr[2:]:
                gexf2txt(f1, linearr[0], all_graphs)
                gexf2txt(f2, tmp, all_graphs)
            if ( (idx + 1) % num_graph_each_file == 0 ):
                f1.close()
                f2.close()
                cur_epoch = str( (int) ((idx + 1) / num_graph_each_file) )
                f1 = open(os.path.join(path, "query-first-4000-100-" + cur_epoch + ".txt"), 'w')
                f2 = open(os.path.join(path, "query-second-4000-100-" + cur_epoch + ".txt"), 'w')




def split_ged_map(filename): # 把ged_map 给划分成training testing, val
    #ged_map_train.txt #ged_map_test.txt #ged_map_val.txt
    p_train, p_test, p_val = 0.6, 0.2, 0.2
    line_arr = []
    with open(filename, 'r') as f:
        line = f.readline()
        line_arr.append(line)
        while line is not None and line != '':
            line_arr.append(line)
            line = f.readline()
    random.shuffle(line_arr)
    ftrain = open(filename.split(".")[0] + "_train.txt", 'w')
    ftest = open(filename.split(".")[0] + "_test.txt", 'w')
    fval = open(filename.split(".")[0] + "_val.txt", 'w')
    for i, line in enumerate(line_arr):
        if i < len(line_arr) * p_train:
            ftrain.write(line)
        if i >= len(line_arr) * (p_train) and i < len(line_arr) * (p_train+ p_val):
            fval.write(line)
        if i >= len(line_arr) * (p_train+ p_val):
            ftest.write(line)
    ftrain.close()
    ftest.close()
    fval.close()


if __name__ == '__main__':
    # sdz2gexf(path, sdz_name)
    # file_name = "D:/datasets/GED/AIDS/gexf/query4000-100.txt"
    # split4runGED(file_name, 4)
    # file_name = "D:/datasets/GED/AIDS/gexf/ged_map.txt"
    # split_ged_map(file_name)
    allgraphs2txt()

