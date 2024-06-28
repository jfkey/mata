import pickle
import copy
import platform
import time
import torch
import math
import glob
import os
from os.path import isfile
from os import getpid
import os.path as osp
import fileinput
import networkx as nx
import numpy as np
from nx_to_gxl import nx_to_gxl
from utils import calculate_ranking_correlation, calculate_prec_at_k
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm, trange

if platform.system().lower() == 'windows':
    java_env = "D:/develop-soft/java/jdk8/bin/"
else:
    java_env = "/usr/lib/jvm/java-11-openjdk-amd64/bin"
os.environ["PATH"] += os.pathsep + os.pathsep.join([java_env])
beam_size = 10  # default beam size
# global dataset_name
# dataset_name = "AIDS700nef"


def ged(g1, g2, algo, dataset_name, debug=False, timeit=False):
    # https://github.com/dan-zam/graph-matching-toolkit
    gp = get_gmt_path()
    append_str = get_append_str(g1, g2)
    if platform.system().lower() == 'windows':
        src, t_datapath = setup_temp_data_folder(gp, append_str)
    else:
        src, t_datapath = setup_temp_data_folder_linux(gp, append_str)

    meta1 = write_to_temp(g1, t_datapath, algo, 'g1')
    meta2 = write_to_temp(g2, t_datapath, algo, 'g2')
    if dataset_name in "AIDS700nef":
        meta1 = '{}_label_type_id'.format(algo)
        meta2 = '{}_label_type_id'.format(algo)
    else:
        meta1 = '{}_label_id'.format(algo)
        meta2 = '{}_label_id'.format(algo)

    if meta1 != meta2:
        raise RuntimeError( 'Different meta data {} vs {}'.format(meta1, meta2))
    if platform.system().lower() == 'windows':
        prop_file = setup_property_file(src, gp, meta1, append_str)
    else:
        prop_file = setup_property_file_linux(src, gp, meta1, append_str)
    rtn = []

    classpath = os.path.join(get_root_path(), 'src', 'gm_toolkit', 'bin')
    classpath = " {} algorithms.GraphMatching ".format(classpath)
    properties_file ='./properties/properties_temp_{}.prop'.format(append_str)
    exec('cd {} '.format(gp))
    exec_ged = ' cd {} && java {} -classpath {} {}'.format (
            gp, '-XX:-UseGCOverheadLimit -XX:+UseConcMarkSweepGC -Xmx100g' if algo == 'astar' else '', classpath, properties_file)
    if not exec(exec_ged):
        rtn.append(-1)
    else:
        d, t, lcnt, g1size, g2size, result_file = get_result(gp, algo, append_str)
        rtn.append(d)
        if g1size != g1.number_of_nodes():
            print('g1size {} g1.number_of_nodes() {}'.format(g1size, g1.number_of_nodes()))
        assert (g1size == g1.number_of_nodes())
        assert (g2size == g2.number_of_nodes())
    if debug:
        rtn += [lcnt, g1, g2]
    if timeit:
        rtn.append(t)
    clean_up(t_datapath, prop_file, result_file)
    # if len(rtn) == 1:
    #     return rtn[0]
    # return tuple(rtn)
    return d, t, lcnt


def normalized_ged(d, g1, g2):
    return 2 * d / (g1.number_of_nodes() + g2.number_of_nodes())

def unnormalized_ged(d, g1, g2):
    return d * (g1.number_of_nodes() + g2.number_of_nodes()) / 2


def setup_temp_data_folder_linux(gp, append_str):
    tp = gp + '/data/temp_{}'.format(append_str)
    exec('rm -rf {} && mkdir {}'.format(tp, tp))
    src = get_root_path() + '/src/gmt_files'
    exec('cp {}/temp.xml {}/temp_{}.xml'.format(src, tp, append_str))
    return src, tp

def setup_temp_data_folder(gp, append_str):
    tp = os.path.join(gp, 'data', 'temp_{}'.format(append_str))
    if os.path.exists(tp):
        exec('rmdir /s/q {}'.format(tp))
    exec('mkdir {}'.format(tp))
    src = os.path.join( get_root_path(), 'src', 'gmt_files', 'temp.xml')
    dest = os.path.join(tp, 'temp_{}.xml'.format(append_str))
    exec('copy {} {}'.format(src, dest))
    from os.path import dirname
    src = dirname(src)
    return src, tp


def write_to_temp(g, tp, algo, g_name):
    node_attres, edge_attrs = nx_to_gxl(g, g_name, os.path.join(tp, "{}.gxl".format(g_name)) )
    return algo + '_' + '_'.join(sorted(list(node_attres.keys())) + \
                                 sorted(list(edge_attrs.keys())))


def setup_property_file(src, gp, meta, append_str):
    destfile = os.path.join(gp, 'properties', 'properties_temp_{}.prop'.format(append_str))
    srcfile = os.path.join(src, '{}.prop'.format(meta))
    if not isfile(srcfile):
        if 'beam' in meta:  # for beam
            metasp = meta.split('_')
            s = int(metasp[0][4:])
            if s <= 0:
                raise RuntimeError('Invalid s for beam search: {}'.format(s))
            newmeta = '_'.join(['beam'] + metasp[1:])
            srcfile = '{}/{}.prop'.format(src, newmeta)
        else:
            raise RuntimeError('File {} does not exist'.format(srcfile))
    exec('copy {} {}'.format(srcfile, destfile))
    for line in fileinput.input(destfile, inplace=True):
        line = line.rstrip()
        if line == 's=':  # for beam
            print('s={}'.format(beam_size))
        else:
            print(line.replace('temp', 'temp_{}'.format(append_str)))
    return destfile


def setup_property_file_linux(src, gp, meta, append_str):

    destfile = '{}/properties/properties_temp_{}.prop'.format( \
        gp, append_str)
    srcfile = '{}/{}.prop'.format(src, meta)
    if not isfile(srcfile):
        if 'beam' in meta:  # for beam
            metasp = meta.split('_')
            s = int(metasp[0][4:])
            if s <= 0:
                raise RuntimeError('Invalid s for beam search: {}'.format(s))
            newmeta = '_'.join(['beam'] + metasp[1:])
            srcfile = '{}/{}.prop'.format(src, newmeta)
        else:
            raise RuntimeError('File {} does not exist'.format(srcfile))
    exec('cp {} {}'.format(srcfile, destfile))
    for line in fileinput.input(destfile, inplace=True):
        line = line.rstrip()
        if line == 's=':  # for beam
            print('s={}'.format(beam_size))
        else:
            print(line.replace('temp', 'temp_{}'.format(append_str)))
    return destfile


def get_result(gp, algo, append_str):
    # result_file = '{}/result/temp_{}'.format(gp, append_str)
    result_file = os.path.join(gp, 'result', 'temp_{}'.format(append_str))
    with open(result_file) as f:
        lines = f.readlines()
        ln = 16 if 'beam' in algo else 15
        t = int(lines[ln].split(': ')[1])  # msec
        ln = 23 if 'beam' in algo else 22
        d = float(lines[ln]) * 2  # alpha=0.5 --> / 2
        assert (d - int(d) == 0)
        d = int(d)
        if d < 0:
            d = -1  # in case rtn == -2
        ln = 26 if 'beam' in algo else 25
        g1size = int(lines[ln])
        ln = 27 if 'beam' in algo else 26
        g2size = int(lines[ln])
        ln = 28 if 'beam' in algo else 27
        lcnt = int(float(lines[ln]))
        return d, t, lcnt, g1size, g2size, result_file

def get_root_path():       # D:\\workspace\\GED\\ourGED'
    from os.path import dirname, abspath
    return dirname(dirname(abspath(__file__)))


exec_print = True
def exec_turnoff_print():
    global exec_print
    exec_print = False
def exec_turnon_print():
    global exec_print
    exec_print = True

def exec(cmd, timeout=None):
    global exec_print
    if not timeout:
        # from os import system
        # if exec_print:
        #     print(cmd)
        # else:
        #     cmd += ' > /dev/null'
        # system(cmd)
        import subprocess
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True  # finished
    else:
        import subprocess as sub
        import threading

        class RunCmd(threading.Thread):
            def __init__(self, cmd, timeout):
                threading.Thread.__init__(self)
                self.cmd = cmd
                self.timeout = timeout

            def run(self):
                self.p = sub.Popen(self.cmd, shell=True)
                self.p.wait()

            def Run(self):
                self.start()
                self.join(self.timeout)

                if self.is_alive():
                    self.p.terminate()
                    self.join()
                    self.finished = False
                else:
                    self.finished = True

        if exec_print:
            print('Timed cmd {}sec {}'.format(timeout, cmd))
        r = RunCmd(cmd, timeout)
        r.Run()
        return r.finished

def get_gmt_path():
    return os.path.join(get_root_path(), 'src', 'gm_toolkit')

tstamp = None

def get_ts():
    import datetime, pytz
    global tstamp
    if not tstamp:
        tstamp = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%dT%H-%M-%S')
    return tstamp

def get_append_str(g1, g2):
    # return '{}_{}_{}_{}'.format(get_ts(), getpid(), g1.graph['gid'], g2.graph['gid'])
    return '{}_{}_{}_{}'.format(get_ts(), getpid(), g1.name, g2.name)

def clean_up_linux(t_datapath, prop_file, result_file):
    for path in [t_datapath, prop_file, result_file]:
        exec('rm -rf {}'.format(path))

def clean_up(t_datapath, prop_file, result_file):
    for path in [t_datapath, prop_file, result_file]:
        if os.path.isfile(path):
            exec('del /s/q {}'.format(path))
        else:
            exec('rmdir /s/q {}'.format(path))


################################################ compute metric ################################################
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
    return training_graphs, val_graphs, testing_graphs, nged_matrix

def load_graph_from_raw(path):
    ids = []
    names = glob.glob(osp.join(path, '*.gexf'))
    ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))
    graph_list= []
    for i, idx in enumerate(ids[-1]):
        g = nx.read_gexf(osp.join(path, f'{idx}.gexf'))
        graph_list.append(g)
    return graph_list

def normalize_ged(g1_nodes, g2_nodes, ged):
    return np.exp(-1 * (2 * ged/ ( g1_nodes+ g2_nodes)))

def denormalize_ged(g1_nodes, g2_nodes, sim_score):
    nged = -math.log(sim_score, math.e)
    return (nged * (g1_nodes + g2_nodes) / 2)

# Input:  (1) batch networkx graph list1, (2) batch networkx graph list2, (3) alg name, e.g., hungarian, beam,
#         (4) dataset_name: e.g., AIDS700nef, (5) norm_gt_list: normalized ground truth of the batch pairs
#
# Output: (1) norm prediction GED; (2) best GED between prediction and ground truth
#         (3) exec time (4) search space (5) superior_num: the number of pred ged less than ground truth; (6) acc_num:
#
def calc_metric_batch_pair(batch_graph1, batch_graph2, alg, dataset_name, norm_gt_list):
    superior_num, acc_num = 0, 0
    prediction_ged, best_ged, exec_time, search_space = [], [], [], []
    assert (len(norm_gt_list) == len(batch_graph2))
    for i in range(len(batch_graph1)):
        g1 = batch_graph1[i]
        g2 = batch_graph2[i]
        cost, t, ss = ged(g1, g2, alg, dataset_name, debug=False, timeit=False)

        g1size = len(g1)
        g2size = len(g2)
        prediction_ged.append(normalize_ged(g1size, g2size, cost))
        exec_time.append(t)
        search_space.append(ss)
        gt_item = denormalize_ged(g1size, g2size, norm_gt_list[i])
        best_item = min(cost, gt_item)
        best_ged.append(normalize_ged(g1size, g2size, best_item) )
        if cost < gt_item - 1: superior_num += 1
        if int(cost) == int (gt_item): acc_num += 1
    return np.array(prediction_ged), np.array(best_ged), np.array(exec_time), np.array(search_space), superior_num, acc_num


def score(train_path:str, test_path:str, dataset:str, alg: str, batch_size:int):
    training_graphs = load_graph_from_raw(train_path)       # networkx of graphs
    testing_graphs = load_graph_from_raw(test_path)         # networkx of graphs

    _, _, _, norm_gt = load_ground_truth(os.path.dirname(train_path), dataset)
    norm_gt = np.array(norm_gt)
    l1_list, l2_list, rho_list, tau_list, pat10_list, pat20_list, tm_list, ss_list  = [], [], [], [], [], [], [], []
    acc_num, superior_num, exec_num = 0, 0, 0
    non_test_size = len(training_graphs)        # 560 for AIDS dataset
    all_trad = []

    if dataset_name in ['AIDS700nef', 'IMDBMulti']:
        training_graphs = training_graphs[:len(training_graphs) // 4 * 3]
    if dataset_name in ['IMDBMulti']:
        training_graphs = training_graphs[: len(training_graphs) // 4 ] # 900:300:300
    ############################################
    # testing_graphs = testing_graphs[0:14]
    # training_graphs = training_graphs[0:13]
    ############################################

    start_t = time.time()
    tq = tqdm(total=len(testing_graphs) * len(training_graphs), ascii=True, position=0)
    for i, g1 in enumerate(testing_graphs):
        data_list_1 = list([g1]* len(training_graphs))
        prediction_row, best_row = np.full(len(training_graphs), 1e-10), np.full(len(training_graphs), 1e-10)
        for start_idx in range(0, len(training_graphs), batch_size):
            nx_graph_list1 = data_list_1[start_idx:start_idx+batch_size]
            nx_graph_list2 = training_graphs[start_idx:start_idx+batch_size]
            norm_gt_list = norm_gt[i + non_test_size, start_idx:start_idx + len(nx_graph_list1)]
            pred_batch, best_batch, exec_time, ss, superior_batch, acc_batch = calc_metric_batch_pair(nx_graph_list1, nx_graph_list2, alg, dataset, norm_gt_list)

            superior_num += superior_batch
            acc_num += acc_batch
            exec_num += len(nx_graph_list1)
            prediction_row[start_idx:start_idx + len(nx_graph_list1)] = pred_batch
            best_row[start_idx:start_idx + len(nx_graph_list1)] = best_batch

            tm_list.append(np.mean(exec_time))
            ss_list.append(np.mean(ss))
            tq.update(len(nx_graph_list1))
            tq.set_description("MAE={:.3f}".format( np.mean( np.abs(pred_batch - best_batch)) ))

        rho_list.append(calculate_ranking_correlation(spearmanr, prediction_row, best_row))
        tau_list.append(calculate_ranking_correlation(kendalltau, prediction_row, best_row))
        pat10_list.append(calculate_prec_at_k(10, prediction_row, best_row))
        pat20_list.append(calculate_prec_at_k(20, prediction_row, best_row))
        l1_list.append(np.mean(np.abs(prediction_row - best_row)))
        l2_list.append(np.mean(np.square(prediction_row - best_row)) )
        all_trad.append(prediction_row)

    print("norm_mae: " + str(round(np.mean(l1_list), 5)))
    print("norm_mse: " + str(round( np.mean(l2_list), 5)))
    print("acc: " + str(round(acc_num*1.0 / exec_num, 5)))
    print("Spearman's_rho: " + str(round(np.mean(rho_list), 5)))
    print("Kendall's_tau: " + str(round(np.mean(tau_list), 5)))
    print("p@10: " + str(round( np.mean(pat10_list), 5 )))
    print("p@20: " + str(round( np.mean(pat20_list), 5)))
    print("search_space: " + str( round(np.mean(ss_list), 5)  ))
    print("average_time(msec.): " + str(round( np.mean(tm_list), 5)))
    print("exec_time(sec.): " + str(round((time.time() - start_t), 5)))
    print("superior_num: " + str(round(superior_num * 1.0 / (len(training_graphs) * len(testing_graphs)), 5)))

    # store all feasible results of the cancer dataset. (.txt),  (pairwise_map.txt store the index and its mapping)
    #   rowid colid rgid cgid g1 g2 beam hun vj
    label_path = os.path.dirname(os.path.dirname(os.path.dirname(train_path)))
    new_label_items = []
    wf = open(os.path.join(label_path, 'processed', 'fea_results_{}.txt'.format(alg)), 'w')
    for i, g1 in enumerate(testing_graphs):
        for j, g2 in enumerate(training_graphs):
            line = "{} {} {} \n".format(i + non_test_size, j, all_trad[i][j])
            item = dict()
            item['rowid'] = i + non_test_size
            item['colid'] = j
            item[alg] = all_trad[i][j]
            new_label_items.append(item)
            wf.write(line)
    wf.close()
    # store all feasible results of the cancer dataset. (.pt)
    with open(os.path.join(label_path, 'processed', "fea_results_{}.pt".format(alg)), 'wb') as f:
        pickle.dump(new_label_items, f)

if __name__ == '__main__':
    # test for pair
    # astar, beam, hungarian, vj
    # alg = "beam"
    # g1 = nx.read_gexf(get_root_path() + '/temp/6.gexf')
    # g2 = nx.read_gexf(get_root_path() + '/temp/21.gexf')
    # a, b, c = ged(g1, g2, alg, debug=False, timeit=False)
    # print(a, b, c)

    # dataset_name = "IMDBMulti"
    # path = "D:/workspace/GED/ourGED/datasets/{}/raw/{}/"
    # alg = "hungarian"
    # batch_size = 20
    # path = path.format(dataset_name, dataset_name)
    # score(os.path.join(path, "train"), os.path.join(path, "test"), dataset_name, alg, batch_size)

    import sys
    dataset_name = sys.argv[1]
    alg = sys.argv[2]
    batch_size = 128                                        # default size
    if platform.system().lower() == 'windows':
        path = "D:/workspace/GED/ourGED/datasets/{}/raw/{}/"
    else:
        path = "/home/LAB/liujf/workspace/GED/ourGED/datasets/{}/raw/{}/"

    path = path.format(dataset_name, dataset_name)
    score(os.path.join(path, "train"), os.path.join(path, "test"), dataset_name, alg, batch_size)

"""
python src/traditional_algs.py AIDS700nef beam   > aids_trad_beam-19.log 2>&1 &
python src/traditional_algs.py AIDS700nef hungarian  > aids_trad_hung-19.log 2>&1 &
python src/traditional_algs.py AIDS700nef vj  > aids_trad_vj-19.log 2>&1 &

python src/traditional_algs.py IMDBMulti beam  > imdb_trad_beam-19.log 2>&1 &
python src/traditional_algs.py IMDBMulti hungarian > imdb_trad_hung-19.log 2>&1 &
python src/traditional_algs.py IMDBMulti vj  > imdb_trad_vj-19.log 2>&1 &

python src/traditional_algs_cancer.py CANCER > cancer_all_trads-19.log 2>&1 &
"""