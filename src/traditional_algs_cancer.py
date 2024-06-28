#encoding=utf-8
#from utils import get_root_path, exec, get_ts
import copy
import platform
from nx_to_gxl import nx_to_gxl
from os.path import isfile
from os import getpid
import fileinput
import networkx as nx
import os
import numpy as np
import torch
import math
import glob
import os.path as osp
import time
from utils import calculate_ranking_correlation, calculate_prec_at_k
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm, trange
import pickle

if platform.system().lower() == 'windows':
    java_env = "D:/develop-soft/java/jdk8/bin/"
else:
    java_env = "/usr/lib/jvm/java-11-openjdk-amd64/bin"
os.environ["PATH"] += os.pathsep + os.pathsep.join([java_env])
beam_size = 10  # default beam size

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
    if dataset_name in ["AIDS700nef", "CANCER"]:
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
    return training_graphs, val_graphs, testing_graphs,   nged_matrix


def load_graph_from_raw(path):
    ids = []
    names = glob.glob(osp.join(path, '*.gexf'))
    ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))
    graph_list = []
    for i, idx in enumerate(ids[-1]):
        g = nx.read_gexf(osp.join(path, f'{idx}.gexf'))
        graph_list.append(g)
    return graph_list



def normalize_ged(g1_nodes, g2_nodes, ged):
    return np.exp(-1 * (2 * ged/ ( g1_nodes+ g2_nodes)))

def denormalize_ged(g1_nodes, g2_nodes, sim_score):
    nged = -math.log(sim_score, math.e)
    return (nged * (g1_nodes + g2_nodes) / 2)

# Input:  (1) batch networkx graph list1, (2) batch networkx graph list2, (3) dataset_name: i.e., cancer,
#
# Output: (1) best_ged: i.e., the best ged of (beam, hun, vj); (2) prediction ged of beam, hun, vj, respectively.
#         (3) exec time of beam, hun, vj, respectively. (4) search space of beam, hun, vj, respectively.
#         (5) acc_num of  beam, hun, vj, respectively.
def calc_metric_batch_pair(batch_graph1, batch_graph2, dataset_name):
    best_ged = []
    prediction_ged_beam, exec_time_beam, search_space_beam = [], [], []
    prediction_ged_hun, exec_time_hun, search_space_hun = [], [], []
    prediction_ged_vj, exec_time_vj, search_space_vj = [], [], []
    acc_num_beam, acc_num_hun, acc_num_vj = 0, 0, 0

    assert (len(batch_graph1) == len(batch_graph2))
    for i in range(len(batch_graph1)):
        g1 = batch_graph1[i]
        g2 = batch_graph2[i]

        cost_beam, t_beam, ss_beam = ged(g1, g2, "beam", dataset_name, debug=False, timeit=False)
        cost_hun, t_hun, ss_hun = ged(g1, g2, "hungarian", dataset_name, debug=False, timeit=False)
        cost_vj, t_vj, ss_vj = ged(g1, g2, "vj", dataset_name, debug=False, timeit=False)

        cost = min(cost_beam, cost_hun, cost_vj)
        g1size = len(g1)
        g2size = len(g2)
        best_ged.append(normalize_ged(g1size, g2size, cost))
        prediction_ged_beam.append(normalize_ged(g1size, g2size, cost_beam))
        exec_time_beam.append(t_beam)
        search_space_beam.append(ss_beam)

        prediction_ged_hun.append(normalize_ged(g1size, g2size, cost_hun))
        exec_time_hun.append(t_hun)
        search_space_hun.append(ss_hun)

        prediction_ged_vj.append(normalize_ged(g1size, g2size, cost_vj))
        exec_time_vj.append(t_vj)
        search_space_vj.append(ss_vj)
        if int(cost_beam) == int(cost): acc_num_beam += 1
        if int(cost_hun) == int(cost): acc_num_hun += 1
        if int(cost_vj) == int(cost): acc_num_vj += 1
    return np.array(best_ged), \
           np.array(prediction_ged_beam), np.array(prediction_ged_hun), np.array(prediction_ged_vj), \
           np.array(exec_time_beam), np.array(exec_time_hun), np.array(exec_time_vj),  \
           np.array(ss_beam), np.array(ss_hun), np.array(ss_vj), \
           acc_num_beam, acc_num_hun, acc_num_vj


def score(graphs_path:str, label_path:str, dataset:str, batch_size:int):
    all_graphs = load_graph_from_raw(graphs_path)       # networkx of graphs
    with open(os.path.join(label_path, "CANCER_labels.pt"), 'rb') as f:
        pairs = pickle.load(f)
    testing_graphs = pairs[len(pairs) // 10 * 8: ]
    ############################################################### for test
    # testing_graphs = testing_graphs[0: 100]
    ###############################################################
    rownum = 20                                         # similar to the training size of  AIDS, IMDB dataset
    colnum = int(len(testing_graphs) / rownum)

    all_beam, all_hun, all_vj =[], [], []

    l1_list_beam, l2_list_beam, rho_list_beam, tau_list_beam, pat10_list_beam, pat20_list_beam, tm_list_beam, ss_list_beam = [], [], [], [], [], [], [], []
    l1_list_hun, l2_list_hun, rho_list_hun, tau_list_hun, pat10_list_hun, pat20_list_hun, tm_list_hun, ss_list_hun = [], [], [], [], [], [], [], []
    l1_list_vj, l2_list_vj, rho_list_vj, tau_list_vj, pat10_list_vj, pat20_list_vj, tm_list_vj, ss_list_vj = [], [], [], [], [], [], [], []
    acc_num_beam, acc_num_hun, acc_num_vj = 0, 0, 0
    exec_num = 0

    tq = tqdm(total= rownum * colnum, ascii=True, position=0)
    start_t = time.time()
    for i in range(rownum):
        row_graphs = testing_graphs[i * colnum: (i + 1) * colnum]
        prediction_row_beam, prediction_row_hun, prediction_row_vj, best_row = \
            np.full(colnum, 1e-10), np.full(colnum, 1e-10), np.full(colnum, 1e-10), np.full(colnum, 1e-10)
        for start_idx in range(0, len(row_graphs), batch_size):  # 从0到len(self.training_graphs)间隔 batch size
            data_list = row_graphs[start_idx:start_idx + batch_size]
            nx_graph_list1, nx_graph_list2 = [], []
            for j in range(len(data_list)):
                item = data_list[j]
                g1 = all_graphs[item['rowid']]
                g2 = all_graphs[item['colid']]
                nx_graph_list1.append(g1)
                nx_graph_list2.append(g2)

            best_batch, \
                pred_batch_beam, pred_batch_hun, pred_batch_vj, \
                exec_time_batch_beam, exec_time_batch_hun, exec_time_batch_vj, \
                ss_batch_beam, ss_batch_hun, ss_batch_vj, \
                acc_batch_beam, acc_batch_hun, acc_batch_vj\
                = calc_metric_batch_pair(nx_graph_list1, nx_graph_list2, dataset)

            exec_num += len(nx_graph_list1)
            acc_num_beam += acc_batch_beam
            acc_num_hun += acc_batch_hun
            acc_num_vj += acc_batch_vj
            prediction_row_beam[start_idx:start_idx + len(nx_graph_list1)] = pred_batch_beam
            prediction_row_hun[start_idx:start_idx + len(nx_graph_list1)] = pred_batch_hun
            prediction_row_vj[start_idx:start_idx + len(nx_graph_list1)] = pred_batch_vj
            best_row[start_idx:start_idx + len(nx_graph_list1)] = best_batch
            tm_list_beam.append(np.mean(exec_time_batch_beam))
            tm_list_hun.append(np.mean(exec_time_batch_hun))
            tm_list_vj.append(np.mean(exec_time_batch_vj))
            ss_list_beam.append(np.mean(ss_batch_beam))
            ss_list_hun.append(np.mean(ss_batch_hun))
            ss_list_vj.append(np.mean(ss_batch_vj))
            tq.update(len(nx_graph_list1))
            tq.set_description("MAE={:.3f}".format( np.mean( np.abs(pred_batch_beam - best_batch)) ))

        # store the results of the three algs
        all_beam.append(prediction_row_beam)
        all_hun.append(prediction_row_hun)
        all_vj.append(prediction_row_vj)

        # update beam results
        rho_list_beam.append(calculate_ranking_correlation(spearmanr, prediction_row_beam, best_row))
        tau_list_beam.append(calculate_ranking_correlation(kendalltau, prediction_row_beam, best_row))
        pat10_list_beam.append(calculate_prec_at_k(10, prediction_row_beam, best_row))
        pat20_list_beam.append(calculate_prec_at_k(20, prediction_row_beam, best_row))
        l1_list_beam.append(np.mean(np.abs(prediction_row_beam - best_row)))
        l2_list_beam.append(np.mean(np.square(prediction_row_beam - best_row)))
        # update hun results
        rho_list_hun.append(calculate_ranking_correlation(spearmanr, prediction_row_hun, best_row))
        tau_list_hun.append(calculate_ranking_correlation(kendalltau, prediction_row_hun, best_row))
        pat10_list_hun.append(calculate_prec_at_k(10, prediction_row_hun, best_row))
        pat20_list_hun.append(calculate_prec_at_k(20, prediction_row_hun, best_row))
        l1_list_hun.append(np.mean(np.abs(prediction_row_hun - best_row)))
        l2_list_hun.append(np.mean(np.square(prediction_row_hun - best_row)))
        # update vj results
        rho_list_vj.append(calculate_ranking_correlation(spearmanr, prediction_row_vj, best_row))
        tau_list_vj.append(calculate_ranking_correlation(kendalltau, prediction_row_vj, best_row))
        pat10_list_vj.append(calculate_prec_at_k(10, prediction_row_vj, best_row))
        pat20_list_vj.append(calculate_prec_at_k(20, prediction_row_vj, best_row))
        l1_list_vj.append(np.mean(np.abs(prediction_row_vj - best_row)))
        l2_list_vj.append(np.mean(np.square(prediction_row_vj - best_row)))

    # output beam results
    print("********the beam metrics of cancer datasets********")
    print("norm_mae: " + str(round( np.mean(l1_list_beam), 5)))
    print("norm_mse: " + str(round( np.mean(l2_list_beam), 5)))
    print("acc: " + str(round( acc_num_beam *1.0 / ( rownum* colnum ) , 5)))
    print("Spearman's_rho: " + str(round( np.mean(rho_list_beam), 5)))
    print("Kendall's_tau: " + str(round( np.mean(tau_list_beam), 5)))
    print("p@10: " + str(round( np.mean(pat10_list_beam), 5 )))
    print("p@20: " + str(round( np.mean(pat20_list_beam), 5)))
    print("search_space: " + str( round( np.mean(ss_list_beam), 5)  ))
    print("average_time(msec.): " + str(round( np.mean(tm_list_beam), 5)))
    # output hun results
    print("********the hungarian metrics of cancer datasets********")
    print("norm_mae: " + str(round(np.mean(l1_list_hun), 5)))
    print("norm_mse: " + str(round(np.mean(l2_list_hun), 5)))
    print("acc: " + str(round(acc_num_hun * 1.0 / (rownum * colnum), 5)))
    print("Spearman's_rho: " + str(round(np.mean(rho_list_hun), 5)))
    print("Kendall's_tau: " + str(round(np.mean(tau_list_hun), 5)))
    print("p@10: " + str(round(np.mean(pat10_list_hun), 5)))
    print("p@20: " + str(round(np.mean(pat20_list_hun), 5)))
    print("search_space: " + str(round(np.mean(ss_list_hun), 5)))
    print("average_time(msec.): " + str(round(np.mean(tm_list_hun), 5)))
    # output vj results
    print("********the hungarian metrics of cancer datasets********")
    print("norm_mae: " + str(round(np.mean(l1_list_vj), 5)))
    print("norm_mse: " + str(round(np.mean(l2_list_vj), 5)))
    print("acc: " + str(round(acc_num_vj * 1.0 / (rownum * colnum), 5)))
    print("Spearman's_rho: " + str(round(np.mean(rho_list_vj), 5)))
    print("Kendall's_tau: " + str(round(np.mean(tau_list_vj), 5)))
    print("p@10: " + str(round(np.mean(pat10_list_vj), 5)))
    print("p@20: " + str(round(np.mean(pat20_list_vj), 5)))
    print("search_space: " + str(round(np.mean(ss_list_vj), 5)))
    print("average_time(msec.): " + str(round(np.mean(tm_list_vj), 5)))

    print("exec time(sec.):" + str(round( (time.time() - start_t), 5)))

    # store all feasible results of the cancer dataset. (.txt)
    #   rowid colid rgid cgid g1 g2 beam hun vj
    new_label_items = []
    wf = open(os.path.join(label_path, "fea_results.txt"), 'w')
    for i in range(rownum):
        row_graphs = testing_graphs[i * colnum: (i + 1) * colnum]
        for start_idx in range(0, len(row_graphs), batch_size):  # 从0到len(self.training_graphs)间隔 batch size
            data_list = row_graphs[start_idx:start_idx + batch_size]
            for j in range(len(data_list)):
                item = data_list[j]
                line = "{} {} {} {} {} {} {} {} {}\n".format( item['rowid'], item['colid'], item['rgid'], item['cgid'], \
                                                        item['g1'], item['g2'], all_beam[i][j],  all_hun[i][j],  all_vj[i][j])
                copy_item = copy.deepcopy(item)
                copy_item['beam'] = all_beam[i][j]
                copy_item['hun'] = all_hun[i][j]
                copy_item['vj'] = all_vj[i][j]
                new_label_items.append(copy_item)
                wf.write(line)
    wf.close()
    # store all feasible results of the cancer dataset. (.pt)
    with open(os.path.join(label_path, "fea_results.pt"), 'wb') as f:
        pickle.dump(new_label_items, f)


if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    batch_size = 10
    if platform.system().lower() == 'windows':
        path = "D:/workspace/GED/ourGED/datasets/{}/"
    else:
        path = "/home/LAB/liujf/workspace/GED/ourGED/datasets/{}/"

    path = path.format(dataset_name)
    score(os.path.join(path, "raw"), os.path.join(path, "processed"), dataset_name, batch_size)

"""
python src/traditional_algs_cancer.py CANCER > cancer_all_trads.log 2>&1 &
"""

