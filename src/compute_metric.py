#encoding=utf-8
import os
import glob
import numpy as np
import torch.functional as F
import torch
from src.utils import *
from scipy.stats import spearmanr, kendalltau

module_path = os.path.dirname(os.path.dirname(__file__))
#compuate imdb
# fea_results_beam.txt fea_results_hungarian.txt, fea_results_vj.txt, fea_results_mata.txt
def load_fea_results(fname:str, line_num:int ):
    sim_score = []
    with open(fname, 'r') as fo:
        lines = fo.readlines()
        for line in lines:
            line = line.split()
            t = float(line[-1])
            sim_score.append(round(t, 5))
    return np.array(sim_score)[0:line_num]

def load_fea_results_prune(fname:str, line_num:int ):
    sim_score = []
    with open(fname, 'r') as fo:
        lines = fo.readlines()
        for line in lines:
            line = line.split()
            if int(line[1]) >= 225: continue
            t = float(line[-1])
            sim_score.append(round(t, 5))
    return np.array(sim_score)[0:line_num]

def load_fea_results_cancer(fname:str, line_num:int ):
    sim_score_beam, sim_score_hun, sim_score_vj = [], [], []
    with open(fname, 'r') as fo:
        lines = fo.readlines()
        assert (line_num == len(lines))
        for line in lines:
            line = line.split()
            sim_score_beam.append(round(float(line[6]),5))
            sim_score_hun.append(round(float(line[7]),5))
            sim_score_vj.append(round(float(line[8]),5))
    return  sim_score_beam, sim_score_hun, sim_score_vj




def print_metric(pred, gt, test_size:int, train_size:int, alg_type:str):
    print( "################{}################".format(alg_type))
    assert (len(pred) == train_size * test_size)
    assert (len(gt) == train_size * test_size)
    pred_item, gt_item = np.full(train_size, 1e-10), np.full(train_size, 1e-10)
    l1_list, l2_list, rho_list, tau_list, prec_at_10_list, prec_at_20_list = [], [], [], [], [], []
    acc_num = 0

    for i in range (test_size):
        for j in range(train_size):
            pred_item[j] = pred[i * train_size + j]
            gt_item[j] = gt[i * train_size + j]
        l1_list.append(np.average(F.l1_loss(torch.from_numpy(pred_item) , torch.from_numpy(gt_item), reduction='none').detach().numpy()))
        l2_list.append(np.average(F.mse_loss(torch.from_numpy(pred_item) , torch.from_numpy(gt_item), reduction='none').detach().numpy()))
        acc_num = acc_num + len(gt_item) - torch.count_nonzero(torch.from_numpy(pred_item) - torch.from_numpy(gt_item), dim=-1).item()
        rho_list.append(calculate_ranking_correlation(spearmanr, pred_item, gt_item))
        tau_list.append(calculate_ranking_correlation(kendalltau, pred_item, gt_item))
        prec_at_10_list.append(calculate_prec_at_k(10, pred_item, gt_item))
        prec_at_20_list.append(calculate_prec_at_k(20, pred_item, gt_item))

    print("mae: " + str(round(np.mean(l1_list), 5)))
    print("mse: " + str(round(np.mean(l2_list), 5)))
    print("acc: " + str(round(acc_num * 1.0 / len(pred), 5)))
    print("Spearman's_rho: " + str(round(np.nanmean(rho_list), 5)))
    print("Kendall's_tau: " + str(round(np.nanmean(tau_list), 5)))
    print("p@10: " + str(round(np.mean(prec_at_10_list), 5)))
    print("p@20: " + str(round(np.mean(prec_at_20_list), 5)))


# handle aids
def handle_aids(fpath:str):
    test_size, train_size = 140, 420        #420:140:140
    line_num = test_size * train_size
    beam_fname = "fea_results_beam.txt"
    hun_fname = "fea_results_hungarian.txt"
    vj_fname = "fea_results_vj.txt"
    # mata_fname = "fea_results_mata.txt"
    mata_fname = "fea_results_mata_b128_k6.txt"
    simgnn_fname = "fea_results_simgnn.txt"
    genn_fname = "fea_results_genn.txt"
    gennas_fname = "fea_results_gennas.txt"
    greed_fname = "fea_results_greed.txt"

    beam_score = load_fea_results(os.path.join(fpath, beam_fname), line_num)
    hun_score = load_fea_results(os.path.join(fpath, hun_fname), line_num)
    vj_score =  load_fea_results(os.path.join(fpath, vj_fname), line_num)
    mata_score = load_fea_results(os.path.join(fpath, mata_fname), line_num)
    simgnn_score = load_fea_results(os.path.join(fpath, simgnn_fname), line_num)
    genn_score = load_fea_results(os.path.join(fpath, genn_fname), line_num)
    gennas_score = load_fea_results(os.path.join(fpath, gennas_fname), line_num)
    greed_score = load_fea_results(os.path.join(fpath, greed_fname), line_num)


    gt_score = np.full(len(beam_score), 1e-10)

    dname = "AIDS700nef"
    from torch_geometric.datasets import GEDDataset
    ori_train = GEDDataset('{}/datasets/{}'.format(module_path, dname), dname, train=True)
    nged_matrix = ori_train.norm_ged
    nged_matrix = torch.exp(-nged_matrix)
    cur_idx = 0
    for i in range(test_size):
        for j in range(train_size):
            gt_score[cur_idx] = round(nged_matrix[test_size+train_size+i][j].item(),5)
            cur_idx += 1

    index_score = np.full(4, 1e-10)
    for i in range(len(beam_score)):

        index_score[0] = beam_score[i]
        index_score[1] = hun_score[i]
        index_score[2] = vj_score[i]
        index_score[3] = mata_score[i]
        max_score = np.max(index_score)

    print_metric(mata_score, gt_score, test_size, train_size, "mata")
    print_metric(beam_score, gt_score, test_size, train_size, "beam")
    print_metric(hun_score, gt_score, test_size, train_size, "hungarian")
    print_metric(vj_score, gt_score, test_size, train_size, "vj")
    print_metric(simgnn_score, gt_score, test_size, train_size, "simgnn")
    print_metric(genn_score, gt_score, test_size, train_size, "genn")
    # print_metric(gennas_score, gt_score, test_size, train_size, "gennA*")
    print_metric(greed_score, gt_score, test_size, train_size, "greed")



# handle imdb
def handle_imdb(fpath:str):
    test_size, train_size = 300, 130        #900:300:300, 900/4 = 225
    line_num = test_size * train_size
    beam_fname = "fea_results_beam.txt"
    hun_fname = "fea_results_hungarian.txt"
    vj_fname = "fea_results_vj.txt"
    # vj_fname = "imdb_fea_results_mata_b128_k6.txt"

    mata_fname_before =  "imdb_fea_results_mata_b128_k6.txt"

    mata_fname = "fea_results_mata_b128_k6_True_t3.txt"     # fea_results_mata_b128_k1  imdb_fea_results_mata_b128_k6
    # mata_fname = "fea_results_mata_b128_k6_True_t3.txt"
    """
    k1 = "fea_results_mata_b128_k1.txt"
    k2 = "fea_results_mata_b128_k6_False_t1.txt"
    k3 = "fea_results_mata_b128_k6_False_t2.txt"
    k4 = "fea_results_mata_b128_k6_True_t3.txt"
    k5 = "imdb_fea_results_mata_b128_k6.txt"
    """

    simgnn_fname = "fea_results_simgnn.txt"
    genn_fname = "fea_results_genn.txt"  # fea_results_genn
    gmn_fname = "fea_results_gmn.txt"  # fea_results_genn
    greed_fname = "fea_results_greed.txt"  # fea_results_genn

    beam_score = load_fea_results(os.path.join(fpath, beam_fname), line_num)
    hun_score = load_fea_results(os.path.join(fpath, hun_fname), line_num)
    vj_score =  load_fea_results(os.path.join(fpath, vj_fname), line_num)
    mata_score = load_fea_results(os.path.join(fpath, mata_fname), line_num)
    mata_score_before = load_fea_results(os.path.join(fpath, mata_fname_before), line_num)

    simgnn_score = load_fea_results(os.path.join(fpath, simgnn_fname), line_num)
    genn_score = load_fea_results_prune(os.path.join(fpath, genn_fname), line_num)
    gmn_score = load_fea_results(os.path.join(fpath, gmn_fname), line_num)
    greed_score = load_fea_results(os.path.join(fpath, greed_fname), line_num)


    gt_score = np.full(len(beam_score), 1e-10)

    dname = "IMDBMulti"
    from torch_geometric.datasets import GEDDataset
    ori_train = GEDDataset('{}/datasets/{}'.format(module_path, dname), dname, train=True)
    nged_matrix = ori_train.norm_ged
    nged_matrix = torch.exp(-nged_matrix)
    cur_idx = 0
    for i in range(test_size):
        for j in range(train_size):
            gt_score[cur_idx] = round(nged_matrix[j][1200 + i].item(),5)
            cur_idx += 1

    index_score = np.full(4, 1e-10)
    sup1, sup2, sup3 = 0, 0, 0
    for i in range(len(beam_score)):

        index_score[0] = beam_score[i]
        index_score[1] = hun_score[i]
        index_score[2] = vj_score[i]
        index_score[3] = mata_score[i]

        # max_score = np.max(index_score)
        # gt_score[i] = max_score
        if mata_score[i] > gt_score[i]:
            gt_score[i] = mata_score[i]
        if mata_score_before[i]> gt_score[i]:
            gt_score[i] = mata_score_before[i]

    # print("{}; rate {}".format(sup, sup*1.0/len(beam_score)))
    print("{}, {}, {}".format(sup1, sup2, sup3))
    print_metric(mata_score, gt_score, test_size, train_size, "mata")
    print_metric(beam_score, gt_score, test_size, train_size, "beam")
    print_metric(hun_score, gt_score, test_size, train_size, "hungarian")
    print_metric(vj_score, gt_score, test_size, train_size, "vj")
    print_metric(simgnn_score, gt_score, test_size, train_size, "simgnn")
    print_metric(genn_score, gt_score, test_size, train_size, "genn")
    print_metric(gmn_score, gt_score, test_size, train_size, "gmn")
    print_metric(greed_score, gt_score, test_size, train_size, "greed")


# handle cancer
def handle_cancer(fpath:str):
    test_size, train_size = 100, 200        #6w:2w:2w
    line_num = train_size * test_size
    three_trad_fname = "fea_results.txt"
    # mata_fname = "fea_results_mata.txt"
    # mata_fname = "cancer_fea_results_mata_b128_k8.txt"
    mata_fname = "fea_results_mata_b128_k8_True_t3.txt" #  fea_results_mata_b128_k1  fea_results_mata_b128_k8_True_t3

    simgnn_fname = "fea_results_simgnn.txt"
    genn_fname = "fea_results_genn.txt"
    gmn_fname = "fea_results_gmn.txt"
    greed_fname = "fea_results_greed.txt"

    beam_score, hun_score,  vj_score =  load_fea_results_cancer(os.path.join(fpath, three_trad_fname), line_num)
    mata_score = load_fea_results(os.path.join(fpath, mata_fname), line_num)
    simgnn_score = load_fea_results(os.path.join(fpath, simgnn_fname), line_num)
    genn_score = load_fea_results(os.path.join(fpath, genn_fname), line_num)
    gmn_score = load_fea_results(os.path.join(fpath, gmn_fname), line_num)
    greed_score = load_fea_results(os.path.join(fpath, greed_fname), line_num)

    gt_score = np.full(len(beam_score), 1e-10)

    index_score = np.full(4, 1e-10)
    for i in range(len(beam_score)):
        index_score[0] = beam_score[i]
        index_score[1] = hun_score[i]
        index_score[2] = vj_score[i]
        index_score[3] = mata_score[i]
        max_score = np.max(index_score)
        gt_score[i] = max_score

    print_metric(mata_score, gt_score, test_size, train_size, "mata")
    print_metric(beam_score, gt_score, test_size, train_size, "beam")
    print_metric(hun_score, gt_score, test_size, train_size, "hungarian")
    print_metric(vj_score, gt_score, test_size, train_size, "vj")
    print_metric(simgnn_score, gt_score, test_size, train_size, "simgnn")
    print_metric(genn_score, gt_score, test_size, train_size, "genn")
    print_metric(gmn_score, gt_score, test_size, train_size, "gmn")
    print_metric(greed_score, gt_score, test_size, train_size, "greed")






def compute_topk_aids(fpath):
    test_size, train_size = 140, 420  # 420:140:140
    line_num = test_size * train_size
    k5 = "fea_results_mata_b128_k5.txt"
    k6 = "fea_results_mata_b128_k6.txt"
    k7 = "fea_results_mata_b128_k7.txt"
    k8 = "fea_results_mata_b128_k8.txt"
    k9 = "fea_results_mata_b128_k9.txt"
    k10 = "fea_results_mata_b128_k10.txt"

    k5_score = load_fea_results(os.path.join(fpath, k5), line_num)
    k6_score = load_fea_results(os.path.join(fpath, k6), line_num)
    k7_score = load_fea_results(os.path.join(fpath, k7), line_num)
    k8_score = load_fea_results(os.path.join(fpath, k8), line_num)
    k9_score = load_fea_results(os.path.join(fpath, k9), line_num)
    k10_score = load_fea_results(os.path.join(fpath, k10), line_num)

    gt_score = np.full(len( k5_score), 1e-10)

    dname = "AIDS700nef"
    from torch_geometric.datasets import GEDDataset
    ori_train = GEDDataset('{}/datasets/{}'.format(module_path, dname), dname, train=True)
    nged_matrix = ori_train.norm_ged
    nged_matrix = torch.exp(-nged_matrix)
    cur_idx = 0
    for i in range(test_size):
        for j in range(train_size):
            gt_score[cur_idx] = round(nged_matrix[test_size + train_size + i][j].item(), 5)
            cur_idx += 1

    print_metric(k5_score, gt_score, test_size, train_size, "topk 5")
    print_metric(k6_score, gt_score, test_size, train_size, "topk 6")
    print_metric(k7_score, gt_score, test_size, train_size, "topk 7")
    print_metric(k8_score, gt_score, test_size, train_size, "topk 8")
    print_metric(k9_score, gt_score, test_size, train_size, "topk 9")
    print_metric(k10_score, gt_score, test_size, train_size, "topk 10")


def compute_topk_imdb(fpath):
    test_size, train_size = 300, 130   # 140, 420
    line_num = test_size * train_size
    k5 = "fea_results_mata_b128_k5.txt"
    k6 = "fea_results_mata_b128_k6.txt"
    k7 = "fea_results_mata_b128_k7.txt"
    k8 = "fea_results_mata_b128_k8.txt"
    k9 = "fea_results_mata_b128_k9.txt"
    k10 = "fea_results_mata_b128_k10.txt"

    k5_score = load_fea_results(os.path.join(fpath, k5), line_num)
    k6_score = load_fea_results(os.path.join(fpath, k6), line_num)
    k7_score = load_fea_results(os.path.join(fpath, k7), line_num)
    k8_score = load_fea_results(os.path.join(fpath, k8), line_num)
    k9_score = load_fea_results(os.path.join(fpath, k9), line_num)
    k10_score = load_fea_results(os.path.join(fpath, k10), line_num)

    gt_score = np.full(len( k5_score), 1e-10)

    dname = "IMDBMulti"
    from torch_geometric.datasets import GEDDataset
    ori_train = GEDDataset('{}/datasets/{}'.format(module_path, dname), dname, train=True)
    nged_matrix = ori_train.norm_ged
    nged_matrix = torch.exp(-nged_matrix)
    cur_idx = 0
    for i in range(test_size):
        for j in range(train_size):
            gt_score[cur_idx] = round(nged_matrix[j][1200 + i].item(), 5)
            cur_idx += 1

    index_score = np.full(6, 1e-10)
    for i in range(len(k5_score)):
        index_score[0] = k5_score[i]
        index_score[1] = k6_score[i]
        index_score[2] = k7_score[i]
        index_score[3] = k8_score[i]
        index_score[4] = k9_score[i]
        index_score[5] = k10_score[i]
        max_score = np.max(index_score)
        # gt_score[i] = max_score
        if max_score > gt_score[i]:
            gt_score[i] = max_score

    print_metric(k5_score, gt_score, test_size, train_size, "topk 5")
    print_metric(k6_score, gt_score, test_size, train_size, "topk 6")
    print_metric(k7_score, gt_score, test_size, train_size, "topk 7")
    print_metric(k8_score, gt_score, test_size, train_size, "topk 8")
    print_metric(k9_score, gt_score, test_size, train_size, "topk 9")
    print_metric(k10_score, gt_score, test_size, train_size, "topk 10")


def compute_topk_cancer(fpath):
    test_size, train_size = 100, 200
    line_num = test_size * train_size
    k5 = "fea_results_mata_b128_k5.txt"
    k6 = "fea_results_mata_b128_k6.txt"
    k7 = "fea_results_mata_b128_k7.txt"
    k8 = "fea_results_mata_b128_k8.txt"
    k9 = "fea_results_mata_b128_k9.txt"
    k10 = "fea_results_mata_b128_k10.txt"
    three_trad_fname = "fea_results.txt"
    beam_score, hun_score, vj_score = load_fea_results_cancer(os.path.join(fpath, three_trad_fname), line_num)

    k5_score = load_fea_results(os.path.join(fpath, k5), line_num)
    k6_score = load_fea_results(os.path.join(fpath, k6), line_num)
    k7_score = load_fea_results(os.path.join(fpath, k7), line_num)
    k8_score = load_fea_results(os.path.join(fpath, k8), line_num)
    k9_score = load_fea_results(os.path.join(fpath, k9), line_num)
    k10_score = load_fea_results(os.path.join(fpath, k10), line_num)

    index_score = np.full(9, 1e-10)

    gt_score = np.full(len(k5_score), 1e-10)

    for i in range(len(beam_score)):
        index_score[0] = k5_score[i]
        index_score[1] = k6_score[i]
        index_score[2] = k7_score[i]
        #index_score[3] = k8_score[i]
        #index_score[4] = k9_score[i]
        #index_score[5] = k10_score[i]
        index_score[6] = beam_score[i]
        index_score[7] = hun_score[i]
        index_score[8] = vj_score[i]
        gt_score[i] =  np.max(index_score)

    print_metric(k5_score, gt_score, test_size, train_size, "topk 5")
    print_metric(k6_score, gt_score, test_size, train_size, "topk 6")
    print_metric(k7_score, gt_score, test_size, train_size, "topk 7")
    print_metric(k8_score, gt_score, test_size, train_size, "topk 8")
    print_metric(k9_score, gt_score, test_size, train_size, "topk 9")
    print_metric(k10_score, gt_score, test_size, train_size, "topk 10")

def load_nx_graph(r_path):
    import networkx as nx
    ids, Ns = [], []
    names = glob.glob(osp.join(r_path, '*.gexf'))
    ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))
    data_list = []
    for i, idx in enumerate(ids[-1]):
        i = i if len(ids) == 1 else i + len(ids[0])
        # Reading the raw `*.gexf` graph:
        G = nx.read_gexf(osp.join(r_path, f'{idx}.gexf'))  # Different from ADIS and IMDB, the nodes of CANCER are already contiguous, hence relabel the index of nodes is not required
        data_list.append(G)
    return data_list

def do_statis():
    # graphs, graph pairs, min, max,  avg V , avg(E/V)

    module_path = os.path.dirname(os.path.dirname(__file__))
    # aids_path_train = os.path.join(module_path, 'datasets/AIDS700nef/raw/AIDS700nef/train')
    # aids_path_test = os.path.join(module_path, 'datasets/AIDS700nef/raw/AIDS700nef/test')

    # aids_path_train = os.path.join(module_path, 'datasets/IMDBMulti/raw/IMDBMulti/train')
    # aids_path_test = os.path.join(module_path, 'datasets/IMDBMulti/raw/IMDBMulti/test')

    # train_graphs =  load_nx_graph(aids_path_train)
    # test_graphs =   load_nx_graph(aids_path_test)
    # aids_graphs = train_graphs + test_graphs
    aids_graphs = load_nx_graph( os.path.join(module_path, 'datasets/CANCER/raw'))

    min_v, max_v, avgV, avgE_V = 1000, 0, 0.0, 0.0
    for g in aids_graphs:
        if g.number_of_nodes() < min_v: min_v = g.number_of_nodes()
        if g.number_of_nodes() > max_v: max_v = g.number_of_nodes()
        avgV += g.number_of_nodes()
        avgE_V += 1.0*g.number_of_edges()/g.number_of_nodes()
    print("min_v:{} \n max_v: {}\navgV:{}\navgE/V:{}\n".format(min_v, max_v, avgV/(len(aids_graphs)), avgE_V/(len(aids_graphs)) ))

if __name__ == '__main__':

    ############### solution quality
    fpath = "D:/PHD/projects/learning-ged/exp-draft/cvpr-results/cancer/"
    handle_cancer(fpath)

    # fpath = "D:/PHD/projects/learning-ged/exp-draft/cvpr-results/imdb/"
    # handle_imdb(fpath)

    # fpath = "D:/PHD/projects/learning-ged/exp-draft/cvpr-results/aids/"
    # handle_aids(fpath)
    ###############

    ############### topk results
    # fpath = "xxx/aids"
    # compute_topk_aids(fpath)

    # fpath = "xxx/imdb"
    # compute_topk_imdb(fpath)

    # fpath = "xxx/cancer"
    # compute_topk_cancer(fpath)
    ###############
    ###################ablation study
    # D:\PHD\phd4\my-papers\GED\exp-draft\cvpr-results\ablation\imdb
