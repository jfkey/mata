#encoding=utf-8
import os.path
import glob
import copy
import time
import pickle

import torch_geometric
from tqdm import tqdm, trange
import torch.nn
from torch_geometric.nn import GCNConv, GINConv, SplineConv
from src.layers import AttentionModule, NeuralTensorNetwork, Affinity, Sinkhorn, NorSim, soft_topk, greedy_perm
from scipy.stats import spearmanr, kendalltau
from torch_geometric.data import DataLoader, Data, Batch
import torch_geometric.transforms as T
from src.myloss_func import *
from src.randomWalk import AddRandomWalkPE
from src.mydegree import MyDegree
from src.cancer_dataset import cancerData


EPS = 1e-8
PRINT_TIMING = False
upper_bound = 1000
SK_ITER_NUM = 6
SK_EPSILON = 1.0e-4
SK_TAU = 1.0
module_path = os.path.dirname(os.path.dirname(__file__))


class OurNN(torch.nn.Module):
    def __init__(self, args, num_of_labels, app_astar):
        super(OurNN, self).__init__()

        self.app_astar = app_astar
        self.args = args
        self.num_labels = num_of_labels
        self.setup_layers()

    # the number of bottleneck features
    def calculate_bottleneck_features(self):
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):

        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.args.filter_1, self.args.filter_1)  # self.num_labels
            self.convolution_2 = GCNConv(self.args.filter_1, self.args.filter_2)
            self.convolution_3 = GCNConv(self.args.filter_2, self.args.filter_3)
        elif self.args.gnn_operator == 'spline':
            self.convolution_1 = SplineConv(self.args.filter_1, self.args.filter_1)  # #  self.num_labels
            self.convolution_2 = SplineConv(self.args.filter_1, self.args.filter_2)
            self.convolution_3 = SplineConv(self.args.filter_2, self.args.filter_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filter_1, self.args.filter_1),
                torch.nn.ReLU(),
            )
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filter_1, self.args.filter_2),
                torch.nn.ReLU(),
            )
            nn3 = torch.nn.Sequential(
                torch.nn.Linear(self.args.filter_2, self.args.filter_3),
                torch.nn.ReLU(),
            )
            self.convolution_1 = GINConv(nn1, train_eps=True)
            self.convolution_2 = GINConv(nn2, train_eps=True)
            self.convolution_3 = GINConv(nn3, train_eps=True)
        else:
            raise NotImplementedError('Unknown GNN-Operator.')

        self.init_layer = torch.nn.Sequential(
            torch.nn.Linear(self.num_labels, self.args.filter_1),
            torch.nn.ReLU()
        )

        self.degree_emb = torch.nn.Parameter(torch.Tensor(self.args.max_degree, self.args.max_degree))
        torch.nn.init.xavier_uniform_(self.degree_emb)
        self.attention = AttentionModule(self.args.filter_3)
        # self.tensor_network = NeuralTensorNetwork(self.args)
        self.affinity = Affinity(self.args.filter_3)
        self.sinkhorn = Sinkhorn(max_iter=SK_ITER_NUM, epsilon=SK_EPSILON, tau=SK_TAU)
        self.nor_sim = NorSim()
        self.scoring_layer = torch.nn.Sequential(
            torch.nn.Linear(8 * self.args.filter_3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def calculate_similarity(self, abstract_feature_1, abstract_feature_2):
        sim_matrix = torch.mm(abstract_feature_1, abstract_feature_2)
        sim_matrix = torch.softmax(sim_matrix, dim=-1)
        return sim_matrix

    def convolutional_pass(self, edge_index, features, edge_weight=None):
        """
        Making convolutional pass
        :param edge_index: Edge indices
        :param features: Feature matrix
        :return: Abstract feature matrix
        """
        features_1 = self.convolution_1(features, edge_index)
        features = F.relu(features_1)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features_2 = self.convolution_2(features, edge_index)
        features = F.relu(features_2)
        features = F.dropout(features, p=self.args.dropout, training=self.training)
        features_3 = self.convolution_3(features, edge_index)
        return features_1, features_2, features_3

    def forward(self, data):
        g1, g2 = data["g1"], data["g2"]
        batch_1, batch_2 = g1.batch, g2.batch
        device = next(self.parameters()).device
        edge_index_1 = g1.edge_index.to(device)
        edge_index_2 = g2.edge_index.to(device)

        if self.args.nonstruc:
            feature_1 = g1.x
            feature_2 = g2.x
        else:
            feature_1 = torch.cat([g1.x, self.degree_emb[g1.cent_pe.squeeze(1)], g1.rw_pe], dim=1).to(device)
            feature_2 = torch.cat([g2.x, self.degree_emb[g2.cent_pe.squeeze(1)], g2.rw_pe], dim=1).to(device)

        feature_1 = self.init_layer(feature_1)
        feature_2 = self.init_layer(feature_2)

        # 对于每个图，abstract_feature_1: 节点个数x特征数
        g1_af_1, g1_af_2, g1_af_3 = self.convolutional_pass(edge_index_1,
                                                            feature_1)  # the first conv abstract features of g1
        g2_af_1, g2_af_2, g2_af_3 = self.convolutional_pass(edge_index_2, feature_2)

        rows = torch.bincount(g1.batch).to(device)
        cols = torch.bincount(g2.batch).to(device)
        sim_mat1 = self.affinity(feature_1, feature_2, batch_1, batch_2)
        gt_ks = torch.full([sim_mat1.shape[0]], self.args.topk / 2, device=device, dtype=torch.float)
        _, sim_mat1 = soft_topk(sim_mat1, gt_ks.view(-1), SK_ITER_NUM, SK_TAU, nrows=rows, ncols=cols, return_prob=True)

        sim_mat2 = self.affinity(g1_af_3, g2_af_3, batch_1, batch_2)
        _, sim_mat2 = soft_topk(sim_mat2, gt_ks.view(-1), SK_ITER_NUM, SK_TAU, nrows=rows, ncols=cols, return_prob=True)

        global_feature1 = torch.cat((feature_1, g1_af_1), dim=1)
        global_feature1 = torch.cat((global_feature1, g1_af_2), dim=1)
        global_feature1 = torch.cat((global_feature1, g1_af_3), dim=1)
        global_feature1 = torch_geometric.nn.global_add_pool(global_feature1, batch_1)

        global_feature2 = torch.cat((feature_2, g2_af_1), dim=1)
        global_feature2 = torch.cat((global_feature2, g2_af_2), dim=1)
        global_feature2 = torch.cat((global_feature2, g2_af_3), dim=1)
        global_feature2 = torch_geometric.nn.global_max_pool(global_feature2, batch_2)

        scores = torch.cat((global_feature1, global_feature2), dim=1)
        ged_score = self.scoring_layer(scores).view(-1)

        return ged_score, sim_mat1, sim_mat2

    @property
    def device(self):
        return next(self.parameters()).device


class OurNNTrainer(object):

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.process_dataset()
        self.init_astar()
        self.setup_model()

    def setup_model(self):
        self.model = OurNN(self.args, self.num_labels, self.app_astar)

    def load_nx_graph(self, r_path):
        import networkx as nx
        ids, Ns = [], []
        names = glob.glob(osp.join(r_path, '*.gexf'))
        ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))
        data_list = []
        for i, idx in enumerate(ids[-1]):
            i = i if len(ids) == 1 else i + len(ids[0])
            # Reading the raw `*.gexf` graph:
            G = nx.read_gexf(osp.join(r_path, f'{idx}.gexf'))  # 由于Cancer中的图中的节点已经是连续的了，所以不需要进行relabel操作
            G_str = nx2txt(G, str(idx), self.args.dataset)
            data_list.append(G_str)
        return data_list

    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")

        if self.args.dataset in ['AIDS700nef', 'LINUX', 'ALKANE', 'IMDBMulti']:  # torch_geometric datasets
            from torch_geometric.datasets import GEDDataset
            ori_train = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=True)
            self.training_graphs = ori_train[:len(ori_train) // 4 * 3]  # // 除 向下取整。 3: 1 : 1
            self.val_graphs = ori_train[len(ori_train) // 4 * 3:]
            self.testing_graphs = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=False)
            self.graph_size = len(self.training_graphs) + len(self.val_graphs) + len(self.testing_graphs)
        elif self.args.dataset in ['CANCER']:
            all_graphs = cancerData('datasets/{}'.format(self.args.dataset), self.args.dataset)
            nx_graphs  = self.load_nx_graph('datasets/{}/raw/'.format(self.args.dataset))
            self.all_graphs = all_graphs
            self.nx_graphs = nx_graphs
        else:
            raise ValueError('Unknown dataset name {}'.format(self.args.dataset))

        if self.args.nonstruc:
            graph_list = []
            for i in range(len(self.all_graphs)):
                gi = self.all_graphs[i]
                graph_list.append(gi)
            self.all_graphs.data = Batch.from_data_list(graph_list)
            self.num_labels = self.all_graphs.data.num_features

        else:
            max_degree = self.args.max_degree
            myDeg = MyDegree(max_degree)
            randwalk = AddRandomWalkPE(self.args.random_walk_step)
            transform = T.Compose([myDeg, randwalk])
            graph_list = []
            for i in range(len(self.all_graphs)):
                gi = transform(self.all_graphs[i])
                graph_list.append(gi)
            self.all_graphs.data = Batch.from_data_list(graph_list)
            self.num_labels = self.all_graphs.data.num_features + self.args.max_degree + self.args.random_walk_step

        pairs = self.all_graphs.pairs

        self.training_graphs = pairs[:len(pairs) // 10 * 6]
        self.val_graphs = pairs[len(pairs) // 10 * 6 : len(pairs) // 10 * 8]
        self.testing_graphs = pairs[len(pairs) // 10 * 8:]

        if self.args.debug:  # 6w:2w:2w
            self.training_graphs = self.training_graphs[0:2000]
            self.val_graphs = self.val_graphs[0:200]
            self.testing_graphs = self.testing_graphs[0:200]
            self.args.epochs = 30
            self.args.val_epochs = 22

    def init_astar(self):
        so_path = os.path.join(module_path, 'Astar', 'mata.so')
        app_astar = ctypes.cdll.LoadLibrary(so_path)  # app_astar: approximate astar
        app_astar.ged.restype = ctypes.c_char_p
        self.app_astar = app_astar
        # load all gexf data and convert to str

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        if self.args.dataset in ['CANCER']:
            random.shuffle(self.training_graphs)
            data_list = self.training_graphs[:self.args.batch_size]
        else:
            data_list_1 = list(self.training_graphs.shuffle()[:self.args.batch_size])  # 先对graph pairs shuffle，再从最开始的地方选择batch size的大小
            data_list_2 = list(self.training_graphs.shuffle()[:self.args.batch_size])

        nor_ged = torch.zeros(self.args.batch_size)
        data_list_1 = []
        data_list_2 = []
        list_map = []
        for i in range(self.args.batch_size):
            item = data_list[i]
            g1 = self.all_graphs.data[item['rowid']]
            g2 = self.all_graphs.data[item['colid']]
            ged = item['ged']
            if g1.num_nodes > g2.num_nodes:
                tmp = g1
                g1 = g2
                g2 = tmp
            nor_ged[i] = normalize_ged(g1.num_nodes, g2.num_nodes, ged)
            data_list_1.append(g1)
            data_list_2.append(g2)

        source_batch = Batch.from_data_list(data_list_1)
        target_batch = Batch.from_data_list(data_list_2)

        max_nodes_row = max(torch.bincount(source_batch.batch)).item()
        max_nodes_col = max(torch.bincount(target_batch.batch)).item()

        for i in range(self.args.batch_size):
            item = data_list[i]
            map_idx = torch.zeros((max_nodes_row, max_nodes_col), dtype=int)
            row = np.arange(0, len(item["int_map"]), dtype=int)
            col = np.array(item["int_map"], dtype=int)
            map_idx[[row, col]] = 1
            list_map.append(map_idx)
        batch_map = torch.stack(list_map, dim=0)

        # # perform transform
        new_data = dict()
        new_data["g1"] = source_batch.to(self.model.device)  # 一共有 batch size的源图
        new_data["g2"] = target_batch.to(self.model.device)  # 一共有 batch size的目标图
        new_data["map"] = batch_map.to(self.model.device)
        new_data["nor_ged"] = nor_ged.to(self.model.device)
        return new_data


    def get_index(self, g1_id, g2_id):  # 传入这里的参数: |g1| <= |g2| 如果g1=g2，那么g1的id要小于g2
        train_index = int(self.graph_size * 0.8)  # train val test, 6:2:2
        if (g1_id < train_index and g2_id < train_index):
            return train_index * g1_id + g2_id
        else:
            return train_index * train_index + (g1_id - train_index) * train_index + g2_id

    def process_batches(self, batch):
        self.optimizer.zero_grad()
        losses = torch.tensor(0.0, requires_grad=True)
        ged_score, simmat1, simmat2 = self.model(batch)
        if self.args.loss_type == 1:
            criterion = BCELoss()
        elif self.args.loss_type == 2:
            criterion = MultiMatchingLoss()
        elif self.args.loss_type == 3:
            criterion = GEDLoss(self.app_astar, self.nx_graphs, batch)
        else:
            print("Unknown loss type")
        rows = torch.bincount(batch["g1"].batch).to(self.model.device)
        cols = torch.bincount(batch["g2"].batch).to(self.model.device)

        if self.args.tasks == 1:
            losses = losses + F.mse_loss(ged_score, batch["nor_ged"], reduction="mean")
        elif self.args.tasks == 2:
            losses = losses + criterion(simmat2, batch["map"], rows, cols)
        else:
            losses = losses + 50 * F.mse_loss(ged_score, batch["nor_ged"], reduction="mean")
            losses = losses + criterion(simmat1, batch["map"], rows, cols)
            losses = losses + criterion(simmat2, batch["map"], rows, cols)

        losses.backward(retain_graph=True)
        self.optimizer.step()
        return losses.item()

    def fit(self):
        print('\nmodel training \n')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model.train()
        epochs = trange(self.args.epochs, ascii=True, leave=True, desc="Epoch", position=0)
        best_sim_score = float(0.0)
        for epoch in epochs:
            batches = self.create_batches()
            loss_score = self.process_batches(batches)
            loss = loss_score
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            if epoch > 0 and epoch % (self.args.val_epochs) == 0:
                cur_sim_score = self.score(test=False)
                if  cur_sim_score > best_sim_score:
                    print("update the model. The average similarity score of epoch({}):{}".format(epoch, cur_sim_score))
                    torch.save(self.model.state_dict(), 'best_model_{}_{}_e{}_lr{}_loss{}_t{}_stru{}_b{}.pt'.format(self.args.dataset, self.args.gnn_operator, self.args.epochs, self.args.learning_rate, self.args.loss_type, self.args.tasks, self.args.nonstruc, self.args.batch_size))
                    best_sim_score = cur_sim_score

    def init_Astar_sim_mat(self, data):
        g1, g2 = data["g1"], data["g2"]
        batch_1, batch_2 = g1.batch, g2.batch
        device = self.model.device
        
        ged_score = 0 
        rows = torch.bincount(g1.batch).max()
        cols = torch.bincount(g2.batch).max()
        batch_size = max(batch_1.max(), batch_2.max()) + 1  
        sim_mat1 = torch.rand(batch_size, rows, cols, device=device)

        return ged_score, sim_mat1, sim_mat1

    def score(self, test=True):
        num_train_graphs, num_test_graphs = 0, 0
        
        print("\n\nModel evaluation.\n")
        if test:
            testing_graphs = self.testing_graphs
            num_train_graphs = len(self.training_graphs)
            num_test_graphs = len(self.testing_graphs)
            testing_graphs = self.testing_graphs[0: len(self.testing_graphs)//8]
        else:
            testing_graphs = self.val_graphs
            testing_graphs = testing_graphs[0:2000]     # Part of the validation set 2w.
        self.model.eval()
 
        rownum = 20
        colnum = int(len(testing_graphs) / rownum)
        scores = np.zeros( (rownum, colnum))
        prediction_mat, ground_truth = np.full((rownum, colnum), 1e-10), np.full((rownum, colnum), 1e-10)
        rho_list, tau_list, prec_at_10_list, prec_at_20_list, avg_tree_size, avg_exc_time, acc = [], [], [], [], 0, 0, 0
        n1s, n2s = np.empty((rownum, colnum), dtype=int),  np.empty((rownum, colnum), dtype=int)
        t = tqdm(total=rownum* colnum, ascii=True, position=0)
        l1_list, l2_list = [], []
        superior_num, acc_num, fea_num, cur_idx, all_space, all_time = 0, 0, 0, 0, 0, 0
        # 需要求出的GED尽可能小，也就是similarity matrix 尽可能大
        ret_score, all_mata = [], []

        for i in range(rownum):
            row_graphs = testing_graphs[i*colnum:(i+1)*colnum]
            for start_idx in range(0, len(row_graphs), self.args.batch_size):  # 从0到len(self.training_graphs)间隔 batch size
                data_list = row_graphs[start_idx:start_idx + self.args.batch_size]
                nor_ged = torch.zeros(len(data_list))
                data_list_1 = []
                data_list_2 = []
                for j in range( len(data_list) ):
                    item = data_list[j]
                    g1 = self.all_graphs.data[item['rowid']]
                    g2 = self.all_graphs.data[item['colid']]
                    ged = item['ged']
                    if g1.num_nodes > g2.num_nodes:
                        tmp = g1
                        g1 = g2
                        g2 = tmp
                    nor_ged[j] = normalize_ged(g1.num_nodes, g2.num_nodes, ged)
                    data_list_1.append(g1)
                    data_list_2.append(g2)

                source_batch = Batch.from_data_list(data_list_1)
                target_batch = Batch.from_data_list(data_list_2)

                n1s[i, start_idx:start_idx + len(data_list)] = torch.bincount(source_batch.batch).detach().numpy()
                n2s[i, start_idx:start_idx + len(data_list)] = torch.bincount(target_batch.batch).detach().numpy()

                cur_idx = cur_idx + len(data_list)

                new_data = dict()
                new_data["g1"] = source_batch.to(self.model.device)  # 一共有 batch size的源图
                new_data["g2"] = target_batch.to(self.model.device)  # 一共有 batch size的目标图
                new_data["nor_ged"] = nor_ged
                target = new_data["nor_ged"].cpu()

                start_time = time.process_time()

                if self.args.beam == True:
                    pred_ged, simmat1, simmat2  = self.init_Astar_sim_mat(new_data) 
                    astar_prediction, search_space = self.do_astar_beam(simmat1.detach().cpu(), simmat2.detach().cpu(), new_data)
                else: 
                    pred_ged, simmat1, simmat2  = self.model(new_data)
                    astar_prediction, search_space = self.do_astar(simmat1.detach().cpu(), simmat2.detach().cpu(), new_data)

                all_time = all_time + time.process_time() - start_time

                all_space = all_space + sum(search_space[:])
                astar_pred_tensor = torch.from_numpy(np.array(astar_prediction))
                acc_num = acc_num + len(target) - torch.count_nonzero(target - astar_pred_tensor, dim=-1).item()
                sup = (target + 1e-3 - astar_pred_tensor < 0) == True
                superior_num = superior_num + torch.sum(sup).item()
                astar_pred_tensor = astar_pred_tensor *(~sup)
                target = target * (~sup)
                prediction_mat[i, start_idx:start_idx + len(data_list)] = np.array(astar_prediction)
                ground_truth[i, start_idx:start_idx + len(data_list)] = target

                scores[i, start_idx:start_idx + len(data_list)] = F.mse_loss(astar_pred_tensor, target, reduction='none').detach().numpy()
                l1_list.append(np.average(F.l1_loss(astar_pred_tensor, target, reduction='none').detach().numpy()))
                l2_list.append(np.average(F.mse_loss(astar_pred_tensor, target, reduction='none').detach().numpy()))
                t.update(len(data_list))
                t.set_description("AvgGED={:.3f}".format( np.mean(prediction_mat[i] )))

            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i]))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i]))
            ret_score.append(np.mean(prediction_mat[i]))
            all_mata.append(prediction_mat[i]) 

        print("acc:            " + str(round(acc_num * 1.0 / cur_idx + superior_num/cur_idx, 4)))
        print("#train pairs:   " + str(num_train_graphs) )
        print("#test pairs:    " + str(num_test_graphs) )
        print("#sample pairs:  " + str(cur_idx) )
        print("mae:            " + str(round(np.mean(l1_list), 4)))
        print("mse:            " + str(round(np.mean(l2_list), 4)))
        print("fea:            " + str(round(1.0)))
        print("Spearman's rho: " + str(round(np.nanmean(rho_list), 4)))
        print("Kendall's tau:  " + str(round(np.nanmean(tau_list), 4)))
        print("p@10:           " + str(round(np.mean(prec_at_10_list), 4)))
        print("p@20:           " + str(round(np.mean(prec_at_20_list), 4)))
        # print("search space:   " + str(round(all_space / cur_idx, 4)))
        print("average time:   " + str(round(all_time / cur_idx, 4)))
        print("superior_num:   " + str(round(superior_num/cur_idx, 4) ))

        if test:    # 完成了所有test集的计算，把计算的结果写到文件中
            new_label_items = []
            wf = open(os.path.join('datasets/{}/processed/'.format(self.args.dataset), "fea_results_mata_b{}_k{}_{}_t{}.txt".format(self.args.batch_size, self.args.topk, self.args.nonstruc, self.args.tasks)), 'w')
            for i in range(rownum):
                row_graphs = testing_graphs[i * colnum: (i + 1) * colnum]
                for start_idx in range(0, len(row_graphs), self.args.batch_size):  # 从0到len(self.training_graphs)间隔 batch size
                    data_list = row_graphs[start_idx:start_idx + self.args.batch_size]
                    for j in range(len(data_list)):
                        item = data_list[j]
                        line = "{} {} {} {} {} {} {}\n".format(item['rowid'], item['colid'], item['rgid'], item['cgid'],\
                                                                     item['g1'], item['g2'], all_mata[i][j],)
                        copy_item = copy.deepcopy(item)
                        copy_item['mata'] = all_mata[i][j]
                        new_label_items.append(copy_item)
                        wf.write(line)
            wf.close()
            with open(os.path.join('datasets/{}/processed/'.format(self.args.dataset), "fea_results_mata_b{}_k_{}_{}_t{}.pt".format(self.args.batch_size, self.args.topk, self.args.nonstruc, self.args.tasks)), 'wb') as f:
                pickle.dump(new_label_items, f)
        return np.mean( np.array(ret_score))

    def do_astar(self, sim_mat, sim_mat2, new_data):
        if len(sim_mat.shape) == 2:
            sim_mat = sim_mat.unsqueeze(0)
        if len(sim_mat2.shape) == 2:
            sim_mat2 = sim_mat2.unsqueeze(0)
        batch_num = sim_mat.shape[0]
        ged_prediction, search_space = [], []
        for b in range(batch_num):
            g1_id = new_data['g1'][b]['i'].item()
            g2_id = new_data['g2'][b]['i'].item()
            n1 = new_data['g1'][b].num_nodes
            n2 = new_data['g2'][b].num_nodes
            e1 = new_data['g1'][b].num_edges
            e2 = new_data['g2'][b].num_edges
            beam_size = get_beam_size(n1, n2, e1 / 2, e2 / 2, self.args.dataset)
            topk = min(self.args.topk, n1, n2)
            if topk == 0: topk = 1
            matching_nodes, matching_order = self.find_topk_hun(sim_mat[b, :n1, :n2].detach(),
                                                                sim_mat2[b, :n1, :n2].detach(), topk, n1, n2)
            matching_order[0], matching_order[1] = 0, 0
            astar_out = self.app_astar.ged(CT(self.nx_graphs[g1_id]), CT(self.nx_graphs[g2_id]),
                                           int1ArrayToPointer(matching_order),
                                           int1ArrayToPointer(matching_order), int2ArrayToPointer(matching_nodes),
                                           CT(2* topk), CT(beam_size))
            astar_out = astar_out.decode('ascii').split()
            pred = normalize_ged(n1, n2, int(astar_out[0]))
            ged_prediction.append(pred)
            search_space.append(int(astar_out[1]))
        return ged_prediction, search_space

    def do_astar_beam(self, sim_mat, sim_mat2, new_data):
        if len(sim_mat.shape) == 2:
            sim_mat = sim_mat.unsqueeze(0)
        if len(sim_mat2.shape) == 2:
            sim_mat2 = sim_mat2.unsqueeze(0)
        batch_num = sim_mat.shape[0]
        ged_prediction, search_space = [], []
        for b in range(batch_num):
            g1_id = new_data['g1'][b]['i'].item()
            g2_id = new_data['g2'][b]['i'].item()
            n1 = new_data['g1'][b].num_nodes
            n2 = new_data['g2'][b].num_nodes
            e1 = new_data['g1'][b].num_edges
            e2 = new_data['g2'][b].num_edges 
            beam_size = get_beam_size(n1, n2, e1/2, e2/2, self.args.dataset) 
            topk = min(self.args.topk, n1, n2) - 1 
            if topk <= 0: topk = 1
            matching_nodes, matching_order = self.find_topk_hun(sim_mat[b, :n1, :n2].detach(), sim_mat2[b, :n1, :n2].detach(),  topk, n1, n2)
            matching_order[0], matching_order[1] = 0, 0
            astar_out = self.app_astar.ged(CT(self.nx_graphs[g1_id]), CT(self.nx_graphs[g2_id]), int1ArrayToPointer(matching_order),
                                           int1ArrayToPointer(matching_order),int2ArrayToPointer(matching_nodes), CT(2 * topk), CT(beam_size))
            astar_out = astar_out.decode('ascii').split()
            pred = normalize_ged(n1, n2, int(astar_out[0]))
            ged_prediction.append(pred)
            search_space.append(int(astar_out[1]))
        return ged_prediction, search_space


    def find_topk_hun(self, sim_matrix, sim_matrix2, topk, n1=None, n2=None):
        if n1 is None and n2 is None:
            n1 = sim_matrix.shape[0]
            n2 = sim_matrix.shape[1]
        matching_nodes, matching_order = [], [n for n in range(n1)]
        mink = min(n2, topk)

        col_inds, col_inds2 = [], []
        for i in range(mink):
            row_ind, col_ind = linear_sum_assignment(cost_matrix=abs(sim_matrix[:, :]), maximize=True)
            sim_matrix[row_ind, col_ind] = 0
            col_inds.append(col_ind)

        for i in range(mink):
            row_ind2, col_ind2 = linear_sum_assignment(cost_matrix=abs(sim_matrix2[:, :]), maximize=True)
            sim_matrix2[row_ind2, col_ind2] = 0
            col_inds2.append(col_ind2)

        for i in range(len(row_ind)):
            t = []
            for j in range(len(col_inds)):
                t.append(col_inds[j][i])
                t.append(col_inds2[j][i])
            matching_nodes.append(t)

        return np.array(matching_nodes, dtype=int), np.array(matching_order, dtype=int)

