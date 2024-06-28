import glob
import os.path
import copy
from tqdm import tqdm, trange
import torch.nn
import time
import pickle
from torch_geometric.nn import GCNConv, GINConv, SplineConv
from src.layers import AttentionModule, NeuralTensorNetwork, Affinity,  Sinkhorn, NorSim
from scipy.stats import spearmanr, kendalltau
from torch_geometric.data import Batch
import torch_geometric.transforms as T
from src.myloss_func import *
from src.randomWalk import AddRandomWalkPE
from src.mydegree import MyDegree
from src.myConstant import MyConstanct

EPS = 1e-8
PRINT_TIMING = False
upper_bound = 1000
SK_ITER_NUM = 10
SK_EPSILON = 1.0e-10
SK_TAU = 0.005
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
        self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.args.filter_1, self.args.filter_1)        #  self.num_labels
            self.convolution_2 = GCNConv(self.args.filter_1, self.args.filter_2)
            self.convolution_3 = GCNConv(self.args.filter_2, self.args.filter_3)
        elif self.args.gnn_operator == 'spline':
            self.convolution_1 = SplineConv(self.args.filter_1, self.args.filter_1)    # #  self.num_labels
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
        self.attention = AttentionModule(self.args)
        # self.tensor_network = NeuralTensorNetwork(self.args)
        self.affinity = Affinity(self.args.filter_3)
        self.sinkhorn = Sinkhorn(max_iter=SK_ITER_NUM, epsilon=SK_EPSILON, tau=SK_TAU)
        self.nor_sim = NorSim()
        self.scoring_layer = torch.nn.Sequential(
            torch.nn.Linear(2 * self.args.filter_3, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,1),
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
        features = F.dropout(features, p = self.args.dropout, training=self.training)

        features_2 = self.convolution_2(features, edge_index)
        features = F.relu(features_2)
        features = F.dropout(features, p = self.args.dropout, training=self.training)

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
        g1_af_1, g1_af_2, g1_af_3 = self.convolutional_pass(edge_index_1, feature_1)  # the first conv abstract features of g1
        g2_af_1, g2_af_2, g2_af_3 = self.convolutional_pass(edge_index_2, feature_2)

        g1_graph_level_3 = self.attention(g1_af_3, batch_1)
        g2_graph_level_3 = self.attention(g2_af_3, batch_2)
        scores = torch.cat((g1_graph_level_3, g2_graph_level_3), dim=1)

        ged_score = self.scoring_layer(scores).view(-1)

        sim_mat = self.affinity(g1_af_3, g2_af_3, batch_1, batch_2)
        rows = torch.bincount(g1.batch).to(device)
        cols = torch.bincount(g2.batch).to(device)
        if self.args.sinkhorn:
            sim_mat = self.sinkhorn(sim_mat, nrows=rows, ncols=cols)
        else:
            sim_mat = self.nor_sim(sim_mat, rows, cols)
        return ged_score, sim_mat, sim_mat, sim_mat

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

    def load_node_matching(self):
        pairwise = 'datasets/{}/processed/{}'.format(self.args.dataset, 'pairwise_map.pk')
        with open(pairwise, 'rb') as f:
            self.node_matching = pickle.load(f)
        # convert to int.
        for i in range(len(self.node_matching)):
            node_map = self.node_matching[i]["map"]
            int_map = []
            for k, v in node_map.items():
                if k != '-1': int_map.append(int(v))
            self.node_matching[i]["int_map"] = int_map

    def process_dataset(self):
        """
        Downloading and processing dataset.
        """
        print("\nPreparing dataset.\n")

        if self.args.dataset in ['AIDS700nef', 'LINUX', 'ALKANE', 'IMDBMulti']: # torch_geometric datasets
            from torch_geometric.datasets import GEDDataset
            ori_train = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=True)
            self.training_graphs = ori_train[:len(ori_train) // 4 * 3]          # // 除 向下取整。 3: 1 : 1
            self.val_graphs = ori_train[len(ori_train) // 4 * 3:]
            self.testing_graphs = GEDDataset('datasets/{}'.format(self.args.dataset), self.args.dataset, train=False)
            self.graph_size = len(self.training_graphs) + len(self.val_graphs) + len(self.testing_graphs)
            nx_graphs_train = self.load_nx_graph('datasets/{}/raw/{}/train'.format(self.args.dataset, self.args.dataset))
            nx_graphs_test = self.load_nx_graph('datasets/{}/raw/{}/test'.format(self.args.dataset, self.args.dataset))
            self.nx_graphs = nx_graphs_train + nx_graphs_test
        else:
            raise ValueError('Unknown dataset name {}'.format(self.args.dataset))

        if self.args.debug: #
            self.training_graphs = self.training_graphs[0:200]
            self.val_graphs = self.val_graphs[0:20]
            self.testing_graphs = self.testing_graphs[0:20]
            self.args.epochs = 300
            self.args.val_epochs = 220


        self.nged_matrix = self.training_graphs.norm_ged
        self.nged_matrix = torch.exp(-self.nged_matrix)

        self.ged = self.training_graphs.ged
        self.real_data_size = self.nged_matrix.size(0)
        self.load_node_matching()       # load node matching

        if self.args.nonstruc:
            const = MyConstanct(1.0, cat=self.training_graphs[0].x is not None)
            self.training_graphs.transform = const
            self.val_graphs.transform = const
            self.testing_graphs.transform = const
            self.num_labels = self.training_graphs.num_features

        else:
            const = MyConstanct(1.0, cat=self.training_graphs[0].x is not None)
            max_degree = self.args.max_degree
            myDeg = MyDegree(max_degree)
            randwalk = AddRandomWalkPE(self.args.random_walk_step)
            transform = T.Compose([const, myDeg, randwalk])
            self.training_graphs.transform = transform
            self.val_graphs.transform = transform
            self.testing_graphs.transform = transform
            self.num_labels = self.training_graphs.num_features + self.args.max_degree + self.args.random_walk_step


    def init_astar(self):
        so_path = os.path.join(module_path, 'Astar', 'mata.so')
        app_astar = ctypes.cdll.LoadLibrary(so_path)  # app_astar: approximate astar
        app_astar.ged.restype = ctypes.c_char_p
        app_astar.mapping_ed.restype = ctypes.c_int
        self.app_astar = app_astar

#####################这个与cancer文件中的读取load_nx_graph是不同的，这里需要对节点连续化#####################
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
            mapping = {name: j for j, name in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            G_str = nx2txt(G, str(idx), self.args.dataset)
            data_list.append(G_str)
        return data_list

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: Zipped loaders as list.
        """
        if hasattr(self.training_graphs, 'pair_list'):
            select_pair_idx = torch.randint(self.training_graphs.pair_list.shape[0], (self.args.batch_size,))
            select_pair_list = self.training_graphs.pair_list[select_pair_idx]
            data_list_1 = list(self.training_graphs[select_pair_list[:, 0]])
            data_list_2 = list(self.training_graphs[select_pair_list[:, 1]])
        else:
            data_list_1 = list(self.training_graphs.shuffle()[:self.args.batch_size])  # 先对graph pairs shuffle，再从最开始的地方选择batch size的大小
            data_list_2 = list(self.training_graphs.shuffle()[:self.args.batch_size])
            # data_list_1 = list(self.training_graphs[0:self.args.batch_size])  # 先对graph pairs shuffle，再从最开始的地方选择batch size的大小
            # data_list_2 = list(self.training_graphs[1:1+self.args.batch_size])

        # 这里构造 g1 g2的时候，需要满足： |g1| <= |g2| 如果g1=g2，那么g1的id要小于g2
        for i in range(self.args.batch_size):
            if data_list_1[i].x.shape[0] > data_list_2[i].x.shape[0] or (data_list_1[i].x.shape[0] == data_list_2[i].x.shape[0] and data_list_2[i]['i'] > data_list_1[i]['i'] ):
                tmp = data_list_1[i]
                data_list_1[i] = data_list_2[i]
                data_list_2[i] = tmp

        source_batch = Batch.from_data_list(data_list_1)
        target_batch = Batch.from_data_list(data_list_2)

        # 构造GED 和 node_matching.
        max_nodes_row = max(torch.bincount(source_batch.batch)).item()
        max_nodes_col = max(torch.bincount(target_batch.batch)).item()          # 一个batch中最大的节点个数，和forward输出的大小一致。便于计算Loss. target_batch是较大的节点集

        nor_ged = self.nged_matrix[source_batch["i"].reshape(-1).tolist(), target_batch["i"].reshape(-1).tolist()].tolist()
        nor_ged = torch.from_numpy(np.array(nor_ged)).view(-1).float().to(self.model.device)
        list_map = []
        for idx in range(source_batch.num_graphs):
            item = self.node_matching[self.get_index(source_batch[idx]['i'], target_batch[idx]['i'])]
            if source_batch[idx]['i'].item() != int(item['g1']) or target_batch[idx]['i'].item() != int(item['g2']):
                raise ValueError('error of the index of node matching')
            else:
                map_idx = torch.zeros((max_nodes_row, max_nodes_col), dtype=int)
                row = np.arange(0, len(item["int_map"]), dtype=int)
                col = np.array(item["int_map"], dtype=int)
                map_idx[[row, col]] = 1
                list_map.append(map_idx)
        batch_map = torch.stack(list_map, dim=0).to(self.model.device)

        # # perform transform
        new_data = dict()
        new_data["g1"] = source_batch.to(self.model.device)  # 一共有 batch size的源图
        new_data["g2"] = target_batch.to(self.model.device)  # 一共有 batch size的目标图
        new_data["map"] = batch_map
        new_data["nor_ged"] = nor_ged
        return new_data

    def get_index(self, g1_id, g2_id):              # 传入这里的参数: |g1| <= |g2| 如果g1=g2，那么g1的id要小于g2
        train_index = int(self.graph_size*0.8)      # train val test, 6:2:2
        if (g1_id < train_index and g2_id < train_index):
            return train_index * g1_id + g2_id
        else:
            return train_index * train_index + (g1_id - train_index) * train_index + g2_id


    def process_batches(self, batch):
        self.optimizer.zero_grad()
        losses = torch.tensor(0.0, requires_grad=True)
        ged_score, simmat1, simmat2, simmat3 = self.model(batch)
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

        losses = losses + F.mse_loss(ged_score, batch["nor_ged"], reduction="mean")
        losses = losses + criterion(simmat3, batch["map"], rows, cols)
        losses.backward(retain_graph=True)
        self.optimizer.step()
        return losses.item()

    def fit(self):
        print('\nmodel training \n')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.model.train()
        epochs = trange(self.args.epochs, ascii=True, leave=True, desc="Epoch", position=0)
        best_sim_score = float(0.0)           # best_sim_score 
        for epoch in epochs:
            batches = self.create_batches()
            loss_score = self.process_batches(batches)
            loss = loss_score
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            if epoch > 0 and epoch % (self.args.val_epochs) == 0:
                cur_sim_score = self.score(test=False)     # GED尽可能小，对应的similarity score尽可能大
                if cur_sim_score > best_sim_score:
                    print("update the model. The average similarity score of epoch({}):{}".format(epoch, cur_sim_score))
                    torch.save(self.model.state_dict(), 'best_model_{}_{}_e{}_lr{}_loss{}_{}_{}_{}.pt'.format(self.args.dataset, self.args.gnn_operator, self.args.epochs, self.args.learning_rate, self.args.loss_type, self.args.filter_1, self.args.filter_2, self.args.filter_3))
                    best_sim_score = cur_sim_score


    def score(self, test=True):
        print("\n\nModel evaluation.\n")
        if test:
            testing_graphs = self.testing_graphs
            if self.args.dataset in ['IMDBMulti']: self.training_graphs = self.training_graphs[0: len(self.training_graphs) // 4]
        else:
            testing_graphs = self.val_graphs            # 300 for imdb, 140 for aids
            testing_graphs = testing_graphs[0:20]



        self.model.eval()
        scores = np.zeros((len(testing_graphs), len(self.training_graphs)))
        ground_truth = np.full((len(testing_graphs), len(self.training_graphs)), 1e-10)
        prediction_mat = np.full((len(testing_graphs), len(self.training_graphs)), 1e-10)
        rho_list, tau_list, prec_at_10_list, prec_at_20_list = [], [], [], []
        n1s = np.empty((len(testing_graphs), len(self.training_graphs)), dtype=int)    # test graph size x training graph size
        n2s = np.empty((len(testing_graphs), len(self.training_graphs)), dtype=int)
        t = tqdm(total=len(testing_graphs) * len(self.training_graphs), ascii=True, position=0)
        acc_num, fea_num, superior_num, cur_idx, all_space,all_time = 0, 0, 0, 0, 0, 0
        l1_list, l2_list, ret_score, all_mata = [], [], [], []

        for i, g in enumerate(testing_graphs):
            data_list_1 = list([g] * len(self.training_graphs))
            data_list_2 = list(self.training_graphs)
            for j in range(len(self.training_graphs)):  #
                g1 = data_list_1[j]
                g2 = data_list_2[j]
                if g1.x.shape[0] > g2.x.shape[0] or (g1.x.shape[0] == g2.x.shape[0] and g2['i'] > g1['i']):
                    data_list_1[j] = g2
                    data_list_2[j] = g1

            for start_idx in range(0, len(self.training_graphs), self.args.batch_size):  # 从0到len(self.training_graphs)间隔 batch size
                _data_list_1 = data_list_1[start_idx:start_idx + self.args.batch_size]
                _data_list_2 = data_list_2[start_idx:start_idx + self.args.batch_size]
                source_batch = Batch.from_data_list(_data_list_1)  # 将一批图看做是一个大图的对象。
                target_batch = Batch.from_data_list(_data_list_2)
                nor_ged = self.nged_matrix[source_batch["i"].reshape(-1).tolist(), target_batch["i"].reshape(-1).tolist()].tolist()
                nor_ged = torch.from_numpy(np.array(nor_ged)).view(-1).float().to(self.model.device)
                n1s[i, start_idx:start_idx + self.args.batch_size] = torch.bincount(source_batch.batch).detach().numpy()
                n2s[i, start_idx:start_idx + self.args.batch_size] = torch.bincount(target_batch.batch).detach().numpy()
                cur_idx = cur_idx + len(_data_list_1)

                new_data = dict()
                new_data["g1"] = source_batch.to(self.model.device)  # 一共有 batch size的源图
                new_data["g2"] = target_batch.to(self.model.device)  # 一共有 batch size的目标图
                new_data["nor_ged"] = nor_ged
                target = new_data["nor_ged"].cpu()
                ground_truth[i, start_idx:start_idx + self.args.batch_size] = target
                start_time = time.process_time()
                pred_ged, simmat1, simmat2, simmat3 = self.model(new_data)
                astar_prediction, search_space = self.do_astar(simmat3.detach().cpu(), new_data)
                all_time = all_time + time.process_time() - start_time

                all_space = all_space + sum(search_space[:])
                astar_pred_tensor = torch.from_numpy(np.array(astar_prediction))
                acc_num = acc_num + len(target) - torch.count_nonzero(target - astar_pred_tensor, dim=-1).item()
                sup = (target + 1e-3 - astar_pred_tensor < 0) == True
                superior_num = superior_num + torch.sum( sup).item()
                astar_pred_tensor = astar_pred_tensor * (~sup)
                target = target * (~sup)

                prediction_mat[i, start_idx:start_idx + len(_data_list_1)] = np.array(astar_prediction)
                ground_truth[i, start_idx:start_idx + len(_data_list_1)] = target
                scores[i, start_idx:start_idx + len(_data_list_1)] = F.mse_loss(astar_pred_tensor, target, reduction='none').detach().numpy()
                l1_list.append(np.average(F.l1_loss(astar_pred_tensor, target, reduction='none').detach().numpy()))
                l2_list.append(np.average(F.mse_loss(astar_pred_tensor, target, reduction='none').detach().numpy()))

                t.update(len(_data_list_1))
                t.set_description("AvgGED={:.3f}".format( np.mean(prediction_mat[i] )))

            rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
            tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
            prec_at_10_list.append(calculate_prec_at_k(10, prediction_mat[i], ground_truth[i]))
            prec_at_20_list.append(calculate_prec_at_k(20, prediction_mat[i], ground_truth[i]))
            ret_score.append(np.mean(prediction_mat[i]))
            all_mata.append(prediction_mat[i])

        print("mae: " + str(round(np.mean(l1_list), 5)))
        print("mse: " + str(round(np.mean(l2_list), 5)))
        print("fea: " + str(round(1.0)))
        print("acc: " + str(round(acc_num * 1.0 / cur_idx, 5)))
        print("Spearman's rho: " + str(round(np.nanmean(rho_list), 5)))
        print("Kendall's tau: " + str(round(np.nanmean(tau_list), 5)))
        print("p@10: " + str(round(np.mean(prec_at_10_list), 5)))
        print("p@20: " + str(round(np.mean(prec_at_20_list), 5)))
        print("search space:" + str(all_space / cur_idx))
        print("average time:" + str(all_time / cur_idx))
        print("superior_num:" + str(round(superior_num/cur_idx, 5) ))


        if test:    # 完成了所有test集的计算，把计算的结果写到文件中
            new_label_items = []
            wf = open(os.path.join('datasets/{}/processed/'.format(self.args.dataset), "fea_results_mata_b{}_k{}.txt".format(self.args.batch_size, self.args.topk)), 'w')
            for i, g1 in enumerate(testing_graphs):
                for j, g2 in enumerate(self.training_graphs):  #
                    line = "{} {} {}\n".format(g1['i'].item(), g2['i'].item(), all_mata[i][j] )
                    item = dict()
                    item['rowid'] = g1['i'].item()
                    item['colid'] = g2['i'].item()
                    item['mata'] = all_mata[i][j]
                    new_label_items.append(item)
                    wf.write(line)
            wf.close()
            with open(os.path.join('datasets/{}/processed/'.format(self.args.dataset), "fea_results_mata_b{}_k_{}.pt".format(self.args.batch_size, self.args.topk)), 'wb') as f:
                pickle.dump(new_label_items, f)
        return np.mean(np.array(ret_score))

    def do_astar(self, sim_mat, new_data):
        if len(sim_mat.shape) == 2:
            sim_mat = sim_mat.unsqueeze(0)
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
            topk = min(self.args.topk, n1, n2)
            if topk == 0: topk = 1
            matching_nodes, matching_order = self.find_topk_hun(sim_mat[b, :n1, :n2].detach(), topk, n1, n2)
            matching_order[0], matching_order[1] = 0, 0
            astar_out = self.app_astar.ged(CT(self.nx_graphs[g1_id]), CT(self.nx_graphs[g2_id]), int1ArrayToPointer(matching_order),
                                           int1ArrayToPointer(matching_order),int2ArrayToPointer(matching_nodes), CT(topk), CT(beam_size))
            astar_out = astar_out.decode('ascii').split()
            pred = normalize_ged(n1, n2, int(astar_out[0]))
            ged_prediction.append(pred)
            search_space.append(int(astar_out[1]))
        return ged_prediction, search_space

    def find_topk_hun(self, sim_matrix, topk, n1 = None, n2 = None):
        if n1 is None and n2 is None:
            n1 = sim_matrix.shape[0]
            n2 = sim_matrix.shape[1]
        matching_nodes, matching_order = [], [n for n in range( n1 )]
        mink = min(n2, topk)

        col_inds = []
        for i in range(mink):
            row_ind, col_ind = linear_sum_assignment(-abs(sim_matrix[:, :]))
            sim_matrix[row_ind, col_ind] = 0
            col_inds.append(col_ind)

        for i in range(len(row_ind)):
            t = []
            for j in range(len(col_inds)):
                t.append(col_inds[j][i])
            matching_nodes.append(t)

        return np.array(matching_nodes, dtype=int), np.array(matching_order, dtype=int)


"""
测试不同的loss
nohup python main.py --loss-type 1  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef > our_loss1.log 2>&1 &
nohup python main.py --loss-type 2  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef > our_loss2.log 2>&1 &
nohup python main.py --loss-type 3  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef > our_loss3.log 2>&1 &
测试不同的 filter 
nohup python main.py --loss-type 1  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef --filter-1 64 --filter-2 64 --filter-3 64 > our_loss1_64_64_64.log 2>&1 &
nohup python main.py --loss-type 1  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef --filter-1 64 --filter-2 32 --filter-3 32 > our_loss1_64_32_32.log 2>&1 &
nohup python main.py --loss-type 1  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef --filter-1 64 --filter-2 32 --filter-3 64 > our_loss1_64_32_64.log 2>&1 &

测试不同的 GNN
nohup python main.py --loss-type 1  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef  --gnn-operator gin > our_loss1_gin.log 2>&1 &
 
在参数，GCN, 64,64,64，加入，ged 预测的分数，

不同的loss表现，
nohup python main.py --loss-type 1  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef > our_loss1.log 2>&1 &
nohup python main.py --loss-type 4  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef > our_loss4.log 2>&1 &
nohup python main.py --loss-type 5  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef > our_loss5.log 2>&1 &
nohup python main.py --loss-type 6  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef > our_loss6.log 2>&1 &
不同topk的表现，
nohup python main.py --topk 0.2 --loss-type 1  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef > our_loss1_topk2.log 2>&1 &
nohup python main.py --topk 0.3 --loss-type 1  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef > our_loss1_topk3.log 2>&1 &
nohup python main.py --topk 0.4 --loss-type 1  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef > our_loss1_topk4.log 2>&1 &
nohup python main.py --topk 0.6 --loss-type 1  --astar --epochs 20000 --learning-rate 0.001 --dataset AIDS700nef > our_loss1_topk6.log 2>&1 &
不同learning rate 的表现，
nohup python main.py --topk 0.4 --loss-type 6  --astar --epochs 20000 --learning-rate 0.0001 --dataset AIDS700nef > our_loss6_topk4_lr.1.log 2>&1 &
nohup python main.py --topk 0.4 --loss-type 6  --astar --epochs 20000 --learning-rate 0.0002 --dataset AIDS700nef > our_loss6_topk4_lr.2.log 2>&1 &
nohup python main.py --topk 0.4 --loss-type 6  --astar --epochs 20000 --learning-rate 0.0005 --dataset AIDS700nef > our_loss6_topk4_lr.5.log 2>&1 &
nohup python main.py --topk 0.4 --loss-type 1  --astar --epochs 20000 --learning-rate 0.0002 --dataset AIDS700nef > our_loss1_topk4_lr.2.log 2>&1 &

structure enhanced GNN.
nohup python main.py --loss-type 1  --astar --epochs 22000 --learning-rate 0.001 --dataset AIDS700nef > our_loss1_structure.log 2>&1 &
nohup python main.py --loss-type 4  --astar --epochs 22000 --learning-rate 0.001 --dataset AIDS700nef > our_loss4_structure.log 2>&1 &
nohup python main.py --loss-type 5  --astar --epochs 22000 --learning-rate 0.001 --dataset AIDS700nef > our_loss5_structure.log 2>&1 &
nohup python main.py --loss-type 6  --astar --epochs 22000 --learning-rate 0.001 --dataset AIDS700nef > our_loss6_structure.log 2>&1 &

nohup python main.py --loss-type 1 --topk 0.4  --astar --epochs 22000 --learning-rate 0.001 --dataset AIDS700nef > our_loss1_topk4_structure.log 2>&1 &

nohup python main.py  --loss-type 6  --astar --epochs 19000 --learning-rate 0.0002 --dataset AIDS700nef > our_loss6_lr.2.log 2>&1 &
nohup python main.py  --loss-type 6  --astar --epochs 19000 --learning-rate 0.0005 --dataset AIDS700nef > our_loss6_lr.5.log 2>&1 &


--cuda 
nohup python main.py --cuda    --astar --epochs 19000 --learning-rate 0.0006 --dataset AIDS700nef > cuda_ls1_lr.6.log 2>&1 &
nohup python main.py --cuda    --astar --epochs 19000 --learning-rate 0.0006 --dataset AIDS700nef > cuda_ls2_lr.6.log 2>&1 &
nohup python main.py --cuda    --astar --epochs 19000 --learning-rate 0.0006 --dataset AIDS700nef > cuda_ls3_lr.6.log 2>&1 &

nohup python main.py --cuda   --astar --epochs 19000 --learning-rate 0.0008 --dataset AIDS700nef > cuda_ls2_lr.8.log 2>&1 &
nohup python main.py --cuda  --loss-type 2  --astar --epochs 19000 --learning-rate 0.001 --dataset AIDS700nef > cuda_ls2_lr1.log 2>&1 &

-- final version 
注意 我这里没有用 --cuda  
nohup python main.py  --loss-type 2 --sinkhorn --astar --epochs 32000 --learning-rate 0.001 --dataset AIDS700nef > aids_default.log 2>&1 &
nohup python main.py --loss-type 2 --sinkhorn --astar --epochs 30000 --learning-rate 0.001 --dataset IMDBMulti --max-degree 40  >  imdb_default.log 2>&1 &

nohup python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12  > aids_9.15.log 2>&1 &
nohup python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  > imdb_9.17.log 2>&1 &
nohup python main.py  --epochs 16000 --dataset CANCER --max-degree 18  > cancer_9.15.log 2>&1 &


需要重新训练imdb data
nohup python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --topk 6 --batch-size 128 > imdb_9.20_k-6-b128.log 2>&1 &
nohup python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --topk 6 --batch-size 64 > imdb_9.20_k-6-b64.log 2>&1 &

# 重新测试cancer
nohup python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 6  > cancer_9.20-k6.log 2>&1 &
nohup python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 7  > cancer_9.20-k7.log 2>&1 &
nohup python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 8  > cancer_9.20-k8.log 2>&1 &
nohup python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 10  > cancer_9.20-k10.log 2>&1 &

# 测试 aids topk选择问题
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 5 > aids_test_k_5.log 2>&1 &   
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 6 > aids_test_k_6.log 2>&1 &   
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 7 > aids_test_k_7.log 2>&1 &   
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 8 > aids_test_k_8.log 2>&1 &   
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 9 > aids_test_k_9.log 2>&1 &   
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 10 > aids_test_k_10.log 2>&1 &   

# 测试 imdb topk选择问题
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 5 > imdb_test_k_5.log 2>&1 &
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 6 > imdb_test_k_6.log 2>&1 &
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 7 > imdb_test_k_7.log 2>&1 &
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 8 > imdb_test_k_8.log 2>&1 &
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 9 > imdb_test_k_9.log 2>&1 &
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 10 > imdb_test_k_10.log 2>&1 &


# 测试 cancer topk选择问题
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 5  > cancer_test_k_5.log 2>&1 &
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 6  > cancer_test_k_6.log 2>&1 &
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 7  > cancer_test_k_7.log 2>&1 &
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 8  > cancer_test_k_8.log 2>&1 &
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 9  > cancer_test_k_9.log 2>&1 &
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 10  > cancer_test_k_10.log 2>&1 &
  
  
# 测试有无 struc
nohup python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --nonstruc --topk 6 > aids_nonstruc-k6.log 2>&1 &
nohup python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40 --nonstruc --topk 6 > imdb_nonstruc-k6.log 2>&1 &
nohup python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --nonstruc --topk 7 > cancer_nonstruc-k7.log 2>&1 &

"""