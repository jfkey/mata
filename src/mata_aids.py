import glob
import os.path
import copy

import torch_geometric.nn
from tqdm import tqdm, trange
import torch.nn
from torch_scatter import scatter
import time
import pickle
from torch_geometric.nn import GCNConv, GINConv, SplineConv, GraphConv
from src.layers import AttentionModule, NeuralTensorNetwork, Affinity,  Sinkhorn, NorSim, soft_topk, greedy_perm
from scipy.stats import spearmanr, kendalltau
from torch_geometric.data import Batch
import torch_geometric.transforms as T
import random

from src.myloss_func import *
from src.randomWalk import AddRandomWalkPE
from src.mydegree import MyDegree
from src.myConstant import MyConstanct

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
        self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        if self.args.gnn_operator == 'gc':
            self.convolution_1 = GraphConv(self.args.filter_1, self.args.filter_1)
            self.convolution_1 = GraphConv(self.args.filter_1, self.args.filter_2)
            self.convolution_1 = GraphConv(self.args.filter_2, self.args.filter_3)

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

        # self.bi_conv_1 = GraphConv(in_channels=(self.args.filter_3, self.args.filter_3), out_channels=self.args.filter_3, aggr="mean", bias=True)
        # self.bi_conv_2 = GraphConv(in_channels=(self.args.filter_3, self.args.filter_3), out_channels=self.args.filter_3, aggr="mean", bias=True)

        self.bi_conv_1 = GraphConv(self.args.filter_3, self.args.filter_3)
        self.bi_conv_2 = GraphConv(self.args.filter_3, self.args.filter_3*2)

        self.init_layer = torch.nn.Sequential(
            torch.nn.Linear(self.num_labels, self.args.filter_1),
            torch.nn.ReLU()
        )

        self.degree_emb = torch.nn.Parameter(torch.Tensor(self.args.max_degree, self.args.max_degree))
        torch.nn.init.xavier_uniform_(self.degree_emb)
        self.attention = AttentionModule(self.args.filter_3)
        self.attention2 = AttentionModule(self.args.filter_3*2)

        # self.tensor_network = NeuralTensorNetwork(self.args)
        self.affinity = Affinity(self.args.filter_3)
        self.sinkhorn = Sinkhorn(max_iter=SK_ITER_NUM, epsilon=SK_EPSILON, tau=SK_TAU)
        self.nor_sim = NorSim()
        self.scoring_layer = torch.nn.Sequential(
            torch.nn.Linear(8 * self.args.filter_3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,1),
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

    def bi_conv_pass(self, edge_index, features_in, features_out, edge_weight=None):
        edge_weight = None
        # edge_index_reverse = torch.stack((edge_index[1], edge_index[0]),dim=0)
        features =  torch.cat((features_in, features_out), dim=0)
        features = self.bi_conv_1(features, edge_index, edge_weight)
        features = F.relu(features)
        features = F.dropout(features, p = self.args.dropout, training=self.training)
        features = self.bi_conv_2(features, edge_index, edge_weight)
        return features



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


        rows = torch.bincount(g1.batch).to(device)
        cols = torch.bincount(g2.batch).to(device)
        sim_mat1 = self.affinity(feature_1, feature_2, batch_1, batch_2)
        gt_ks = torch.full([sim_mat1.shape[0]], self.args.topk / 2, device=device, dtype=torch.float)
        _, sim_mat1 = soft_topk(sim_mat1, gt_ks.view(-1), SK_ITER_NUM, SK_TAU, nrows=rows, ncols=cols, return_prob=True)

        sim_mat2 = self.affinity(g1_af_3, g2_af_3, batch_1, batch_2)
        _, sim_mat2 = soft_topk(sim_mat2, gt_ks.view(-1), SK_ITER_NUM, SK_TAU, nrows=rows, ncols=cols, return_prob=True)
 

        # if self.args.sinkhorn:
        #     sim_mat = self.sinkhorn(sim_mat, nrows=rows, ncols=cols)
        # else:
        #     sim_mat = self.nor_sim(sim_mat, rows, cols)

        # gt_ks = torch.full([sim_mat.shape[0]], self.args.topk, device=device, dtype=torch.float)
        # _, sim_mat = soft_topk(sim_mat, gt_ks.view(-1), SK_ITER_NUM, SK_TAU, nrows=rows, ncols=cols, return_prob=True)

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


        '''
        if len(sim_mat.shape) == 2:
            sim_mat = sim_mat.unsqueeze(0)
        batch_num = sim_mat.shape[0]
        v1_v2_list, v2_v1_list = [], []
        for i in range(batch_num):
            before_r, before_c = 0, 0
            v1_to_v2 = torch.mm(torch.t(sim_mat[i, 0:rows[i], 0:cols[i]]), g1_af_3[before_r : before_r+rows[i]])
            before_r += rows[i]
            v1_v2_list.append(v1_to_v2)

            v2_to_v1 = torch.mm(sim_mat[i, 0:rows[i], 0:cols[i]], g2_af_3[before_c : before_c+cols[i]])
            v2_v1_list.append(v2_to_v1)
            before_c += cols[i]

        v1_v2_list = torch.cat(v1_v2_list, dim=0)
        v2_v1_list = torch.cat(v2_v1_list, dim=0)
        g1_graph_level = self.attention(v1_v2_list, batch_2)
        g2_graph_level = self.attention(v2_v1_list, batch_1)
        scores = torch.cat((g1_graph_level, g2_graph_level), dim=1)
        ged_score = self.scoring_layer(scores).view(-1)
        '''


        '''
        # for i in range(torch.sum(rows * cols)):
        bi_edge_weight = torch.zeros( 2 * torch.sum(rows * cols), dtype=torch.float)
        v1_list, v2_list = list(), list()
        bi_batch_index, v1_idx, v2_idx = 0, 0, torch.sum(rows).item()
        bi_batch = torch.zeros(torch.sum(rows + cols), dtype=torch.long)
        for i in range(len(rows)):
            before_list_size = len(v1_list)
            for i_0 in range(rows[i]):
                for j_0 in range(cols[i]):
                    v1_list.append(i_0 + v1_idx)
                    v2_list.append(j_0 + v2_idx)
                    v1_list.append(j_0 + v2_idx)        # bi-connect-graph
                    v2_list.append(i_0 + v1_idx)

            bi_batch[bi_batch_index:bi_batch_index+rows[i]+cols[i]] = i
            bi_batch_index += (rows[i]+cols[i])
            bi_edge_weight[before_list_size : before_list_size + rows[i].item() * cols[i].item() ] = torch.flatten(sim_mat[i, 0:rows[i].item(), 0:cols[i].item()])
            bi_edge_weight[before_list_size + rows[i].item() * cols[i].item(): before_list_size + 2*rows[i].item() * cols[i].item()] = torch.flatten( sim_mat[i, 0:rows[i].item(), 0:cols[i].item()])
            v1_idx += rows[i].item()
            v2_idx += cols[i].item()

        bi_edge_index = torch.from_numpy(np.array([v1_list, v2_list], dtype=np.long)).long()
        bi_features = self.bi_conv_pass(bi_edge_index, g1_af_3, g2_af_3, bi_edge_weight)

        scores = self.attention2(bi_features, bi_batch)
        #g1_graph_level = self.attention(g1_af_3, batch_1)
        #g2_graph_level = self.attention(g2_af_3, batch_2)
        #scores = torch.cat((g1_graph_level, g2_graph_level), dim=1)
        #ged_score = self.scoring_layer(scores).view(-1)
        ged_score = self.scoring_layer(scores).view(-1)
        '''

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

            graph_list = []
            for i in range(len(self.training_graphs)):
                gi = const(self.training_graphs[i])
                graph_list.append(gi)
            self.training_graphs.data = Batch.from_data_list(graph_list)
            graph_list = []

            for i in range(len(self.val_graphs)):
                gi = const(self.val_graphs[i])
                graph_list.append(gi)
            self.val_graphs.data = Batch.from_data_list(graph_list)

            graph_list = []
            for i in range(len(self.testing_graphs)):
                gi = const(self.testing_graphs[i])
                graph_list.append(gi)
            self.testing_graphs.data = Batch.from_data_list(graph_list)
            self.num_labels = self.training_graphs.data.num_features

        else:
            const = MyConstanct(1.0, cat=self.training_graphs[0].x is not None)
            max_degree = self.args.max_degree
            myDeg = MyDegree(max_degree)
            randwalk = AddRandomWalkPE(self.args.random_walk_step)
            transform = T.Compose([const, myDeg, randwalk])

            graph_list = []
            for i in range(len(self.training_graphs)):
                gi = transform(self.training_graphs[i])
                graph_list.append(gi)
            self.training_graphs.data = Batch.from_data_list(graph_list)
            graph_list = []

            for i in range(len(self.val_graphs)):
                gi = transform(self.val_graphs[i])
                graph_list.append(gi)
            self.val_graphs.data = Batch.from_data_list(graph_list)

            graph_list = []
            for i in range(len(self.testing_graphs)):
                gi = transform(self.testing_graphs[i])
                graph_list.append(gi)
            self.testing_graphs.data = Batch.from_data_list(graph_list)
            self.num_labels = self.training_graphs.data.num_features + self.args.max_degree + self.args.random_walk_step


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
            # data_list_1 = list(self.training_graphs.shuffle()[:self.args.batch_size])  # 先对graph pairs shuffle，再从最开始的地方选择batch size的大小
            # data_list_2 = list(self.training_graphs.shuffle()[:self.args.batch_size])
            # data_list_1 = list(self.training_graphs[0:self.args.batch_size])  # 先对graph pairs shuffle，再从最开始的地方选择batch size的大小
            # data_list_2 = list(self.training_graphs[1:1+self.args.batch_size])
            data_list = Batch.to_data_list(self.training_graphs.data)
            random.shuffle(data_list)
            data_list_1 = data_list[:self.args.batch_size]
            random.shuffle(data_list)
            data_list_2 = data_list[:self.args.batch_size]


        # 这里构造 g1 g2的时候，需要满足： |g1| <= |g2| 如果g1=g2，那么g1的id要小于g2
        for i in range(self.args.batch_size):
            if data_list_1[i].x.shape[0] > data_list_2[i].x.shape[0] or (data_list_1[i].x.shape[0] == data_list_2[i].x.shape[0] and data_list_2[i]['i'] > data_list_1[i]['i'] ):
                tmp = data_list_1[i]
                data_list_1[i] = data_list_2[i]
                data_list_2[i] = tmp

        source_batch = Batch.from_data_list(data_list_1)
        target_batch = Batch.from_data_list(data_list_2) ##########################################
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
                    torch.save(self.model.state_dict(), 'best_model_{}_{}_e{}_lr{}_loss{}_t{}_stru{}_b{}.pt'.format(self.args.dataset, self.args.gnn_operator, self.args.epochs, self.args.learning_rate, self.args.loss_type,  self.args.tasks, self.args.nonstruc, self.args.batch_size ))
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
        training_graphs = Batch.to_data_list(self.training_graphs.data)
        train_size = len(training_graphs)
        num_train_graphs = train_size * train_size 
        if test:
            testing_graphs = self.testing_graphs.data
            testing_graphs = Batch.to_data_list(testing_graphs)
            num_test_graphs = len(testing_graphs) * train_size
            if self.args.dataset in ['IMDBMulti']:
                training_graphs = training_graphs[0: train_size // 8]
                train_size = len(training_graphs) 
        else:
            testing_graphs = self.val_graphs.data            # 300 for imdb, 140 for aids
            testing_graphs = Batch.to_data_list(testing_graphs)
            testing_graphs = testing_graphs[0:20]
        test_size = len(testing_graphs)
  
        self.model.eval()
        scores = np.zeros((test_size, train_size))
        ground_truth = np.full((test_size, train_size), 1e-10)
        prediction_mat = np.full((test_size, train_size), 1e-10)
        rho_list, tau_list, prec_at_10_list, prec_at_20_list = [], [], [], []
        n1s = np.empty((test_size, train_size), dtype=int)    # test graph size x training graph size
        n2s = np.empty((test_size, train_size), dtype=int)
        t = tqdm(total=test_size * train_size, ascii=True, position=0)
        acc_num, fea_num, superior_num, cur_idx, all_space,all_time = 0, 0, 0, 0, 0, 0
        l1_list, l2_list, ret_score, all_mata = [], [], [], []

        for i, g in enumerate(testing_graphs):
            data_list_1 = list([g] * train_size)
            data_list_2 = list(training_graphs)
            for j in range(train_size):  #
                g1 = data_list_1[j]
                g2 = data_list_2[j]
                if g1.x.shape[0] > g2.x.shape[0] or (g1.x.shape[0] == g2.x.shape[0] and g2['i'] > g1['i']):
                    data_list_1[j] = g2
                    data_list_2[j] = g1

            for start_idx in range(0, train_size, self.args.batch_size):  # 从0到len(self.training_graphs)间隔 batch size
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
            for i, g1 in enumerate(testing_graphs):
                for j, g2 in enumerate(training_graphs):  #
                    line = "{} {} {}\n".format(g1['i'].item(), g2['i'].item(), all_mata[i][j] )
                    item = dict()
                    item['rowid'] = g1['i'].item()
                    item['colid'] = g2['i'].item()
                    item['mata'] = all_mata[i][j]
                    new_label_items.append(item)
                    wf.write(line)
            wf.close()
            with open(os.path.join('datasets/{}/processed/'.format(self.args.dataset), "fea_results_mata_b{}_k_{}_{}_t{}.pt".format(self.args.batch_size, self.args.topk, self.args.nonstruc, self.args.tasks)), 'wb') as f:
                pickle.dump(new_label_items, f)
        return np.mean(np.array(ret_score))

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
            beam_size = get_beam_size(n1, n2, e1/2, e2/2, self.args.dataset)
            topk = min(self.args.topk, n1, n2)
            if topk == 0: topk = 1
            matching_nodes, matching_order = self.find_topk_hun(sim_mat[b, :n1, :n2].detach(), sim_mat2[b, :n1, :n2].detach(),  topk, n1, n2)
            matching_order[0], matching_order[1] = 0, 0
            astar_out = self.app_astar.ged(CT(self.nx_graphs[g1_id]), CT(self.nx_graphs[g2_id]), int1ArrayToPointer(matching_order),
                                           int1ArrayToPointer(matching_order),int2ArrayToPointer(matching_nodes), CT(2 * topk), CT(beam_size))
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

    def find_topk_hun(self, sim_matrix, sim_matrix2, topk, n1 = None, n2 = None):
        if n1 is None and n2 is None:
            n1 = sim_matrix.shape[0]
            n2 = sim_matrix.shape[1]
        matching_nodes, matching_order = [], [n for n in range( n1 )]
        mink = min(n2, topk)

        col_inds,col_inds2 = [], []
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


"""
# iclr results
nohup python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --topk 6  > aids_9.15.log 2>&1 & 
nohup python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --topk 6 > imdb_9.17.log 2>&1 &
nohup python main.py  --epochs 16000 --dataset CANCER --max-degree 18  --topk 8 > cancer_9.15.log 2>&1 &

# topk on aids
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 5 > aids_test_k_5.log 2>&1 &   
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 6 > aids_test_k_6.log 2>&1 &   
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 7 > aids_test_k_7.log 2>&1 &   
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 8 > aids_test_k_8.log 2>&1 &   
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 9 > aids_test_k_9.log 2>&1 &   
python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --test --topk 10 > aids_test_k_10.log 2>&1 &   

# topk on imdb 
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 5 > imdb_test_k_5.log 2>&1 &
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 6 > imdb_test_k_6.log 2>&1 &
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 7 > imdb_test_k_7.log 2>&1 &
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 8 > imdb_test_k_8.log 2>&1 &
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 9 > imdb_test_k_9.log 2>&1 &
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40  --test --topk 10 > imdb_test_k_10.log 2>&1 &


#  topk on cancer 
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 5  > cancer_test_k_5.log 2>&1 &
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 6  > cancer_test_k_6.log 2>&1 &
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 7  > cancer_test_k_7.log 2>&1 &
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 8  > cancer_test_k_8.log 2>&1 &
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 9  > cancer_test_k_9.log 2>&1 &
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --test --topk 10  > cancer_test_k_10.log 2>&1 &
  
  
# our GEgcn vs gcn 
nohup python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --nonstruc --topk 6 > aids_nonstruc-k6.log 2>&1 &
nohup python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40 --nonstruc --topk 6 > imdb_nonstruc-k6.log 2>&1 &
nohup python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --nonstruc --topk 7 > cancer_nonstruc-k7.log 2>&1 &


python main.py  --epochs 30000 --dataset AIDS700nef --max-degree 12 --nonstruc --topk 6 --test 
python main.py  --epochs 30000 --dataset IMDBMulti --max-degree 40 --nonstruc --topk 6 --test
python main.py  --epochs 16000 --dataset CANCER --max-degree 18 --nonstruc --topk 8 --test
################################################

nohup python main.py --val-epochs 3000 --epochs 30000 --dataset AIDS700nef --max-degree 12  --topk 6 --batch-size 128 --tasks 1 >   aids_tasks1-k6.log 2>&1 &
nohup python main.py --val-epochs 3000 --epochs 30000 --dataset AIDS700nef --max-degree 12  --topk 6 --batch-size 128 --tasks 2  >  aids_tasks2-k6.log 2>&1 &
nohup python main.py --val-epochs 3000 --epochs 30000 --dataset AIDS700nef --max-degree 12  --topk 6 --batch-size 128 --tasks 3  >  aids_tasks3-k6.log 2>&1 &
nohup python main.py --val-epochs 3000 --epochs 30000 --dataset AIDS700nef --max-degree 12  --topk 6 --batch-size 128 --tasks 4  >  aids_tasks_iclr-k6.log 2>&1 &

nohup python main.py --val-epochs 3000 --epochs 24000 --dataset AIDS700nef --max-degree 12  --topk 6 --batch-size 128 --tasks 1  >  aids_tasks_cross_attention-k6.log 2>&1 &
nohup python main.py --val-epochs 3000 --epochs 24000 --dataset AIDS700nef --max-degree 12  --topk 6 --batch-size 128 --tasks 3  >  aids_tasks_cross_attention_3-k6.log 2>&1 &
 


##########################################################################################################################################################
########################################cvpr final results aids################################################################################
nohup python main.py --val-epochs 3000 --epochs 30000 --dataset AIDS700nef --max-degree 12  --topk 6 --batch-size 128 --tasks 1  >  aids_cross_nosiy_1-k6.log 2>&1 &
nohup python main.py --val-epochs 3000 --epochs 30000 --dataset AIDS700nef --max-degree 12  --topk 6 --batch-size 128 --tasks 2  >  aids_cross_nosiy_2-k6.log 2>&1 &
nohup python main.py --val-epochs 3000 --epochs 30000 --dataset AIDS700nef --max-degree 12  --topk 6 --batch-size 128 --tasks 3  >  aids_cross_nosiy_3-k6.log 2>&1 &
nohup python main.py --val-epochs 3000 --epochs 30000 --dataset AIDS700nef --max-degree 12 --nonstruc --topk 6 --batch-size 128 --tasks 3 > aids_nonstruc-k6.log 2>&1 &
---batch size 
 nohup python main.py --val-epochs 4000 --epochs 50000 --dataset AIDS700nef --max-degree 12  --topk 6 --batch-size 64 --tasks 3  >  aids_batch64-k6.log 2>&1 &
 nohup python main.py --val-epochs 5000 --epochs 60000 --dataset AIDS700nef --max-degree 12  --topk 6 --batch-size 32 --tasks 3  >  aids_batch32-k6.log 2>&1 &

cvpr final results imdb
nohup python main.py --val-epochs 3000 --epochs 30000 --dataset IMDBMulti --max-degree 40  --topk 6 --batch-size 128 --tasks 1 > imdb_cross_nosiy_1-k6.log 2>&1 &
nohup python main.py --val-epochs 3000 --epochs 30000 --dataset IMDBMulti --max-degree 40  --topk 6 --batch-size 128 --tasks 2 > imdb_cross_nosiy_2-k6.log 2>&1 &
nohup python main.py --val-epochs 3000 --epochs 30000 --dataset IMDBMulti --max-degree 40  --topk 6 --batch-size 128 --tasks 3 > imdb_cross_nosiy_3-k6.log 2>&1 &
nohup python main.py --val-epochs 3000 --epochs 30000 --dataset IMDBMulti --max-degree 40 --nonstruc --topk 6 --batch-size 128 --tasks 3 > imdb_nonstruc-k6.log 2>&1 &
  
---batch size 
nohup python main.py --val-epochs 4000 --epochs 50000 --dataset IMDBMulti --max-degree 40  --topk 6 --batch-size 64 --tasks 3 > imdb_batch64-k6.log 2>&1 &
nohup python main.py --val-epochs 5000 --epochs 60000 --dataset IMDBMulti --max-degree 40  --topk 6 --batch-size 32 --tasks 3 > imdb_batch32-k6.log 2>&1 &
 
cvpr final results cancer 
nohup python main.py --val-epochs 2000 --epochs 16000 --dataset CANCER --max-degree 18  --topk 8 --batch-size 128  --tasks 1 > cancer_cross_nosiy_1-k8.log 2>&1 &
nohup python main.py --val-epochs 2000 --epochs 16000 --dataset CANCER --max-degree 18  --topk 8 --batch-size 128  --tasks 2 > cancer_cross_nosiy_2-k8.log 2>&1 &
nohup python main.py --val-epochs 2000 --epochs 16000 --dataset CANCER --max-degree 18  --topk 8 --batch-size 128  --tasks 3 > cancer_cross_nosiy_3-k8.log 2>&1 &
nohup python main.py --val-epochs 2000 --epochs 16000 --dataset CANCER --max-degree 18  --nonstruc --topk 8 --batch-size 128  --tasks 3 > cancer_nonstruc-k8.log 2>&1 &

---batch size 
nohup python main.py --val-epochs 3000 --epochs 25000 --dataset CANCER --max-degree 18  --topk 8 --batch-size 64 --tasks 3 > cancer_batch64-k8.log 2>&1 &
nohup python main.py --val-epochs 4000 --epochs 30000 --dataset CANCER --max-degree 18  --topk 8 --batch-size 32 --tasks 3 > cancer_batch32-k8.log 2>&1 &
 
nostruc
nohup python main.py --test --epochs 30000 --dataset AIDS700nef --max-degree 12 --nonstruc --topk 6 > aids_nonstruc-k6.log 2>&1 &
nohup python main.py --test --epochs 30000 --dataset IMDBMulti --max-degree 40 --nonstruc --topk 6 > imdb_nonstruc-k6.log 2>&1 &
nohup python main.py --test --epochs 16000 --dataset CANCER --max-degree 18 --nonstruc --topk 8 > cancer_nonstruc-k8.log 2>&1 &

nohup python main.py --test --epochs 30000 --dataset AIDS700nef --max-degree 12 --task 1  --topk 6 > aids_t1-k6.log 2>&1 &
nohup python main.py --test --epochs 30000 --dataset IMDBMulti --max-degree 40 --task 1 --topk 6 > imdb_t1-k6.log 2>&1 &
nohup python main.py --test --epochs 16000 --dataset CANCER --max-degree 18 --task 1 --topk 8 > cancer_t1-k8.log 2>&1 &

nohup python main.py --test --epochs 30000 --dataset AIDS700nef --max-degree 12 --task 2  --topk 6 > aids_t2-k6.log 2>&1 &
nohup python main.py --test --epochs 30000 --dataset IMDBMulti --max-degree 40 --task 2 --topk 6 > imdb_t2-k6.log 2>&1 &
nohup python main.py --test --epochs 16000 --dataset CANCER --max-degree 18 --task 2 --topk 8 > cancer_t2-k8.log 2>&1 &

nohup python main.py --test  --epochs 30000 --dataset AIDS700nef --max-degree 12   --topk 1 > aids_top1-k6.log 2>&1 &
nohup python main.py --test --epochs 30000 --dataset IMDBMulti --max-degree 40  --topk 1 > imdb_top1-k6.log 2>&1 &
nohup python main.py --test --epochs 16000 --dataset CANCER --max-degree 18  --topk 1 > cancer_top1-k8.log 2>&1 &

CIKM AIDS
python main.py  --dataset AIDS700nef --max-degree 12  --topk 4 --batch-size 128 --epochs 10000  --val-epochs 1000  > aids.log 2>&1 &
python main.py  --dataset AIDS700nef --max-degree 12  --topk 4 --batch-size 128 --epochs 10000  --val-epochs 1000 --nonstruc  > aids_nonstruc.log 2>&1 &
python main.py  --dataset AIDS700nef --max-degree 12  --topk 4 --batch-size 128 --epochs 10000  --val-epochs 1000 --task 1  > aids_t1.log 2>&1 &
python main.py  --dataset AIDS700nef --max-degree 12  --topk 4 --batch-size 128 --epochs 10000  --val-epochs 1000 --task 2  > aids_t2.log 2>&1 &

CIKM IMDB
python main.py  --dataset IMDBMulti --max-degree 40  --topk 4 --batch-size 128 --epochs 10000  --val-epochs 1000  > imdb.log 2>&1 &
python main.py  --dataset IMDBMulti --max-degree 40  --topk 4 --batch-size 128 --epochs 10000  --val-epochs 1000 --nonstruc  > imdb_nonstruc.log 2>&1 &
python main.py  --dataset IMDBMulti --max-degree 40  --topk 4 --batch-size 128 --epochs 10000  --val-epochs 1000 --task 1  > imdb_t1.log 2>&1 &
python main.py  --dataset IMDBMulti --max-degree 40  --topk 4 --batch-size 128 --epochs 10000  --val-epochs 1000 --task 2  > imdb_t2.log 2>&1 &

CIKM CANCER
python main.py  --dataset CANCER --max-degree 18 --topk 4 --batch-size 128 --epochs 6000  --val-epochs 800 > cancer.log 2>&1 &
python main.py  --dataset CANCER --max-degree 18 --topk 4 --batch-size 128 --epochs 6000  --val-epochs 800 --nonstruc > cancer_nonstruc.log 2>&1 &
python main.py  --dataset CANCER --max-degree 18 --topk 4 --batch-size 128 --epochs 6000  --val-epochs 800 --task 1 > cancer_t1.log 2>&1 &
python main.py  --dataset CANCER --max-degree 18 --topk 4 --batch-size 128 --epochs 6000  --val-epochs 800 --task 2 > cancer_t2.log 2>&1 &

 
"""
