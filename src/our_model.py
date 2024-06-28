import os.path
import pickle
import copy
import time
from tqdm import tqdm, trange
import torch.nn
from src.astar_utils import *
from torch_geometric.nn import GCNConv, GINConv, SplineConv
from src.layers import AttentionModule, NeuralTensorNetwork
from src.utils import *
from scipy.stats import spearmanr, kendalltau
from torch_geometric.utils import to_undirected, subgraph
from scipy.optimize import linear_sum_assignment
from src import global_var as glo_dict
import numpy.ctypeslib as npct

training_graph = "ged_map_train.txt"
testing_graph = "ged_map_test.txt"
val_graph = "ged_map_val.txt"
all_graphs_name = "AIDS_nx.pk"
all_graphs_name_txt = "AIDS_txt.txt"


EPS = 1e-8
PRINT_TIMING = False
upper_bound = 1000
topk = 9

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
        self.calculate_bottleneck_features()
        if self.args.gnn_operator == 'gcn':
            self.convolution_1 = GCNConv(self.num_labels, self.args.filter_1)
            self.convolution_2 = GCNConv(self.args.filter_1, self.args.filter_2)
            self.convolution_3 = GCNConv(self.args.filter_2, self.args.filter_3)
        elif self.args.gnn_operator == 'spline':
            self.convolution_1 = SplineConv(self.num_labels, self.args.filter_1)
            self.convolution_2 = SplineConv(self.args.filter_1, self.args.filter_2)
            self.convolution_3 = SplineConv(self.args.filter_2, self.args.filter_3)
        elif self.args.gnn_operator == 'gin':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(self.num_labels, self.args.filter_1),
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

        args1 = copy.deepcopy(self.args)
        args2 = copy.deepcopy(self.args)
        args1.filter_3 = self.args.filter_1
        args2.filter_3 = self.args.filter_2

        self.attention1 = AttentionModule(args1)
        self.attention2 = AttentionModule(args2)
        self.attention3 = AttentionModule(self.args)

        args1.tensor_neurons = args1.filter_3
        args2.tensor_neurons = args1.filter_2

        self.tensor_network1 = NeuralTensorNetwork(args1)
        self.tensor_network2 = NeuralTensorNetwork(args2)
        self.tensor_network3 = NeuralTensorNetwork(self.args)

        self.scoring_layer = torch.nn.Sequential(
            # torch.nn.Linear(self.feature_count  , 16),
            torch.nn.Linear(self.args.filter_1 + self.args.filter_2 + self.args.filter_3 , 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,1),
            torch.nn.Sigmoid()
        )

    def calculate_histogram(self, abstract_feature_1, abstract_feature_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_feature_1, abstract_feature_2).detach()
        scores = scores.view(-1,1)
        hist = torch.histc(scores, bins = self.args.bins)
        hist = hist/ torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def calculate_similarity(self, abstract_feature_1, abstract_feature_2):
        sim_matrix = torch.mm(abstract_feature_1, abstract_feature_2).detach()
        sim_matrix = torch.softmax(sim_matrix,dim=-1)
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



    def forward(self, all_graphs, data):
        """"
        Forward pass with graphs
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        device = next(self.parameters()).device
        edge_index_1 = all_graphs[data['g1']].edge_index.to(device)
        edge_index_2 = all_graphs[data['g2']].edge_index.to(device)
        feature_1 = all_graphs[data['g1']].x.to(device)
        feature_2 = all_graphs[data['g2']].x.to(device)

        # 对于每个图，abstract_feature_1: 节点个数x特征数
        g1_af_1, g1_af_2, g1_af_3  = self.convolutional_pass(edge_index_1, feature_1)  # the first conv abstract features of g1
        g2_af_1, g2_af_2, g2_af_3 = self.convolutional_pass(edge_index_2, feature_2)

        if self.args.histogram:
            hist = self.calculate_histogram(g1_af_3, torch.t(g1_af_3))
        sim_matrix_1,sim_matrix_2,sim_matrix_3 = [],[], []
        if self.args.matching_loss or self.args.triple_loss or self.args.nll_loss:
            sim_matrix_1 = self.calculate_similarity(g1_af_1, torch.t(g2_af_1))
            sim_matrix_2 = self.calculate_similarity(g1_af_2, torch.t(g2_af_2))
            sim_matrix_3 = self.calculate_similarity(g1_af_3, torch.t(g2_af_3))

        g1_pf_1 = self.attention1(g1_af_1)  # the first conv pool feature of g1
        g2_pf_1 = self.attention1(g2_af_1)
        g1_pf_2 = self.attention2(g1_af_2)
        g2_pf_2 = self.attention2(g2_af_2)
        g1_pf_3 = self.attention3(g1_af_3)
        g2_pf_3 = self.attention3(g2_af_3)

        scores_1 = self.tensor_network1(g1_pf_1, g2_pf_1)
        scores_2 = self.tensor_network2(g1_pf_2, g2_pf_2)
        scores_3 = self.tensor_network3(g1_pf_3, g2_pf_3)

        scores = torch.cat((scores_1, scores_2, scores_3), dim= 0)
        scores = torch.t(scores)
        if self.args.histogram:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)  # dim = 1 表示横着拼接
        scores = self.scoring_layer(scores)

        search_space = 0

        if self.args.astar:
            topk = int(sim_matrix_1.shape[1] * 0.75)
            matching_nodes, matching_order = self.find_topk_hun(sim_matrix_1.detach())
            matching_nodes2, matching_order2 = self.find_topk_hun(sim_matrix_2.detach())
            matching_nodes3, matching_order3 = self.find_topk_hun(sim_matrix_3.detach())
            t = np.concatenate((matching_nodes,matching_nodes2), axis=1)
            t = np.concatenate((t,matching_nodes3), axis=1)
            k = topk*3
            #k = min(topk, sim_matrix_1.shape[0],  sim_matrix_1.shape[1]) * 3

            matching_order[0],matching_order[1] = 0, 0
            res = self.app_astar.ged(CT(data['g1']), CT(data['g2']), int1ArrayToPointer(matching_order), int1ArrayToPointer(matching_order), int2ArrayToPointer(t), CT(k))
            scores = normalize_ged(sim_matrix_1.shape[0], sim_matrix_1.shape[1], res[0])
            search_space = res[1]
        return scores, sim_matrix_1, sim_matrix_2, sim_matrix_3, search_space



    def find_topk_hun(self, sim_matrix):
        topk = int(sim_matrix.shape[1] *0.75)
        matching_nodes, matching_order = [], [n for n in range(sim_matrix.shape[0])]
        mink = min(sim_matrix.shape[1], topk)

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

    def find_topk_match(self, sim_matrix):
        matching_order, matching_nodes = [], [n for n in range(sim_matrix.shape[0])]
        K = self.args.topk
        if sim_matrix.shape[0] <= K or sim_matrix.shape[1] <= K:
            sim_matrix_idx = sim_matrix.argsort(dim=1)
            return matching_order, sim_matrix_idx

        cols = set()    # 已经匹配的节点所占用 列的个数
        z = 0
        sim_matrix_idx = sim_matrix.argsort(dim=1, descending=True)
        for i in range(sim_matrix.shape[0]):
            i_nodes = sim_matrix_idx[i, 0:K].tolist()
            cols.update(i_nodes)
            j = K
            while len(cols) < K + z and j < sim_matrix_idx.shape[1]:
                i_nodes.append(sim_matrix_idx[i, j].item())
                cols.add(sim_matrix_idx[i, j].item())
                j += 1
            matching_nodes.append(i_nodes)
            if (len(cols) >= sim_matrix_idx.shape[1] ):
                cols.clear()
                z = 0
            z += 1

        return matching_order, matching_nodes

    def find_topk_match2(self, sim_matrix):
        matching_order, matching_nodes = [], [n for n in range(sim_matrix.shape[0])]
        K = self.args.topk
        if sim_matrix.shape[0] <= K or sim_matrix.shape[1] <= K:
            sim_matrix_idx = sim_matrix.argsort(dim=1)
            return matching_order, sim_matrix_idx

        cols = set()  # 已经匹配的节点所占用 列的个数
        z = 0
        sim_matrix_idx = sim_matrix.argsort(dim=1, descending=True)
        for i in range(sim_matrix.shape[0]):
            i_nodes = sim_matrix_idx[i, 0:K].tolist()
            cols.update(i_nodes)
            j = K
            while len(cols) < K + z and j < sim_matrix_idx.shape[1]:
                i_nodes.append(sim_matrix_idx[i, j].item())
                cols.add(sim_matrix_idx[i, j].item())
                j += 1
            matching_nodes.append(i_nodes)
            if (len(cols) >= sim_matrix_idx.shape[1]):
                cols.clear()
                z = 0
            z += 1

        return matching_order, matching_nodes

    @property
    def device(self):
        return next(self.parameters()).device



    def net_prediction(self, edge_index_1, edge_index_2, feature_1, feature_2):
        # 对于每个图，abstract_feature_1: 节点个数x特征数
        start = time.perf_counter()

        g1_af_1, g1_af_2, g1_af_3  = self.convolutional_pass(edge_index_1, feature_1)  # the first conv abstract features of g1
        g2_af_1, g2_af_2, g2_af_3 = self.convolutional_pass(edge_index_2, feature_2)

        g1_pf_1 = self.attention1(g1_af_1)  # the first conv pool feature of g1
        g2_pf_1 = self.attention1(g2_af_1)
        g1_pf_2 = self.attention2(g1_af_2)
        g2_pf_2 = self.attention2(g2_af_2)
        g1_pf_3 = self.attention3(g1_af_3)
        g2_pf_3 = self.attention3(g2_af_3)

        scores_1 = self.tensor_network1(g1_pf_1, g2_pf_1)
        scores_2 = self.tensor_network2(g1_pf_2, g2_pf_2)
        scores_3 = self.tensor_network3(g1_pf_3, g2_pf_3)

        scores = torch.cat((scores_1, scores_2, scores_3), dim= 0)
        scores = torch.t(scores)
        scores = self.scoring_layer(scores)
        # 返回

        g1_nodes = feature_1.shape[0]
        g2_nodes = feature_2.shape[0]
        nged = -math.log(scores, math.e)

        tmp = glo_dict.get_value('net_prediction')
        glo_dict.set_value('net_prediction', tmp + time.perf_counter() - start)
        return int(nged * (g1_nodes + g2_nodes) / 2)


class OurNNTrainer(object):
    def __init__(self, args):
        self.args = args
        self.all_graphs = self.load_all_graphs(self.args.graph_path, all_graphs_name)
        self.training_graphs = self.load_data(self.args.graph_path, training_graph)
        self.testing_graphs = self.load_data(self.args.graph_path, testing_graph)
        self.val_graphs = self.load_data(self.args.graph_path, val_graph)
        if self.args.test:
            self.init_astar()

        if self.args.debug:
            self.training_graphs = self.training_graphs[0: int (0.005* len(self.training_graphs))]
            self.testing_graphs = self.testing_graphs[0: int(0.005* len(self.testing_graphs))]
            self.val_graphs = self.val_graphs[0: int (0.005* len(self.val_graphs))]

        self.setup_model()

    def init_astar(self):
        app_astar = ctypes.cdll.LoadLibrary(r'.\Astar\AstarGED.so')            # app_astar: approximate astar
        app_astar.init.restype = ctypes.c_void_p
        app_astar.ged.restype = PINT
        app_astar.init( CT(os.path.join(self.args.graph_path, all_graphs_name_txt) ))
        self.app_astar = app_astar


    def load_all_graphs(self, path, all_graphs_name ):
        if os.path.exists("all_graphs.pk") is True:  # 判断ged_map 的文件是否已经处理好。
            with open("all_graphs.pk", 'rb') as f:
                all_graphs = pickle.load(f)
        else:
            module_path = os.path.dirname(__file__)
            with open(os.path.join(module_path, path, all_graphs_name), 'rb') as f:
                all_graphs = pickle.load(f)
                del all_graphs['types']
                # aids_types = all_graphs.pop('types')
                i = 0
                for k, v in all_graphs.items():
                    i += 1
                    if (i % 10000 == 0): print( "load: {}".format(i))
                    myg = myGraph()
                    myg.G = v
                    edge_index = torch.tensor(list(v.edges)).t().contiguous()
                    if edge_index.numel() == 0:
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_index = to_undirected(edge_index, num_nodes=v.number_of_nodes())

                    x = torch.zeros(v.number_of_nodes(), dtype=torch.long)
                    for node, info in v.nodes(data=True):
                        x[int(node)] = types.index(info['type'])        # types 在utils中定义了
                    x = F.one_hot(x, num_classes=len( types)).to(torch.float)

                    myg.num_nodes = v.number_of_nodes()
                    myg.edge_index = edge_index
                    myg.x = x
                    all_graphs[k] = myg

            with open("all_graphs.pk", 'wb') as f:
                pickle.dump(all_graphs, f)
        return all_graphs

    def load_data(self, r_path, graph_type):
        # 首先会 根据所有的graphs更新当前的graphs， 返回的data 存放 pair list的信息。
        data = []
        module_path = os.path.dirname(__file__)
        with open(os.path.join(module_path, r_path, graph_type), 'r') as f:
                line = f.readline()
                while line is not None and line != '':
                    item = dict()
                    line = line.split()
                    item['g1'] = line[0]
                    item['g2'] = line[1]
                    item['ged'] = int(line[2])
                    item['norm_ged'] = normalize_ged(self.all_graphs[line[0]].num_nodes, self.all_graphs[line[1]].num_nodes, item['ged'])
                    map = dict()  # key:g1 nodes, value: g2 nodes
                    for str in line:
                        str_arr = str.split("|")
                        if (len(str_arr)) > 1 and str_arr[0] != -1:
                            map[str_arr[0]] = str_arr[1]
                    item['map'] = map
                    data.append(item)
                    line = f.readline()  # 读取下一行
        return data


    def setup_model(self):
        self.num_labels = len(types)
        self.model = OurNN(self.args, self.num_labels, self.app_astar)


    def create_batches(self):
        """
        Create batches from the training graph pairs
        :return batches: List of graph pairs
        """
        # random.shuffle(self.training_graphs)
        batches = []
        for graph in range (0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph: graph + self.args.batch_size])
        return batches

    def process_batches(self, batch):
        """
        Forward pass with a batch of data
        :param batch: Batch of graph pairs
        :return loss: Loss on the Batch
        """
        self.optimizer.zero_grad()
        losses = 0

        for graph_pair in batch:
            target = graph_pair['norm_ged'].to(self.model.device)

            prediction, sim_matrix_1, sim_matrix_2, sim_matrix_3, search_space = self.model(self.all_graphs, graph_pair)
            indicator_loss, matching_loss, triple_loss, nll = 0, 0, 0, 0
            if prediction > target: indicator_loss = prediction - target
            if self.args.matching_loss:
                y = [[], []]
                for k, v in graph_pair['map'].items():
                    if k != '-1':
                        y[0].append(int(k))
                        y[1].append(int(v))
                y_val_1 = torch.sum(sim_matrix_1[y[0], y[1]])
                rand_1 = random_assign(sim_matrix_1.shape[0])
                rand_val_1 = torch.sum(sim_matrix_1[rand_1[0], rand_1[1]])
                if y_val_1 < rand_val_1 : matching_loss = F.mse_loss(y_val_1, rand_val_1)

                y_val_2 = torch.sum(sim_matrix_2[y[0], y[1]])
                rand_2 = random_assign(sim_matrix_2.shape[0])
                rand_val_2 = torch.sum(sim_matrix_2[rand_2[0], rand_2[1]])
                if y_val_2 < rand_val_2: matching_loss += F.mse_loss(y_val_2, rand_val_2)

                y_val_3 = torch.sum(sim_matrix_3[y[0], y[1]])
                rand_3 = random_assign(sim_matrix_3.shape[0])
                rand_val_3 = torch.sum(sim_matrix_3[rand_3[0], rand_3[1]])
                if y_val_3 < rand_val_3: matching_loss += F.mse_loss(y_val_3, rand_val_3)

            if self.args.triple_loss:
                y = [[], []]
                for k, v in graph_pair['map'].items():
                    if k != '-1':
                        y[0].append(int(k))
                        y[1].append(int(v))
                y_val = torch.sum(sim_matrix_3[y[0], y[1]])
                max_rand_val = 0
                for i in range (5):
                    rand = random_assign(sim_matrix_3.shape[0])
                    rand_val = torch.sum(sim_matrix_3[rand[0], rand[1]])
                    if rand_val > max_rand_val: max_rand_val = rand_val
                triple_loss = max(max_rand_val - y_val + self.args.margin, 0)
                triple_loss = triple_loss * np.power(10.0, self.args.scale)

            if self.args.nll_loss:
                y = [[], []]
                for k, v in graph_pair['map'].items():
                    if k != '-1':
                        y[0].append(int(k))
                        y[1].append(int(v))
                # y_val = torch.sum(sim_matrix[y[0], y[1]])
                y_val = sim_matrix_3[y[0], y[1]]
                nll = -torch.sum(torch.log(y_val +EPS))
                nll = nll * np.power(10.0, self.args.scale)

            losses = losses + F.mse_loss(target, prediction) + matching_loss + triple_loss + nll + indicator_loss
            if self.args.nll_loss and self.args.scale == 0:
                losses = losses + nll

        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):
        print('\nmodel training \n')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.learning_rate, weight_decay = self.args.weight_decay)
        self.model.train()

        epochs = trange(self.args.epochs, ascii=True, leave=True, desc="Epoch", position=0)
        best_mse = float('inf')

        for epoch in epochs:
            batches = self.create_batches()
            loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batches(batch)
                main_index = main_index + len(batch)
                loss_sum = loss_sum + loss_score
                loss = loss_sum / main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

            if epoch > 0:
                cur_scores = self.score(test=False)
                if best_mse > cur_scores:
                    torch.save(self.model.state_dict(), 'best_model_{}_{}_e{}_lr{}.pt'.format(self.args.dataset, self.args.gnn_operator, self.args.epochs, self.args.learning_rate))
                    best_mse = cur_scores

    def score_before(self, test=True):
        """
        Evaluate on the test data
        :param test: if test is True, use the testing set; else use the validation set
        """
        print("\n\nModel evaluation.\n")
        if test:
            testing_graphs = self.testing_graphs
        else:
            testing_graphs = self.val_graphs
        self.model.eval()
        self.scores = []

        for graph_pair in tqdm(testing_graphs):
            target = graph_pair['norm_ged'].item()
            prediction, sim_matrix = self.model(self.all_graphs, graph_pair)
            prediction = prediction.view(-1).detach().item()
            self.scores.append( (prediction-target)* (prediction-target))
        print("\nMSE error: " + str(round( np.mean(self.scores), 5)) + ".")
        return np.mean(self.scores)


    def score(self, test=True):
        """
        Evaluate on the test data
        :param test: if test is True, use the testing set; else use the validation set
        """
        print("\n\nModel evaluation.\n")
        if test:
            testing_graphs = self.testing_graphs
        else:
            testing_graphs = self.val_graphs
        self.model.eval()
        l1_list, l2_list, rho_list, tau_list, prec_at_10_list, prec_at_20_list = [], [], [], [], [], []
        pred_list, target_list = [], []
        acc, fea = 0, 0

        graph_num = len(testing_graphs)
        all_space = 0

        for i, graph_pair in enumerate(tqdm(testing_graphs)):
            target = graph_pair['norm_ged'].item()
            if len( graph_pair['map']) <= 1000:
                # print("current index:", i)

                prediction, s1, s2, s3, search_space = self.model(self.all_graphs, graph_pair)
                all_space += search_space
                prediction = prediction.view(-1).detach().item()
                pred_list.append(prediction)
                target_list.append(target)

                l1_list.append(abs(prediction-target))
                l2_list.append((prediction - target)**2)
                n1 = self.all_graphs[graph_pair['g1']].num_nodes
                n2 = self.all_graphs[graph_pair['g2']].num_nodes
                if ( round( denormalize_ged(n1, n2, prediction) ) == round( denormalize_ged(n1, n2, target)) ):
                    acc += 1
                if ( round( denormalize_ged(n1, n2, prediction) ) >= round( denormalize_ged(n1, n2, target)) ):
                    fea += 1
                if ( (i+1) % 400 == 0 ):
                    pred_list = np.array(pred_list, dtype=np.float32)
                    target_list = np.array(target_list, dtype=np.float32)
                    rho_list.append(calculate_ranking_correlation(spearmanr, pred_list, target_list))
                    tau_list.append(calculate_ranking_correlation(kendalltau, pred_list, target_list))
                    prec_at_10_list.append(calculate_prec_at_k(10, pred_list, target_list))
                    prec_at_20_list.append(calculate_prec_at_k(20, pred_list, target_list))
                    pred_list,target_list = [], []

        if ( len(pred_list) != 0):
            pred_list = np.array(pred_list, dtype=np.float32)
            target_list = np.array(target_list, dtype=np.float32)
            rho_list.append(calculate_ranking_correlation(spearmanr, pred_list, target_list))
            tau_list.append(calculate_ranking_correlation(kendalltau, pred_list, target_list))
            prec_at_10_list.append(calculate_prec_at_k(10, pred_list, target_list))
            prec_at_20_list.append(calculate_prec_at_k(20, pred_list, target_list))
            pred_list, target_list = [], []

        self.scores = np.mean(l2_list)
        print("mae: " + str(round(np.mean(l1_list), 5)) + ".")
        print("mse: " + str(round(np.mean(l2_list), 5)) + ".")
        print("fea: " + str(round(fea*1.0/len(testing_graphs), 5)) + ".")
        print("acc: " + str(round(acc*1.0/len(testing_graphs), 5)) + ".")
        print("Spearman's rho: " + str(round(np.nanmean(rho_list), 5)) + ".")
        print("Kendall's tau: " + str(round(np.nanmean(tau_list), 5)) + ".")
        print("p@10: " + str(round(np.mean(prec_at_10_list), 5)) + ".")
        print("p@20: " + str(round(np.mean(prec_at_20_list), 5)) + ".")
        print("search space:" + str(all_space/graph_num))


        glo_dict.printvar()
        return self.scores

