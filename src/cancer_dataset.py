
import os
import torch
import os.path as osp
from torch_geometric.data import (InMemoryDataset, Data)
import glob
import pickle
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

class cancerData(InMemoryDataset):
#    types = ['Pt', 'P', 'N', 'Cu', 'Mo', 'Pb', 'F', 'Se', 'As', 'Cl',
#            'Ru', 'Ni', 'Sn', 'C', 'Nd', 'Fe', 'Te', 'O', 'B', 'S', 'Co',
#            'Zr', 'Na', 'Zn', 'I', 'Er', 'Ti', 'Si', 'Br', 'K']

    types =  ['Ag', 'In', 'S', 'La', 'Pd', 'Br', 'Co', 'Fe', 'Ge', 'Zn',
              'W', 'O', 'Ni', 'P', 'Rh', 'B', 'Na', 'U', 'Mo', 'N', 'Ti',
              'F', 'Ir', 'Cl', 'Pt', 'As', 'Os', 'Sn', 'C', 'Au', 'Nd',
              'Ga', 'Bi', 'I', 'Ru', 'Zr', 'Si']

    def __init__(self, root, name, transform=None,  pre_transform=None, pre_filter=None):
        self.name = name
        assert self.name in ['CANCER']
        super(cancerData, self).__init__(root, transform, pre_transform,  pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        path = self.processed_paths[1]
        with open(path, 'rb') as f:
            self.pairs = pickle.load(f)

    @property
    def raw_file_names(self):
        return ['']

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.name), '{}_labels.pt'.format(self.name)]

    def download(self):
        pass

    def process(self):
        import networkx as nx

        ids, Ns = [], []
        r_path = self.raw_dir

        # Find the paths of all raw graphs:
        names = glob.glob(osp.join(r_path, '*.gexf'))
        ids.append(sorted([int(i.split(os.sep)[-1][:-5]) for i in names]))
        data_list = []
        # Convert graphs in .gexf format to a NetworkX Graph:
        for i, idx in enumerate(ids[-1]):
            i = i if len(ids) == 1 else i + len(ids[0])
            # Reading the raw `*.gexf` graph:
            G = nx.read_gexf(osp.join(r_path, f'{idx}.gexf'))
            mapping = {name: int(name) for name in G.nodes()}
            G = nx.relabel_nodes(G, mapping)

            Ns.append(G.number_of_nodes())
            edge_index = torch.tensor(list(G.edges)).t().contiguous()
            edge_index = to_undirected(edge_index, num_nodes=Ns[-1])
            data = Data(edge_index=edge_index, i=i)
            data.num_nodes = Ns[-1]

            # Create a one-hot encoded feature matrix denoting the atom
            # type (for the `AIDS700nef` dataset):
            if self.name == 'CANCER':
                x = torch.zeros(data.num_nodes, dtype=torch.long)
                for node, info in G.nodes(data=True):
                    x[int(node)] = self.types.index(info['type'])
                data.x = F.one_hot(x, num_classes=len(self.types)).to( torch.float)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

        # Extracting ground-truth GEDs from the GED pickle file
        label_path = osp.join(self.processed_dir,  '{}_labels.txt'.format(self.name))
        f = open(label_path, 'r')
        line = f.readline()
        data = []
        while line is not None and line != '':
            item = dict()
            line = line.split()
            item['rowid'] = int(line[0])
            item['colid'] = int(line[1])
            item['rgid'] = int(line[2])
            item['cgid'] = int(line[3])
            item['g1'] = int(line[4])
            item['g2'] = int(line[5])
            item['ged'] = int(line[6])
            #map = dict()  # key:g1 nodes, value: g2 nodes
            int_map = []
            for str in line:
                str_arr = str.split("|")
                if (len(str_arr)) > 1 and str_arr[0] != '-1':
                    int_map.append(int(str_arr[1]))
            item['int_map'] = int_map
            data.append(item)
            line = f.readline()
        f.close()

        with open(self.processed_paths[1], 'wb') as f:
            pickle.dump(data, f)


    def __repr__(self):
        return '{}({})'.format(self.name, len(self))
