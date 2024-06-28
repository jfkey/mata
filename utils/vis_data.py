import networkx as nx
import pickle
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

all_graph_path = "D:/workspace/GED/ourGED/src/dataset/AIDS/AIDS_nx.pk"
save_path = "D:/workspace/GED/ourGED/src/dataset/AIDS/"

#31592, 630612
#id = "122231"
id= "630612"
#id= "622153"
# id = "635120"
types = [
 'C', 'S', 'Cl', 'N', 'O'
]
types_id = {v:(i+1) for i, v in enumerate(types)}


def draw_gexf():
    with open(os.path.join(all_graph_path), 'rb') as f:
        all_graphs = pickle.load(f)
        G = all_graphs[id]

        fig, ax = plt.subplots()
        labs =nx.get_node_attributes(G, "type" )
        color_lookup = dict()
        for k in G.nodes():
            if labs[k] in types_id:
                color_lookup[k] = types_id[labs[k]]
            else:
                color_lookup[k] =len(types)

        low, *_, high = sorted(color_lookup.values())
        norm = mpl.colors.Normalize(vmin=1, vmax=len(types), clip=True)
        mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Spectral)
        for k in labs:
            labs[k] = labs[k] + "," + str(k)
        nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels=True, labels=labs, node_color = [mapper.to_rgba(i) for i in color_lookup.values()], node_size=150 )

        ax.axis("off")
        plt.title( "id:" + str(id),loc ='center')
        fig.show()
        fig.savefig(save_path + str(id) +  ".png", format='png', dpi=600)

if __name__ == '__main__':
    draw_gexf()

