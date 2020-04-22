import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import pandas as pd

class AidsDataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(AidsDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['aids.dataset']

    def download(self):
        pass

    def process(self):
        node_attrs = pd.read_csv('data/Aids/AIDS.node_attrs', header=None).values
        node_labels = pd.read_csv('data/Aids/AIDS.node_labels', header=None).values
        link_labels = pd.read_csv('data/Aids/AIDS.link_labels', header=None).values
        edges = pd.read_csv('data/Aids/AIDS.edges', header=None).values
        graph_labels = pd.read_csv('data/Aids/AIDS.graph_labels', header=None).values
        graph_index = pd.read_csv('data/Aids/AIDS.graph_idx', header=None).values

        data_list = []
        graph_index -= 1
        edges -= 1
        node_1_list = edges[:, 0]
        for graph_idx in range(graph_index.max()):

            node_mask = graph_index == graph_idx
            node_mask = node_mask.reshape(-1)

            x = np.concatenate((node_attrs[node_mask], node_labels[node_mask][:, 1].reshape(-1, 1)), axis=1)
            x = torch.LongTensor(x)
            
            edge_mask = np.in1d(node_1_list, np.where(node_mask))

            edge_index = edges[edge_mask] - np.where(node_mask)[0][0]
            edge_index = torch.LongTensor(edge_index.T)

            edge_attr = link_labels[edge_mask]
            edge_attr = torch.LongTensor(edge_attr)

            y = np.zeros((np.where(node_mask)[0].shape[0], 1))
            y.fill(graph_labels[graph_idx][0])
            y = torch.FloatTensor(y)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

if __name__ == "__main__":
    node_attrs = pd.read_csv('data/Aids/AIDS.node_attrs', header=None).values
    print(node_attrs)