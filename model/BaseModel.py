import torch
from torch_geometric.nn import GINConv, SAGPooling, GCNConv, SAGEConv, TopKPooling   # noqa
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class Net(torch.nn.Module):
    def __init__(self, data, dataset):
        super(Net, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(dataset.num_features, 128),
            nn.Linear(128, 128)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, 64)
        )
        # self.conv1 = GCNConv(dataset.num_features, 128)
        # self.conv2 = GCNConv(128, 64)
        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)
        self.data = data
        self.dataset = dataset

    def forward(self, pos_edge_index, neg_edge_index):

        # x = F.relu(self.conv1(self.data.x, self.data.train_pos_edge_index))
        # x = self.conv2(x, self.data.train_pos_edge_index)
        x = self.conv1(self.data.x, self.data.train_pos_edge_index)
        x = self.conv2(x, self.data.train_pos_edge_index)

        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)

class AidsModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(AidsModel, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.conv1 = GINConv(self.mlp1)
        # self.pool1 = SAGPooling(64, min_score=1e-3, GNN=GCNConv)
        self.conv2 = GINConv(self.mlp2)
        # self.pool2 = SAGPooling(64, min_score=1e-3, GNN=GCNConv)
        self.lin1 = nn.Linear(64, 16)
        self.lin2 = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(x.shape)

        x = F.relu(self.conv1(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.sig(x)
        return x
