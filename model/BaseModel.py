import torch
from torch_geometric.nn import GCNConv, ChebConv  # noqa
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, data, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.data = data
        self.dataset = dataset

    def forward(self, pos_edge_index, neg_edge_index):

        x = F.relu(self.conv1(self.data.x, self.data.train_pos_edge_index))
        x = self.conv2(x, self.data.train_pos_edge_index)

        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)