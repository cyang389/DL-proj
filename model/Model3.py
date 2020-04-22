import torch
from torch_geometric.nn import SAGEConv, GCNConv, TopKPooling
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        embed_dim = 128

        self.conv1 = SAGEConv(embed_dim, 128)
        # self.pool1 = TopKPooling(128, ratio=0.8)
        self.lin1 = torch.nn.Linear(256, 1)
        self.item_embedding = torch.nn.Embedding(num_embeddings=20215, embedding_dim=embed_dim)

  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.item_embedding(x)
        x = x.squeeze(1)        

        x = F.relu(self.conv1(x, edge_index))

        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.lin1(x)).squeeze(1)

        return x