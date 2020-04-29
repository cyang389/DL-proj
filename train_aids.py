import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, SAGPooling, GCNConv, SAGEConv, TopKPooling
from torch_geometric.utils import train_test_split_edges
from AidsDataset import AidsDataset
from model.BaseModel import AidsModel
import numpy as np

def train():
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = F.binary_cross_entropy(output, label)
        loss.backward()
        total_loss += data.num_graphs * loss.item()
        optimizer.step()

    return float(total_loss / len(train_loader.dataset))

def test(loader):
    model.eval()
    
    correct = []
    predict = []
    for data in loader:
        data = data.to(device)
        label = data.y.detach().cpu().numpy().reshape(-1)
        pred = model(data).detach().cpu().numpy().reshape(-1)
        # print(pred.shape)
        correct.extend(label)
        predict.extend(pred)
    correct = np.array(correct)
    predict = np.array(predict)
    return roc_auc_score(np.array(correct), np.array(predict))
    

if __name__ == "__main__":

    # read data from Cora/CiteSeer/PubMed
    dataset = AidsDataset(root='./')

    batch_size = 32
    data = dataset[0]
    train_loader = DataLoader(dataset[:1500], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[1500:1800], batch_size=batch_size)
    test_loader = DataLoader(dataset[1800:], batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AidsModel(dataset.num_features).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    best_val_perf = test_perf = 0
    for epoch in range(1, 51):
        train_loss = train()
        train_acc = test(train_loader)
        val_acc = test(val_loader)
        test_acc = test(test_loader)
        log = 'Epoch: {:03d}, Loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_loss, train_acc, val_acc, test_acc))
