import numpy as np
import torch
import os.path as osp
import sys
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as transforms

def read_data(title):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', title)
    if title in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, title, transforms.NormalizeFeatures())
        data = dataset[0]
    return data, dataset