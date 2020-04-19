import numpy as np
import torch

class ProcessData():
    def __init__(self, title):
        self.title = title
        
    def process(self):
        with open("data/" + self.title + ".nodes") as f:
            nodes = f.read().splitlines()
        
        with open("data/" + self.title + ".edges") as f:
            edges = f.read().splitlines()

        # data.x, Node feature matrix with shape [num_nodes, num_node_features]
        x = []
        for line in nodes[1:]:
            x.append([int(line.split(',')[-1])])

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = []
        for line in edges:
            node1, node2 = line.split(',')
            edge_index.append([int(node1), int(node2)])

        #TODO split dataset into training, test, val sets, remove some edges

        x = torch.tensor(x, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
test = ProcessData("fb-pages-food")
test.process()