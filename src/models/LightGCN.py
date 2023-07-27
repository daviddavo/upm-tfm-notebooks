import itertools as it

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as PyG
from torch_geometric.nn.conv import MessagePassing

# From https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377
# and https://github.com/microsoft/recommenders/blob/main/recommenders/models/deeprec/models/graphrec/lightgcn.py
class LightGCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')
    
    # def aggregate(self, x, messages, index):
    #     return torch_scatter.scatter(messages, index, self.node_dim, reduce='sum')
    
    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index
        deg = PyG.utils.degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]
        # Start propagating messages
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
# Define model
class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, num_layers: int = 3, embedding_dim: int = 32):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        
        self.embedding = nn.Embedding(n_users + n_items, embedding_dim)
        
        self.layers = nn.ModuleList(it.repeat(LightGCNConv(), num_layers))
        nn.init.normal_(self.embedding.weight, std=0.1)
    
    def forward(self, edge_index):
        embs = [self.embedding.weight]

        for conv in self.layers:
            embs.append(conv(x=embs[-1], edge_index=edge_index))
        
        # perform weighted sum on output of all layers to yield final embedding
        out = torch.mean(torch.stack(embs, dim=0), dim=0)
        return out
        