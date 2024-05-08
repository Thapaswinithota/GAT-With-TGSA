import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
class GAT_cell(nn.Module):
    def __init__(self, num_features, num_layers, hidden_dim, heads=4, dropout=0.25):
        super(GAT_cell, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dropout = dropout

        # GAT layers
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(1, num_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout))

        # Batch normalization layers
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim * heads) for _ in range(1, num_layers)])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        # Global mean pooling to aggregate features at the graph level
        x = global_mean_pool(x, data.batch)

        return x
