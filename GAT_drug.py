import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GAT_drug(nn.Module):
    """
    Implements a Graph Attention Network for predicting drug responses using protein-protein interactions.
    """
    def __init__(self, num_features, num_layers, hidden_dim, heads=4, dropout=0.5):
        super(GAT_drug, self).__init__()
        self.num_features = num_features  # Initial node features count
        self.num_layers = num_layers  # Number of GAT layers
        self.hidden_dim = hidden_dim  # Dimension of each hidden layer
        self.heads = heads  # Attention heads in each GAT layer
        self.dropout = dropout  # Dropout rate

        # Initialize GAT layers with attention mechanisms
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = num_features if i == 0 else hidden_dim * heads
            out_channels = hidden_dim if i < num_layers - 1 else hidden_dim
            concat = True if i < num_layers - 1 else False  # Concatenate for depth, not last layer
            self.gat_layers.append(GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout))

        # Post-processing layers for final prediction
        self.post_proc = nn.Sequential(
            nn.Linear(hidden_dim * heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # Extract features and connections

        # Process through GAT layers with ReLU activation
        for gat_layer in self.gat_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout
            x = gat_layer(x, edge_index)  # Apply GAT
            x = F.relu(x)  # Apply ReLU

        # Global mean pooling and final prediction
        x = global_mean_pool(x, data.batch)  # Pool to graph-level feature
        x = self.post_proc(x)  # Make final prediction
        return x

# Example of initializing and describing the model
model = GAT_drug(num_features=77, num_layers=3, hidden_dim=64, heads=2)
print("GAT Drug Model Initialized:\n", model)



