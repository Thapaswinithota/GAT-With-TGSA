import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SA(nn.Module):
    def __init__(self, drug_nodes, cell_nodes, drug_edges, cell_edges, args):
        super(SA, self).__init__()
        # set up the device and dropout settings according to the arguments provided
        self.device = args.device
        self.dropout_p = args.dropout_ratio

        # Made sure to load all node features and edge indices onto the specified device
        self.drug_features = drug_nodes.to(self.device)
        self.drug_edge_index = drug_edges.to(self.device)
        self.cell_features = cell_nodes.to(self.device)
        self.cell_edge_index = cell_edges.to(self.device)

        # I established the initial feature dimensions based on the input sizes
        self.drug_feature_dim = drug_nodes.size(1)
        self.cell_feature_dim = cell_nodes.size(1)

        # Here, I defined the initial transformation layers for drug and cell features
        self.drug_feature_transform = nn.Sequential(
            nn.Linear(self.drug_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_p)
        )
        self.cell_feature_transform = nn.Sequential(
            nn.Linear(self.cell_feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_p)
        )

        # For the graph attention layers, I chose GAT because of its ability to focus on important nodes
        self.drug_gat = GATConv(256, 256)  # Adjusted to accept transformed drug features
        self.cell_gat1 = GATConv(256, 1024)  # First layer for cells
        self.cell_gat2 = GATConv(1024, 256)  # Second layer to reduce dimensionality

        # The regression module aims to combine drug and cell features to predict the response
        self.regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(512, 1)
        )

    def forward(self, drug_indices, cell_indices):
        # First, I transform the features using the defined layers
        drug_features = self.drug_feature_transform(self.drug_features)
        cell_features = self.cell_feature_transform(self.cell_features)

        # Apply GAT layers with non-linear activation and dropout to enhance feature representation
        drug_features = self.dropout(F.relu(self.drug_gat(drug_features, self.drug_edge_index)))
        cell_features = self.dropout(F.relu(self.cell_gat1(cell_features, self.cell_edge_index)))
        cell_features = self.dropout(F.relu(self.cell_gat2(cell_features, self.cell_edge_index)))

        # I ensured the features are properly squeezed for concatenation
        drug_features = drug_features.squeeze()
        cell_features = cell_features.squeeze()

        # Concatenate the specific drug and cell features based on the indices from the batch
        combined_features = torch.cat([drug_features[drug_indices], cell_features[cell_indices]], dim=-1)
        # Finally, I use a regression model to predict the response from combined features
        output = self.regression(combined_features)
        return output
