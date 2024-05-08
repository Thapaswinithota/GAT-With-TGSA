import torch
import torch.nn as nn
from models.GAT_drug import GAT_drug  # GAT implementation specific to drug data
from models.GAT_cell import GAT_cell  # GAT implementation specific to cell data

class TGDRP(nn.Module):
    def __init__(self, cluster_predefine, args):
        super(TGDRP, self).__init__()
        
        # Setup configuration from args
        self.configure_model(args)
        
        # Setting up GAT modules for drugs and cells with corresponding embedding layers
        self.setup_gat_modules(cluster_predefine)

        # Final regression layers to predict outcomes from combined features
        self.setup_regression_module()

    def configure_model(self, args):
        # Model parameters defined from args for ease of tracking and modifications
        self.batch_size = args.batch_size
        self.layer_drug = args.layer_drug
        self.dim_drug = args.dim_drug
        self.num_feature = args.num_feature
        self.layer_cell = args.layer
        self.dim_cell = args.hidden_dim
        self.dropout_ratio = args.dropout_ratio

    def setup_gat_modules(self, cluster_predefine):
        # Drug branch: GAT for feature extraction followed by a linear transformation and activation
        self.GAT_drug = GAT_drug(self.layer_drug, self.dim_drug)
        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * self.layer_drug, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio)
        )

        # Cell branch: Similar setup as drugs but starting from raw feature dimensions
        self.GAT_cell = GAT_cell(self.num_feature, self.layer_cell, self.dim_cell, cluster_predefine)
        self.cell_emb = nn.Sequential(
            nn.Linear(self.dim_cell * self.GAT_cell.final_node, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio)
        )

    def setup_regression_module(self):
        # Defining a sequence of fully connected layers to regress to the output
        self.regression = nn.Sequential(
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, 1)
        )

    def forward(self, drug, cell):
        # Process drug and cell inputs through their respective GAT and embedding layers
        x_drug = self.drug_emb(self.GAT_drug(drug))
        x_cell = self.cell_emb(self.GAT_cell(cell))

        # Concatenate the features from both branches
        combined_features = torch.cat([x_drug, x_cell], dim=-1)

        # Pass the combined features through the regression model to get the final output
        output = self.regression(combined_features)
        return output
