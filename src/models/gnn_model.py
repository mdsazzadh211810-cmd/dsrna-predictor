# Improved GNN Model with Dropout
# AI4S Lab - Sazzad Hossain

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class dsRNAPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        # Layer 1
        self.conv1 = GATConv(8, 64, heads=4, concat=False)
        self.bn1 = nn.BatchNorm1d(64)

        # Layer 2
        self.conv2 = GATConv(64, 64, heads=4, concat=False)
        self.bn2 = nn.BatchNorm1d(64)

        # Layer 3 (নতুন)
        self.conv3 = GATConv(64, 32, heads=4, concat=False)
        self.bn3 = nn.BatchNorm1d(32)

        # Dropout - overfitting রোধ করে
        self.dropout = nn.Dropout(p=0.3)

        # Prediction
        self.predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(16, 1)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index).relu()
        x = self.bn1(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index).relu()
        x = self.bn2(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index).relu()
        x = self.bn3(x)

        x = global_mean_pool(x, batch)
        out = self.predictor(x)
        return out

# Test
model = dsRNAPredictor()
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")