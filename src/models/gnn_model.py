# GNN Model for dsRNA Efficacy Prediction
# AI4S Lab - Sazzad Hossain

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class dsRNAPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layer 1: node features প্রসেস করো
        self.conv1 = GATConv(8, 64, heads=4, concat=False)
        
        # Layer 2: আরো গভীরে শেখো
        self.conv2 = GATConv(64, 64, heads=4, concat=False)
        
        # Final prediction
        self.predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        # Message passing
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        
        # Graph level representation
        x = global_mean_pool(x, batch)
        
        # Prediction
        out = self.predictor(x)
        return out

# Test
model = dsRNAPredictor()
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")