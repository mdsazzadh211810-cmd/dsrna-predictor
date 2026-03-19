# Model Evaluation and Visualization
# AI4S Lab - Sazzad Hossain

import sys
sys.path.append('src/data')
sys.path.append('src/models')

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.loader import DataLoader
from dataset import load_dataset
from gnn_model import dsRNAPredictor

# Data load
graphs = load_dataset("data/raw/sirna_data.csv")
test_graphs = graphs[180:]
test_loader = DataLoader(test_graphs, batch_size=4)

# Model load
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = dsRNAPredictor().to(device)
model.load_state_dict(torch.load('best_model.pt', weights_only=True))
model.eval()

# Predictions collect করো
predicted = []
actual = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        predicted.extend(pred.squeeze().tolist())
        actual.extend(batch.y.tolist())

predicted = np.array(predicted)
actual = np.array(actual)

# Metrics calculate করো
mae = np.mean(np.abs(predicted - actual))
rmse = np.sqrt(np.mean((predicted - actual) ** 2))
correlation = np.corrcoef(predicted, actual)[0, 1]

print(f"MAE:  {mae:.2f}%")
print(f"RMSE: {rmse:.2f}%")
print(f"Correlation: {correlation:.3f}")

# Plot তৈরি করো
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Predicted vs Actual
axes[0].scatter(actual, predicted, alpha=0.7, color='blue')
axes[0].plot([0, 100], [0, 100], 'r--', label='Perfect prediction')
axes[0].set_xlabel('Actual Knockdown (%)')
axes[0].set_ylabel('Predicted Knockdown (%)')
axes[0].set_title(f'Predicted vs Actual\nCorrelation: {correlation:.3f}')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Error distribution
errors = predicted - actual
axes[1].hist(errors, bins=10, color='green', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--')
axes[1].set_xlabel('Prediction Error (%)')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Error Distribution\nMAE: {mae:.2f}%')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('results.png', dpi=150)
print("\nPlot saved: results.png")