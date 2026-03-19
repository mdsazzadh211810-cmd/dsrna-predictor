# Improved Training Pipeline
# AI4S Lab - Sazzad Hossain

import sys
sys.path.append('src/data')
sys.path.append('src/models')

import torch
from torch_geometric.loader import DataLoader
from dataset import load_dataset
from gnn_model import dsRNAPredictor

# Step 1: Data load
graphs = load_dataset("data/raw/sirna_data.csv")

# Step 2: Train/Val/Test split (70/15/15)
n = len(graphs)
train_graphs = graphs[:400]
val_graphs = graphs[400:450]
test_graphs = graphs[450:]

print(f"\nTrain: {len(train_graphs)}")
print(f"Val:   {len(val_graphs)}")
print(f"Test:  {len(test_graphs)}")

# Step 3: DataLoader
train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=4)
test_loader = DataLoader(test_graphs, batch_size=4)

# Step 4: Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = dsRNAPredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Step 5: Training
print("\nTraining started...")
best_val_loss = float('inf')

for epoch in range(1, 301):    # Train
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            val_loss += criterion(pred, batch.y.view(-1, 1)).item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}/100 | Train: {train_loss:.2f} | Val: {val_loss:.2f}")

    # Best model save
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')

# Step 6: Test
print("\nTest Results:")
model.load_state_dict(torch.load('best_model.pt', weights_only=True))
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        for p, a in zip(pred.squeeze().tolist(), batch.y.tolist()):
            print(f"  Predicted: {p:.1f}% | Actual: {a:.1f}%")