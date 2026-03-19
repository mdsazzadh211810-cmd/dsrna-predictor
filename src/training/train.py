# Model Training
# AI4S Lab - Sazzad Hossain

import sys
sys.path.append('src/data')
sys.path.append('src/models')

import torch
from torch_geometric.loader import DataLoader
from dataset import load_dataset
from gnn_model import dsRNAPredictor

# Step 1: Data load করো
graphs = load_dataset("data/raw/sirna_data.csv")

# Step 2: Train/Test split করো
train_graphs = graphs[:8]
test_graphs = graphs[8:]

# Step 3: DataLoader তৈরি করো
train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=2)

# Step 4: Model তৈরি করো
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = dsRNAPredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Step 5: Training loop
print("\nTraining just started...")
for epoch in range(1, 51):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/50 | Loss: {total_loss:.4f}")

print("\nTraining finished!")

# Step 6: Test করো
model.eval()
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        print(f"\nPredicted: {pred.squeeze().tolist()}")
        print(f"Actual:    {batch.y.tolist()}")