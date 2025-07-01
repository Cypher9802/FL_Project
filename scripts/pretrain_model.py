import sys
from pathlib import Path

# Add project root to sys.path for clean imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data.data_loader import load_and_preprocess

# Device auto-detection (works on Mac, Linux, Windows)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Feedforward Neural Network as per your PDF
class HAR_FNN(nn.Module):
    def __init__(self, input_dim=561, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Load and preprocess data
subject_data = load_and_preprocess()
if subject_data is None:
    print("Dataset not found or not loaded. Exiting.")
    sys.exit(1)

# Merge all subjects for centralized training
X_train, y_train, X_val, y_val = [], [], [], []
for subj in subject_data:
    X_train.append(subject_data[subj]['train']['X'])
    y_train.append(subject_data[subj]['train']['y'])
    X_val.append(subject_data[subj]['validate']['X'])
    y_val.append(subject_data[subj]['validate']['y'])
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
X_val = np.concatenate(X_val)
y_val = np.concatenate(y_val)

# If data is 3D (samples, timesteps, features), flatten to 2D (samples, features*timesteps)
if X_train.ndim == 3:
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

# Torch datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.long)
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

# Model, optimizer, loss
model = HAR_FNN(input_dim=X_train.shape[1], num_classes=6).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
EPOCHS = 30
best_acc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"Epoch {epoch:02d}: loss={total_loss/len(train_loader):.4f}, val_acc={acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/centralized_har_fnn.pt")

print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
print("Model saved to models/centralized_har_fnn.pt")
