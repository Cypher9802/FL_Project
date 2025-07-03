import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# --- Logging & Device ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                     "mps"    if getattr(torch.backends, "mps", False) and torch.backends.mps.is_available() else
                     "cpu")
log.info(f"Using device: {DEVICE}")

# --- Paths ---
ROOT       = Path(__file__).resolve().parent.parent
PROC_DIR   = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "centralized_har_cnnlstm_best.pt"
MODEL_PATH.parent.mkdir(exist_ok=True)

# --- Hyperparameters ---
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL   = 128
LR               = 3e-4
WEIGHT_DECAY     = 1e-4
EPOCHS           = 60
PATIENCE         = 10  # early stopping patience

# --- Dataset with On-the-Fly Augmentation ---
class HARWindowDataset(Dataset):
    def __init__(self, X, y, augment=False, noise=0.01, scale=0.1):
        self.X = X.astype(np.float32)  # Ensure float32 for MPS!
        self.y = y.astype(np.int64)
        self.augment = augment
        self.noise = noise
        self.scale = scale

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = int(self.y[idx])
        if self.augment:
            x = x + np.random.normal(0, self.noise, x.shape).astype(np.float32)
            factor = 1.0 + np.random.uniform(-self.scale, self.scale)
            x = (x * factor).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

# --- Channel Attention Block ---
class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, bias=False),
            nn.SiLU(),
            nn.Linear(channels // ratio, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: (batch, channels, seq_len)
        avg = self.avg_pool(x).squeeze(-1)
        max = self.max_pool(x).squeeze(-1)
        out = self.fc(avg) + self.fc(max)
        return x * out.unsqueeze(-1)

# --- Tuned CNN–LSTM with Attention ---
class CNN_LSTM_Attn(nn.Module):
    def __init__(self, channels=9, hidden=128, nclass=6, k=7, d1=0.2, d2=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=k, padding=k//2),
            nn.BatchNorm1d(64), nn.SiLU(),
            ChannelAttention(64),
            nn.Conv1d(64, 128, kernel_size=k, padding=k//2),
            nn.BatchNorm1d(128), nn.SiLU(),
            ChannelAttention(128),
            nn.Dropout(d1)
        )
        self.lstm = nn.LSTM(128, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.fc   = nn.Sequential(
            nn.Linear(hidden*2, 64), nn.SiLU(), nn.Dropout(d2),
            nn.Linear(64, nclass)
        )
    def forward(self, x):
        x = x.transpose(1,2)           # (batch,9,128)
        x = self.cnn(x)                # (batch,128,128)
        x = x.transpose(1,2)           # (batch,128,128)
        _, (h, _) = self.lstm(x)       # h: (num_layers*2, batch, hidden)
        h = torch.cat([h[-2], h[-1]], 1)  # (batch, hidden*2)
        return self.fc(h)

# --- Loading Raw Windows ---
def load_raw_windows(split: str):
    Xs, ys = [], []
    for d in PROC_DIR.glob("subject_*"):
        xf = d / f"X_win_{split}.npy"; yf = d / f"y_win_{split}.npy"
        if xf.exists() and yf.exists():
            Xs.append(np.load(xf))
            ys.append(np.load(yf))
    if not Xs:
        log.error(f"No `{split}` windows found in {PROC_DIR}"); sys.exit(1)
    X = np.vstack(Xs).astype(np.float32)
    y = np.hstack(ys).astype(np.int64)
    log.info(f"Loaded {split}: X={X.shape}, y={y.shape}")
    return X, y

# --- Training Loop with Early Stopping ---
def main():
    Xtr, ytr = load_raw_windows("train")
    Xvl, yvl = load_raw_windows("test")

    train_ds = HARWindowDataset(Xtr, ytr, augment=True)
    val_ds   = HARWindowDataset(Xvl, yvl, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE_VAL, shuffle=False)

    model = CNN_LSTM_Attn().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    no_improve = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.float().to(DEVICE)  # ensure float32 for MPS!
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.float().to(DEVICE)  # ensure float32 for MPS!
                yb = yb.to(DEVICE)
                preds = model(Xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        log.info(f"Epoch {epoch:02d}: train_loss={train_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            log.info(f"✔ New best model saved: {best_acc:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                log.info(f"No improvement for {PATIENCE} epochs—early stopping.")
                break

    log.info(f"Training complete. Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
