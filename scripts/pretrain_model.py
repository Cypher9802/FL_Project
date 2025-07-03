import os
import sys
import logging
import numpy as np
import torch
from pathlib import Path
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

# --- Logging & Device ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
DEV = torch.device('cuda' if torch.cuda.is_available() else
                   'mps'    if getattr(torch.backends, "mps", False) and torch.backends.mps.is_available() else
                   'cpu')
log.info(f"Using device: {DEV}")

# --- Paths ---
ROOT       = Path(__file__).resolve().parent.parent
PROC_DIR   = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "centralized_har_cnnlstm_best.pt"
MODEL_PATH.parent.mkdir(exist_ok=True)

# --- Load Raw Windows ---
def load_raw_windows(proc_dir: Path, split: str):
    X_list, y_list = [], []
    for subj in proc_dir.glob("subject_*"):
        xp = subj / f"X_win_{split}.npy"
        yp = subj / f"y_win_{split}.npy"
        if xp.exists() and yp.exists():
            X_list.append(np.load(xp))
            y_list.append(np.load(yp))
    if not X_list:
        log.error(f"No `{split}` windows found in {proc_dir}")
        sys.exit(1)
    X = np.vstack(X_list)  # (N,128,9)
    y = np.hstack(y_list)
    log.info(f"Loaded {split}: X={X.shape}, y={y.shape}")
    return X, y

# --- Model: Tuned CNN-LSTM ---
class CNN_LSTM(nn.Module):
    def __init__(self, channels=9, hidden=128, nclass=6, kernel_size=7, dropout1=0.2, dropout2=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(64), nn.SiLU(),  # Swish activation
            nn.Conv1d(64, 128, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(128), nn.SiLU(),
            nn.Dropout(dropout1)
        )
        self.lstm = nn.LSTM(128, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.fc   = nn.Sequential(
            nn.Linear(hidden*2, 64), nn.SiLU(), nn.Dropout(dropout2),
            nn.Linear(64, nclass)
        )
    def forward(self, x):
        x = x.transpose(1,2)          # (batch,9,128)
        x = self.cnn(x)               # (batch,128,128)
        x = x.transpose(1,2)          # (batch,128,128)
        _, (h, _) = self.lstm(x)      # h: (num_layers*2, batch, hidden)
        h = torch.cat([h[-2], h[-1]], 1)  # (batch, hidden*2)
        return self.fc(h)

def main():
    # Load train/val raw windows
    Xtr, ytr = load_raw_windows(PROC_DIR, "train")
    Xvl, yvl = load_raw_windows(PROC_DIR, "test")

    # Torch Datasets & Loaders
    train_ds = TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr).long())
    val_ds   = TensorDataset(torch.tensor(Xvl).float(), torch.tensor(yvl).long())
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)

    # Instantiate model, optimizer, loss, scheduler (tuned hyperparams)
    model     = CNN_LSTM(kernel_size=7, dropout1=0.2, dropout2=0.3, hidden=128).to(DEV)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, 81):
        model.train()
        train_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEV), yb.to(DEV)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval(); correct=total=0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEV), yb.to(DEV)
                preds = model(Xb).argmax(1)
                correct += (preds==yb).sum().item(); total += yb.size(0)
        val_acc = correct/total
        log.info(f"Epoch {epoch:02d}: train_loss={train_loss/len(train_loader):.4f}  val_acc={val_acc:.4f}")
        if val_acc>best_acc:
            best_acc=val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            log.info(f"✔ New best model saved: {best_acc:.4f}")

    log.info(f"Training complete. Best validation accuracy: {best_acc:.4f}")

if __name__=="__main__":
    main()
