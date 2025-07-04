import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import numpy as np
from torch import optim, nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from models.mobile_optimized import CNN_LSTM_Attn

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

DEV      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

def load_data(split: str):
    X = np.vstack([np.load(p) for p in PROC_DIR.glob(f"subject_*/X_win_{split}.npy")])
    y = np.hstack([np.load(p) for p in PROC_DIR.glob(f"subject_*/y_win_{split}.npy")])
    return X, y

def main():
    X, y = load_data("train")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}")
        Xtr, Xvl = X[train_idx], X[val_idx]
        ytr, yvl = y[train_idx], y[val_idx]

        tr_ds = TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr).long())
        vl_ds = TensorDataset(torch.tensor(Xvl).float(), torch.tensor(yvl).long())
        tr_loader = DataLoader(tr_ds, batch_size=cfg["federated"]["batch_size"], shuffle=True)
        vl_loader = DataLoader(vl_ds, batch_size=128, shuffle=False)

        model     = CNN_LSTM_Attn().to(DEV)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"])
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=float(cfg["scheduler"]["base_lr"]),
            max_lr=float(cfg["scheduler"]["max_lr"]),
            step_size_up=int(cfg["scheduler"]["step_size_up"]),
            mode="triangular"
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["training"]["label_smoothing"]))

        best_fold_acc = 0.0
        for epoch in range(1, cfg["federated"]["local_epochs"] + 1):
            model.train()
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(DEV), yb.to(DEV)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Validation
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for Xb, yb in vl_loader:
                    preds = model(Xb.to(DEV)).argmax(dim=1)
                    correct += (preds == yb.to(DEV)).sum().item()
                    total += yb.size(0)
            fold_acc = correct / total
            best_fold_acc = max(best_fold_acc, fold_acc)
        print(f"Fold {fold} Best Accuracy: {best_fold_acc:.4f}")
        accs.append(best_fold_acc)

    print(f"Cross-validation Mean Accuracy: {np.mean(accs):.4f}")

if __name__ == "__main__":
    main()
