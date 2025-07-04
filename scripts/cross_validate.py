import yaml
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from models.mobile_optimized import CNN_LSTM_Attn

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Paths & device
PROC_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_all_data(split):
    X = np.vstack([np.load(p) for p in PROC_DIR.glob(f"subject_*/X_win_{split}.npy")])
    y = np.hstack([np.load(p) for p in PROC_DIR.glob(f"subject_*/y_win_{split}.npy")])
    return X, y

def main():
    X, y = load_all_data("train")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        train_ds = TensorDataset(torch.tensor(X_tr).float(), torch.tensor(y_tr).long())
        val_ds   = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
        tr_loader = DataLoader(train_ds, batch_size=cfg["federated"]["batch_size"], shuffle=True)
        vl_loader = DataLoader(val_ds,   batch_size=128, shuffle=False)

        model = CNN_LSTM_Attn().to(DEV)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg["training"]["learning_rate"],
                                      weight_decay=cfg["training"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=cfg["scheduler"]["base_lr"],
            max_lr=cfg["scheduler"]["max_lr"],
            step_size_up=cfg["scheduler"]["step_size_up"],
            mode="triangular"
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])

        best_acc = 0
        for epoch in range(1, cfg["federated"]["local_epochs"] + 1):
            model.train()
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(DEV), yb.to(DEV)
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                optimizer.step()
                scheduler.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for Xb, yb in vl_loader:
                    preds = model(Xb.to(DEV)).argmax(1)
                    correct += (preds == yb.to(DEV)).sum().item()
                    total += yb.size(0)
            acc = correct / total
            best_acc = max(best_acc, acc)

        print(f"Fold {fold} best accuracy: {best_acc:.4f}")
        fold_accuracies.append(best_acc)

    print(f"Cross-validation mean accuracy: {np.mean(fold_accuracies):.4f}")

if __name__ == "__main__":
    main()
