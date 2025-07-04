import yaml
import optuna
import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

from data.data_loader import save_processed, PROC_DIR
from models.mobile_optimized import CNN_LSTM_Attn

# Load base config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    base_cfg = yaml.safe_load(f)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    dropout1 = trial.suggest_uniform("dropout1", 0.1, 0.5)
    dropout2 = trial.suggest_uniform("dropout2", 0.1, 0.5)
    hidden = trial.suggest_int("lstm_hidden", 64, 256)
    heads = trial.suggest_int("n_heads", 2, 8)

    # Prepare data
    save_processed()
    X, y = (np.vstack([np.load(p) for p in PROC_DIR.glob("subject_*/X_win_train.npy")]),
            np.hstack([np.load(p) for p in PROC_DIR.glob("subject_*/y_win_train.npy")]))
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_ds = TensorDataset(torch.tensor(X_tr).float(), torch.tensor(y_tr).long())
    val_ds   = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
    tr_loader = DataLoader(train_ds, batch_size=base_cfg["federated"]["batch_size"], shuffle=True)
    vl_loader = DataLoader(val_ds, batch_size=128)

    # Build model
    model = CNN_LSTM_Attn(hidden=hidden, n_heads=heads).to(DEV)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=base_cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_cfg["scheduler"]["base_lr"],
        max_lr=base_cfg["scheduler"]["max_lr"],
        step_size_up=base_cfg["scheduler"]["step_size_up"],
        mode="triangular"
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=base_cfg["training"]["label_smoothing"])

    # Training loop (few epochs for tuning)
    for _ in range(10):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEV), yb.to(DEV)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Validation accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in vl_loader:
            preds = model(Xb.to(DEV)).argmax(1)
            correct += (preds == yb.to(DEV)).sum().item()
            total += yb.size(0)
    return correct / total

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    print("Best hyperparameters:", study.best_params)
