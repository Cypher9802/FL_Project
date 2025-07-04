import yaml
import logging
import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CyclicLR
from pathlib import Path

from data.data_loader import save_processed, PROC_DIR
from models.mobile_optimized import CNN_LSTM_Attn

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Setup
logging.basicConfig(level=logging.INFO)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(cfg["model_save_path"])
MODEL_PATH.parent.mkdir(exist_ok=True)

def load_data():
    save_processed()
    Xtr = np.vstack([np.load(p) for p in PROC_DIR.glob("subject_*/X_win_train.npy")])
    ytr = np.hstack([np.load(p) for p in PROC_DIR.glob("subject_*/y_win_train.npy")])
    Xvl = np.vstack([np.load(p) for p in PROC_DIR.glob("subject_*/X_win_test.npy")])
    yvl = np.hstack([np.load(p) for p in PROC_DIR.glob("subject_*/y_win_test.npy")])
    return Xtr, ytr, Xvl, yvl

def main():
    Xtr, ytr, Xvl, yvl = load_data()
    train_ds = TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr).long())
    val_ds   = TensorDataset(torch.tensor(Xvl).float(), torch.tensor(yvl).long())
    tr_loader = DataLoader(train_ds, batch_size=cfg["federated"]["batch_size"], shuffle=True)
    vl_loader = DataLoader(val_ds,   batch_size=128)

    model = CNN_LSTM_Attn().to(DEV)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = CyclicLR(
        optimizer,
        base_lr=cfg["scheduler"]["base_lr"],
        max_lr=cfg["scheduler"]["max_lr"],
        step_size_up=cfg["scheduler"]["step_size_up"],
        mode="triangular"
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])

    best_acc = 0
    for epoch in range(1, cfg["federated"]["local_epochs"] + 1):
        model.train()
        tot_loss = 0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEV), yb.to(DEV)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tot_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for Xb, yb in vl_loader:
                preds = model(Xb.to(DEV)).argmax(1)
                correct += (preds == yb.to(DEV)).sum().item()
                total += yb.size(0)
        acc = correct / total
        logging.info(f"Epoch {epoch:02d}: loss={tot_loss/len(tr_loader):.4f} acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "accuracy_history": best_acc
            }, MODEL_PATH)

    logging.info(f"Training complete. Best acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()
