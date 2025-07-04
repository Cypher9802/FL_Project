import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
import logging
import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from models.neural_network import HARFNN

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

logging.basicConfig(level=logging.INFO)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(cfg["model_save_path"])
MODEL_PATH.parent.mkdir(exist_ok=True)

def load_data():
    """
    Load 561-feature vectors and labels from processed data directory.
    Make sure your preprocessing saves these files:
      - X_vec_train.npy, y_vec_train.npy
      - X_vec_test.npy, y_vec_test.npy
    under each subject folder in data/processed/
    """
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    proc_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    for subj_dir in proc_dir.glob("subject_*"):
        xtr_path = subj_dir / "X_vec_train.npy"
        ytr_path = subj_dir / "y_vec_train.npy"
        xts_path = subj_dir / "X_vec_test.npy"
        yts_path = subj_dir / "y_vec_test.npy"
        if xtr_path.exists() and ytr_path.exists():
            X_train_list.append(np.load(xtr_path))
            y_train_list.append(np.load(ytr_path))
        if xts_path.exists() and yts_path.exists():
            X_test_list.append(np.load(xts_path))
            y_test_list.append(np.load(yts_path))
    Xtr = np.vstack(X_train_list)
    ytr = np.hstack(y_train_list)
    Xvl = np.vstack(X_test_list)
    yvl = np.hstack(y_test_list)
    logging.info(f"Loaded train data: X={Xtr.shape}, y={ytr.shape}")
    logging.info(f"Loaded test data: X={Xvl.shape}, y={yvl.shape}")
    return Xtr, ytr, Xvl, yvl

def main():
    Xtr, ytr, Xvl, yvl = load_data()
    train_ds = TensorDataset(torch.tensor(Xtr).float(), torch.tensor(ytr).long())
    val_ds   = TensorDataset(torch.tensor(Xvl).float(), torch.tensor(yvl).long())
    tr_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    vl_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

    model = HARFNN().to(DEV)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 20
    patience_counter = 0

    for epoch in range(1, 151):  # up to 150 epochs
        model.train()
        total_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEV), yb.to(DEV)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for Xb, yb in vl_loader:
                preds = model(Xb.to(DEV)).argmax(dim=1)
                correct += (preds == yb.to(DEV)).sum().item()
                total += yb.size(0)
        acc = correct / total
        logging.info(f"Epoch {epoch:03d}: loss={total_loss/len(tr_loader):.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict()}, MODEL_PATH)
            logging.info(f"✔ New best model saved: {best_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

    logging.info(f"Training complete. Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
