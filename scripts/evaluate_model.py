import yaml
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from models.mobile_optimized import CNN_LSTM_Attn

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(cfg["model_save_path"])
PROC_DIR   = Path(__file__).resolve().parent.parent / "data" / "processed"

def load_data(split):
    X = np.vstack([np.load(p) for p in PROC_DIR.glob(f"subject_*/X_win_{split}.npy")])
    y = np.hstack([np.load(p) for p in PROC_DIR.glob(f"subject_*/y_win_{split}.npy")])
    return X, y

def main():
    # Load model
    model = CNN_LSTM_Attn().to(DEV)
    checkpoint = torch.load(MODEL_PATH, map_location=DEV)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Evaluate on train and test
    for split in ["train", "test"]:
        X, y = load_data(split)
        ds = TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
        loader = DataLoader(ds, batch_size=128)

        correct = total = 0
        with torch.no_grad():
            for Xb, yb in loader:
                preds = model(Xb.to(DEV)).argmax(1)
                correct += (preds == yb.to(DEV)).sum().item()
                total += yb.size(0)
        print(f"{split.capitalize()} Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    main()
