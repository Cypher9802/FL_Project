import yaml
import torch
import numpy as np
from pathlib import Path

from models.mobile_optimized import MobileHARModel

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

DEV        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(cfg["model_save_path"])

def infer(window: np.ndarray):
    model = MobileHARModel().to(DEV)
    checkpoint = torch.load(MODEL_PATH, map_location=DEV)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x = torch.tensor(window).float().to(DEV).unsqueeze(0)
        preds = model(x).argmax(1).item()
    return preds

if __name__ == "__main__":
    # Example: load a single window and run inference
    test_window = np.load("data/processed/subject_01/X_win_test.npy")[0]
    label = infer(test_window)
    print(f"Predicted class: {label}")
