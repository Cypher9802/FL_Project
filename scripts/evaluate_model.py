#!/usr/bin/env python3
import sys
from pathlib import Path
import warnings

# Suppress unwanted sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure project root on PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data.data_loader import UCIHARDataLoader
from models.neural_network import FeedForwardNN


def load_config():
    cfg_path = Path(__file__).parent.parent / "config" / "config.yaml"
    return yaml.safe_load(open(cfg_path))

def load_privacy_epsilon():
    p = Path("server_privacy_analysis.json")
    if p.exists():
        data = json.loads(p.read_text())
        return data.get("spent_epsilon") or data.get("spent_eps") or data.get("spent_privacy", {}).get("epsilon")
    return None

def main():
    # 1. Load config
    cfg = load_config()

    # 2. Data
    loader = UCIHARDataLoader(cfg)
    _, test_loader = loader.get_data_loaders()

    # 3. Build model without the stray pretrain_path
    mcfg = dict(cfg["model"])
    mcfg.pop("pretrain_path", None)
    model = FeedForwardNN(
        input_size=mcfg["input_size"],
        hidden_layers=mcfg["hidden_layers"],
        num_classes=mcfg["num_classes"],
        dropout_rate=mcfg["dropout_rate"]
    )

    # 4. Load weights
    wp = Path("models/saved/federated_model.pth")
    if wp.exists():
        state = torch.load(wp, map_location="cpu")
        model.load_state_dict(state)
        print(f"\nLoaded model from {wp}\n")
    else:
        print("\nNo trained model found; using random initialization.\n")

    # 5. Evaluate
    model.eval()
    y_true, y_pred = [], []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            pred = out.argmax(dim=1)
            y_true += y.tolist()
            y_pred += pred.tolist()
            correct += (pred == y).sum().item()
            total += len(y)

    acc = correct / total if total else 0.0
    print(f"Accuracy: {acc:.4f}\n")

    # 6. Classification report (only for present labels)
    from sklearn.utils.multiclass import unique_labels
    labels = unique_labels(y_true, y_pred)
    class_names = [
        "WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS",
        "SITTING","STANDING","LAYING"
    ]
    names = [class_names[i] for i in labels]
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0))

    # 7. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    # 8. Privacy ε
    eps = load_privacy_epsilon()
    print(f"\nPrivacy: spent ε = {eps:.4f}" if eps is not None else "\nPrivacy: spent ε = N/A")

if __name__ == "__main__":
    main()
