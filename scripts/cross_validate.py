#!/usr/bin/env python3
import sys
from pathlib import Path
# add project root for imports
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import numpy as np
from sklearn.model_selection import KFold

from data.data_loader import load_and_preprocess
from federated.client import Client
from federated.server import Server
from models.mobile_optimized import MobileNetHAR

# load config and data
cfg = yaml.safe_load(open("config/config.yaml"))
data = load_and_preprocess()
subjects = list(data.keys())

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(subjects), 1):
    train_subs = [subjects[i] for i in train_idx]
    val_subs   = [subjects[i] for i in val_idx]

    print(f"\n--- Fold {fold} ---")
    # prepare clients
    train_clients = [Client(sid, data[sid], cfg) for sid in train_subs]
    val_clients   = [Client(sid, data[sid], cfg) for sid in val_subs]

    # train on train_clients
    model = MobileNetHAR(cfg)
    server = Server(train_clients, model, cfg)
    _, train_acc = server.train()

    # evaluate on val_clients
    server.clients = val_clients
    val_acc = server.evaluate()
    print(f"Fold {fold} ▶ Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    fold_results.append(val_acc)

mean_val = np.mean(fold_results)
print(f"\n✅ Cross‐validation mean val accuracy: {mean_val:.4f}")
