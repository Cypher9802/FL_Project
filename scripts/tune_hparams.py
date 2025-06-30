#!/usr/bin/env python3
import sys
from pathlib import Path
# add project root for imports
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import itertools
import torch
import random
import numpy as np

from data.data_loader import load_and_preprocess
from federated.client import Client
from federated.server import Server
from models.mobile_optimized import MobileNetHAR

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# load base config
cfg = yaml.safe_load(open("config/config.yaml"))

# simple hyperparameter grid
grid = {
    "training.learning_rate": [0.001, 0.005, 0.01],
    "federated.local_epochs": [3, 5],
    "federated.batch_size": [16, 32],
}

keys, values = zip(*grid.items())

for combo in itertools.product(*values):
    # build a working copy of config
    config = yaml.safe_load(open("config/config.yaml"))
    for key, val in zip(keys, combo):
        section, param = key.split(".")
        config[section][param] = val

    print(f"\n🔍 Testing hyperparams: {dict(zip(keys, combo))}")
    set_seed(42)

    # load data and create clients
    data = load_and_preprocess()
    clients = [Client(sid, data[sid], config) for sid in data]

    # init model and server
    model = MobileNetHAR(config)
    server = Server(clients, model, config)

    # train and record accuracy
    _, acc = server.train()
    print(f"→ Resulting accuracy: {acc:.4f}")
