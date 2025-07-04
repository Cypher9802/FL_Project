import yaml
import torch
import random
import numpy as np
import logging
from pathlib import Path

from federated.client import Client
from federated.server import Server
from models.mobile_optimized import CNN_LSTM_Attn
from data.data_loader import save_processed, prepare_split

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# Process data and create clients
save_processed()
clients_data = []
for s, ((Xtr, ytr), (Xvl, yvl)) in prepare_split("train").items():
    clients_data.append(Client(
        client_id=s,
        data={"train": {"X": Xtr, "y": ytr}, "validate": {"X": Xvl, "y": yvl}}
    ))

# Initialize server
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
global_model = CNN_LSTM_Attn().to(DEV)
server = Server(clients=clients_data, model=global_model)

def main():
    logging.basicConfig(level=logging.INFO)
    best_acc = server.train()
    print(f"Federated training complete. Best global accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
