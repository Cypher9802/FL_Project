from data.data_loader import load_and_preprocess
from models.mobile_optimized import MobileHARModel
from federated.client import Client
from federated.server import Server
from config import Config

def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

data = load_and_preprocess()
clients = [Client(sid, data[sid]) for sid in data]
model = MobileHARModel()
server = Server(clients, model)
server.train()
