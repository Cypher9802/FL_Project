#!/usr/bin/env python3
import yaml, torch, random, numpy as np
from pathlib import Path
from data.data_loader import load_and_preprocess
from models.mobile_optimized import MobileNetHAR
from federated.client import Client
from federated.server import Server

# Load config
cfg = yaml.safe_load(open("config/config.yaml"))

def seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
seed()

# Load data
data = load_and_preprocess()
clients = [Client(sid, data[sid], cfg) for sid in data]

# Initialize model & server
model = MobileNetHAR(cfg)
server = Server(clients, model, cfg)
server.train()
