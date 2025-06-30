#!/usr/bin/env python3
# Optional centralized pretrain to boost DP utility
import yaml, torch, torch.nn.functional as F
from data.data_loader import load_and_preprocess
from models.mobile_optimized import MobileNetHAR

cfg = yaml.safe_load(open("config/config.yaml"))
data = load_and_preprocess()
model = MobileNetHAR(cfg).to(cfg['training']['device'])
optimizer = torch.optim.SGD(model.parameters(), lr=cfg['training']['learning_rate'])
for epoch in range(5):
    total=0; corr=0
    for X,y in __import__('torch').utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(data[1]['train']['X']), torch.tensor(data[1]['train']['y'])),
        batch_size=cfg['federated']['batch_size'], shuffle=True):
        X,y=X.to(model.net[0].weight.device),y.to(model.net[0].weight.device)
        optimizer.zero_grad()
        loss=F.cross_entropy(model(X.transpose(1,2).float()), y)
        loss.backward(); optimizer.step()
