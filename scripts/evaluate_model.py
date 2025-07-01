#!/usr/bin/env python3
import yaml, torch
from data.data_loader import load_and_preprocess
from models.mobile_optimized import MobileNetHAR

cfg = yaml.safe_load(open("config/config.yaml"))
model = MobileNetHAR(cfg)
model.load_state_dict(torch.load(cfg['model']['save_path']))
model.eval()

data = load_and_preprocess()
accs = []
for sid, d in data.items():
    Xv, yv = d['validate']['X'], d['validate']['y']
    preds = model(torch.tensor(Xv).transpose(1,2).float()).argmax(1)
    accs.append((preds.numpy()==yv).mean())
print(f"Overall Accuracy: {sum(accs)/len(accs):.4f}")
