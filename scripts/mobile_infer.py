#!/usr/bin/env python3
import yaml, torch
from models.mobile_optimized import MobileNetHAR
cfg = yaml.safe_load(open("config/config.yaml"))
model = MobileNetHAR(cfg)
model.load_state_dict(torch.load(cfg['model']['save_path']))
model.eval()
# Example: load a sample from UCI HAR and run inference
