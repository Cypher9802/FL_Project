#!/usr/bin/env python3
import yaml
from federated.privacy import compute_rdp

cfg = yaml.safe_load(open("config/config.yaml"))
eps = compute_rdp(
    cfg['privacy']['epsilon'], cfg['privacy']['delta'],
    cfg['federated']['rounds'], cfg['federated']['client_fraction'],
    cfg['privacy']['noise_multiplier'])
print(f"Privacy ε={eps:.3f}, target ε={cfg['privacy']['epsilon']}")
