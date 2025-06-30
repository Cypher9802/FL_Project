import torch.nn as nn
import tempfile, os, torch

class MobileNetHAR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        f, t = cfg['dataset']['input_features'], cfg['dataset']['input_size']
        self.net = nn.Sequential(
            nn.Conv1d(f, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,kernel_size=5,padding=2,groups=32),
            nn.Conv1d(64,128,kernel_size=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(),
            nn.Linear(128*4,64), nn.ReLU(),
            nn.Linear(64,cfg['dataset']['num_classes'])
        )
        self._check_size(cfg['model']['max_size_mb'])

    def forward(self, x):
        return self.net(x.transpose(1,2)) if x.dim()==3 else self.net(x)

    def _check_size(self, limit_mb):
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            torch.save(self.state_dict(), tmp.name)
            size = os.path.getsize(tmp.name)/(1024*1024)
        if size>limit_mb: raise RuntimeError(f"Model {size:.2f}MB > {limit_mb}MB")
