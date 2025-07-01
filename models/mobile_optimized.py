import torch
import torch.nn as nn
import tempfile
import os
from config import Config

class MobileHARModel(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = Config
        f, t = cfg.INPUT_FEATURES, cfg.INPUT_SIZE
        self.features = nn.Sequential(
            nn.Conv1d(f, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, groups=32),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, cfg.NUM_CLASSES)
        )
        self._check_model_size(cfg.MAX_MODEL_SIZE_MB)

    def forward(self, x):
        # x: (batch, timesteps, features)
        if x.dim() == 3 and x.size(2) == Config.INPUT_FEATURES:
            x = x.transpose(1, 2)
        x = self.features(x)
        return self.classifier(x)

    def _check_model_size(self, max_mb):
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            torch.save(self.state_dict(), tmp.name)
            size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        if size_mb > max_mb:
            raise RuntimeError(f"Model size {size_mb:.2f}MB exceeds {max_mb}MB limit")
        print(f"Model size: {size_mb:.2f}MB (within limit)")
