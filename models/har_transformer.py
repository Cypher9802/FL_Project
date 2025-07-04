# models/har_transformer.py
import torch
import torch.nn as nn
import yaml
from pathlib import Path

# Load config
CFG = yaml.safe_load(open(Path(__file__).resolve().parent.parent/"config"/"config.yaml"))

class ResidualBlockCAM(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=kernel//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        # CBAM-like channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_ch, out_ch//8, 1), nn.ReLU(),
            nn.Conv1d(out_ch//8, out_ch, 1), nn.Sigmoid()
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch!=out_ch else nn.Identity()
    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # apply channel attention
        ca = self.ca(out)
        out = out * ca + identity
        return self.act(out)

class HARTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d_model = CFG["transformer"]["d_model"]  # e.g., 128
        nhead   = CFG["transformer"]["n_heads"]  # e.g., 4
        nlayers = CFG["transformer"]["n_layers"] # e.g., 2
        # initial CNN backbone
        self.backbone = nn.Sequential(
            ResidualBlockCAM(9, 64),
            ResidualBlockCAM(64, 128),
            nn.AdaptiveAvgPool1d(CFG["window_size"])  # keep length
        )
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256, dropout=0.2, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(64, CFG["num_classes"])
        )
    def forward(self, x):
        # x: (batch, seq_len, 9) → (batch, 9, seq_len)
        x = x.transpose(1,2)
        x = self.backbone(x)            # (batch, d_model, seq_len)
        x = x.permute(2,0,1)            # (seq_len, batch, d_model) for transformer
        x = self.transformer(x)         # same shape
        x = x.mean(0)                   # global pooling → (batch, d_model)
        return self.cls_head(x)
