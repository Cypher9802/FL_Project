import torch
import torch.nn as nn
import yaml
from pathlib import Path

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

class CNN_LSTM_Attn(nn.Module):
    """
    Mobile-optimized CNN + BiLSTM + Self-Attention for HAR.
    """
    def __init__(self,
                 channels: int = None,
                 hidden: int = None,
                 nclass: int = None,
                 kernel_size: int = 7,
                 dropout1: float = 0.2,
                 dropout2: float = 0.3):
        super().__init__()
        channels = channels or cfg["input_features"]
        hidden   = hidden   or cfg["lstm_hidden"]
        nclass   = nclass   or cfg["num_classes"]
        n_heads  = cfg["n_heads"]

        # CNN backbone with residual shortcut
        self.conv1    = nn.Conv1d(channels, 64,  kernel_size, padding=kernel_size//2)
        self.bn1      = nn.BatchNorm1d(64)
        self.conv2    = nn.Conv1d(64,      128, kernel_size, padding=kernel_size//2)
        self.bn2      = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout1)
        self.res      = nn.Conv1d(128, 128, 1)

        # BiLSTM
        self.lstm     = nn.LSTM(128, hidden, num_layers=2,
                                batch_first=True, bidirectional=True)

        # Self-attention
        self.attn     = nn.MultiheadAttention(hidden*2, n_heads, batch_first=True)
        self.dropout2 = nn.Dropout(dropout2)

        # Classifier
        self.fc1      = nn.Linear(hidden*2, 64)
        self.act      = nn.SiLU()
        self.fc2      = nn.Linear(64, nclass)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        # Transpose to (batch, features, seq_len)
        if x.shape[1] == cfg["input_size"] and x.shape[2] == cfg["input_features"]:
            x = x.transpose(1, 2)

        # CNN
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout1(self.act(self.bn2(self.conv2(out))))
        out = out + self.res(out)

        # Prepare for LSTM: (batch, seq_len, channels)
        out = out.transpose(1, 2)

        # BiLSTM
        lstm_out, _ = self.lstm(out)

        # Self-attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)

        # Global average pooling over time
        h = attn_out.mean(dim=1)

        # Classifier
        h = self.dropout2(self.act(self.fc1(h)))
        return self.fc2(h)
