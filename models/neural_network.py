import torch
import torch.nn as nn

class HARFNN(nn.Module):
    """
    Feed-Forward Neural Network for UCI HAR Dataset baseline.
    Input: 561 features
    Architecture: 2 hidden layers (128, 64 neurons), ReLU activations, dropout.
    Output: 6 classes
    """
    def __init__(self, input_dim=561, nclass=6, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, nclass)
        )
    
    def forward(self, x):
        return self.net(x)
