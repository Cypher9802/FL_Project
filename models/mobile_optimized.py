import torch
import torch.nn as nn
import tempfile
import os
import yaml
from pathlib import Path

# Load configuration
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

class MobileHARModel(nn.Module):
    """
    Mobile-optimized CNN model for Human Activity Recognition.
    Designed for efficiency while maintaining high accuracy.
    """
    def __init__(self):
        super().__init__()
        
        # Model parameters from config
        input_features = config["input_features"]
        num_classes = config["num_classes"]
        max_model_size = config["max_model_size_mb"]
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First convolution block
            nn.Conv1d(input_features, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Depthwise separable convolution
            nn.Conv1d(32, 32, kernel_size=5, padding=2, groups=32),  # Depthwise
            nn.Conv1d(32, 64, kernel_size=1),  # Pointwise
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True),
            
            # Second convolution block
            nn.Conv1d(64, 64, kernel_size=5, padding=2, groups=64),  # Depthwise
            nn.Conv1d(64, 128, kernel_size=1),  # Pointwise
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool1d(4),
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Check model size
        self._check_model_size(max_model_size)

    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.
        Accepts input of shape (batch, 128, 9) or (batch, 9, 128)
        """
        if x.dim() == 3:
            if x.shape[1] == config["input_size"] and x.shape[2] == config["input_features"]:
                x = x.permute(0, 2, 1)  # (batch, features, timesteps)
            elif x.shape[1] == config["input_features"] and x.shape[2] == config["input_size"]:
                pass  # Already in correct format
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
        
        features = self.features(x)
        output = self.classifier(features)
        return output

    def _check_model_size(self, max_mb):
        """Check if model size is within the specified limit."""
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            torch.save(self.state_dict(), tmp.name)
            size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
            if size_mb > max_mb:
                raise RuntimeError(f"Model size {size_mb:.2f}MB exceeds {max_mb}MB limit")
            print(f"Model size: {size_mb:.2f}MB (within limit)")

    def get_model_size(self):
        """Get model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def get_num_parameters(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

class CNN_LSTM_Attn(nn.Module):
    """
    Enhanced CNN-LSTM model with attention for high accuracy.
    This model targets ≥95% accuracy on UCI-HAR dataset.
    """
    def __init__(self, channels=None, hidden=None, nclass=None, 
                 kernel_size=7, dropout1=0.2, dropout2=0.3):
        super().__init__()
        
        # Load parameters from config
        channels = channels or config["input_features"]
        hidden = hidden or config["lstm_hidden"]
        nclass = nclass or config["num_classes"]
        n_heads = config["n_heads"]
        
        # CNN feature extraction with residual connections
        self.conv1 = nn.Conv1d(channels, 64, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout1)
        
        # Residual connection
        self.residual = nn.Conv1d(128, 128, 1)
        
        # LSTM with bidirectional processing
        self.lstm = nn.LSTM(
            128, hidden, 
            num_layers=2,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout1
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden * 2, n_heads, 
            batch_first=True,
            dropout=dropout2
        )
        
        # Classification layers
        self.dropout2 = nn.Dropout(dropout2)
        self.fc1 = nn.Linear(hidden * 2, 64)
        self.activation = nn.SiLU()
        self.fc2 = nn.Linear(64, nclass)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass with CNN -> LSTM -> Attention -> Classification.
        Input shape: (batch, sequence_length, features)
        """
        # Ensure correct input format: (batch, features, sequence_length)
        if x.shape[1] == config["input_size"] and x.shape[2] == config["input_features"]:
            x = x.transpose(1, 2)  # (batch, features, sequence_length)
        
        # CNN feature extraction
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout1(self.activation(self.bn2(self.conv2(out))))
        
        # Residual connection
        out = out + self.residual(out)
        
        # Transpose for LSTM: (batch, sequence_length, features)
        out = out.transpose(1, 2)
        
        # LSTM processing
        lstm_out, _ = self.lstm(out)
        
        # Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over sequence dimension
        h = attn_out.mean(dim=1)
        
        # Classification
        h = self.dropout2(self.activation(self.fc1(h)))
        output = self.fc2(h)
        
        return output

    def get_model_size(self):
        """Get model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
