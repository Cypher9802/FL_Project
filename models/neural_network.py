import torch
import torch.nn as nn
import yaml
from pathlib import Path

# Load configuration
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

class FeedForwardNN(nn.Module):
    """
    Enhanced Feed-Forward Neural Network for HAR classification.
    Supports both raw features and windowed data.
    """
    def __init__(self, input_size=None, hidden_layers=None, num_classes=None, dropout_rate=0.2):
        super().__init__()
        
        # Load parameters from config
        input_size = input_size or (config["input_size"] * config["input_features"])
        hidden_layers = hidden_layers or [256, 128, 64]
        num_classes = num_classes or config["num_classes"]
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for i, hidden_size in enumerate(hidden_layers):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass. Handles both flattened and windowed inputs.
        """
        if x.dim() == 3:  # (batch, sequence, features)
            x = x.view(x.size(0), -1)  # Flatten to (batch, sequence*features)
        
        return self.network(x)
    
    def get_model_size(self):
        """Get model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def get_num_parameters(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    def __init__(self, in_features, out_features, dropout_rate=0.2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        
        out += identity
        out = self.relu(out)
        
        return out

class DeepFeedForwardNN(nn.Module):
    """
    Deep Feed-Forward Neural Network with residual connections.
    Designed for higher accuracy on complex HAR tasks.
    """
    def __init__(self, input_size=None, hidden_layers=None, num_classes=None, dropout_rate=0.2):
        super().__init__()
        
        # Load parameters from config
        input_size = input_size or (config["input_size"] * config["input_features"])
        hidden_layers = hidden_layers or [512, 256, 128, 64]
        num_classes = num_classes or config["num_classes"]
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_layers[0])
        self.input_bn = nn.BatchNorm1d(hidden_layers[0])
        self.input_relu = nn.ReLU(inplace=True)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.residual_blocks.append(
                ResidualBlock(hidden_layers[i], hidden_layers[i + 1], dropout_rate)
            )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the deep network."""
        if x.dim() == 3:  # (batch, sequence, features)
            x = x.view(x.size(0), -1)  # Flatten to (batch, sequence*features)
        
        # Input projection
        x = self.input_dropout(self.input_relu(self.input_bn(self.input_projection(x))))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Output
        x = self.output_layer(x)
        
        return x
    
    def get_model_size(self):
        """Get model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def get_num_parameters(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

class HybridCNN_MLP(nn.Module):
    """
    Hybrid CNN-MLP model combining convolutional and fully connected layers.
    Optimized for both accuracy and efficiency.
    """
    def __init__(self, input_channels=None, input_size=None, num_classes=None):
        super().__init__()
        
        # Load parameters from config
        input_channels = input_channels or config["input_features"]
        input_size = input_size or config["input_size"]
        num_classes = num_classes or config["num_classes"]
        
        # CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(8)
        )
        
        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through CNN and MLP."""
        # Ensure correct input format: (batch, channels, sequence_length)
        if x.shape[1] == config["input_size"] and x.shape[2] == config["input_features"]:
            x = x.transpose(1, 2)
        
        # CNN feature extraction
        features = self.cnn(x)
        
        # MLP classification
        output = self.mlp(features)
        
        return output
    
    def get_model_size(self):
        """Get model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
