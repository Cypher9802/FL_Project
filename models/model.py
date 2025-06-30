import torch
import torch.nn as nn
import tempfile
import os

class MobileHARModel(nn.Module):
    """Mobile-optimized HAR model <5MB for federated learning"""
    
    def __init__(self, config):
        super(MobileHARModel, self).__init__()
        self.config = config
        
        # Lightweight CNN for time-series data
        self.features = nn.Sequential(
            # First conv block - efficient feature extraction
            nn.Conv1d(config.INPUT_FEATURES, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Second conv block - depthwise separable for efficiency
            nn.Conv1d(32, 32, kernel_size=5, groups=32, padding=2),  # Depthwise
            nn.Conv1d(32, 64, kernel_size=1),  # Pointwise
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # Third block - final feature extraction
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(4),  # Fixed output size
        )
        
        # Compact classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, config.NUM_CLASSES)
        )
        
        self._verify_size_constraint()
    
    def _verify_size_constraint(self):
        """Ensure model is under 5MB"""
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            torch.save(self.state_dict(), tmp.name)
            size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        
        if size_mb > self.config.MAX_MODEL_SIZE_MB:
            raise ValueError(f"Model {size_mb:.2f}MB exceeds {self.config.MAX_MODEL_SIZE_MB}MB limit")
        
        print(f"✓ Model size: {size_mb:.2f}MB (compliant)")
    
    def forward(self, x):
        # Handle input format: (batch, timesteps, features) -> (batch, features, timesteps)
        if x.dim() == 3 and x.size(2) == self.config.INPUT_FEATURES:
            x = x.transpose(1, 2)
        
        features = self.features(x)
        return self.classifier(features)
