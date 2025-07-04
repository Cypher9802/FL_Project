import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CyclicLR
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import yaml
from pathlib import Path
from imblearn.over_sampling import SMOTE
import numpy as np

# Load configuration
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

class Client:
    def __init__(self, client_id, data, device=None):
        self.client_id = client_id
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        
        # Prepare data with SMOTE balancing for training
        X_train, y_train = data['train']['X'], data['train']['y']
        X_val, y_val = data['validate']['X'], data['validate']['y']
        
        # Apply SMOTE balancing to training data
        X_train_balanced, y_train_balanced = self._balance_data(X_train, y_train)
        
        # Create data loaders
        self.train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train_balanced, dtype=torch.float32),
                torch.tensor(y_train_balanced, dtype=torch.long)
            ),
            batch_size=config["federated"]["batch_size"],
            shuffle=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            ),
            batch_size=128,
            shuffle=False
        )
        
        self.privacy_spent = 0.0
        self.local_epochs = config["federated"]["local_epochs"]
        
    def _balance_data(self, X, y):
        """Apply SMOTE balancing to training data."""
        if len(X.shape) == 3:  # (N, window_size, features)
            n, w, f = X.shape
            X_flat = X.reshape(n, w * f)
        else:
            X_flat = X
        
        try:
            smote = SMOTE(k_neighbors=config["smote_k_neighbors"])
            X_balanced, y_balanced = smote.fit_resample(X_flat, y)
            
            if len(X.shape) == 3:
                X_balanced = X_balanced.reshape(-1, w, f)
            
            return X_balanced, y_balanced
        except Exception as e:
            print(f"SMOTE balancing failed for client {self.client_id}: {e}")
            return X, y
    
    def local_train(self, global_model):
        """
        Enhanced local training with cyclical learning rate and label smoothing.
        """
        # Debug: print batch shape
        sample = next(iter(self.train_loader))[0]
        print(f"Client {self.client_id} batch shape: {sample.shape}")
        
        # Create local model copy
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        
        # Enhanced optimizer with better hyperparameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        
        # Cyclical learning rate scheduler
        scheduler = CyclicLR(
            optimizer,
            base_lr=config["scheduler"]["base_lr"],
            max_lr=config["scheduler"]["max_lr"],
            step_size_up=config["scheduler"]["step_size_up"],
            mode="triangular"
        )
        
        # Loss function with label smoothing
        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=config["training"]["label_smoothing"]
        )
        
        # Setup differential privacy if enabled
        if config["privacy"]["epsilon"] > 0:
            privacy_engine = PrivacyEngine()
            model, optimizer, train_loader_private = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                epochs=self.local_epochs,
                target_epsilon=config["privacy"]["epsilon"] / config["federated"]["rounds"],
                target_delta=config["privacy"]["delta"],
                max_grad_norm=config["training"]["max_grad_norm"]
            )
            data_loader = train_loader_private
        else:
            data_loader = self.train_loader
        
        # Training loop
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            if config["privacy"]["epsilon"] > 0:
                with BatchMemoryManager(
                    data_loader=data_loader,
                    max_physical_batch_size=config["federated"]["batch_size"],
                    optimizer=optimizer
                ) as memory_safe_loader:
                    for X, y in memory_safe_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        
                        optimizer.zero_grad()
                        outputs = model(X)
                        loss = criterion(outputs, y)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
            else:
                for X, y in data_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config["training"]["max_grad_norm"]
                    )
                    
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"Client {self.client_id}, Epoch {epoch+1}/{self.local_epochs}: Loss = {avg_loss:.4f}")
        
        # Update privacy spent
        if config["privacy"]["epsilon"] > 0:
            self.privacy_spent = privacy_engine.get_epsilon(config["privacy"]["delta"])
        
        return model.state_dict()
    
    def evaluate(self, model):
        """Evaluate model on validation data."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def get_data_size(self):
        """Return the size of client's training data."""
        return len(self.train_loader.dataset)
    
    def get_privacy_spent(self):
        """Return the privacy budget spent by this client."""
        return self.privacy_spent
