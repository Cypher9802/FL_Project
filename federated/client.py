import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

class FederatedClient:
    def __init__(self, client_id, data, config):
        self.client_id = client_id
        self.config = config
        self.device = config.DEVICE
        
        # Prepare data
        self.train_loader, self.val_loader = self._prepare_data(data)
        self.privacy_engine = None
        self.privacy_spent = 0.0
        
    def _prepare_data(self, data):
        """Convert data to PyTorch loaders"""
        X_train, y_train = data['train']['X'], data['train']['y']
        X_val, y_val = data['validate']['X'], data['validate']['y']
        
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, 
                                shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
        
        return train_loader, val_loader
    
    def setup_privacy(self, model, optimizer):
        """Setup differential privacy (ε=8.0, δ=1e-5)"""
        self.privacy_engine = PrivacyEngine()
        
        model, optimizer, train_loader = self.privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            epochs=self.config.LOCAL_EPOCHS,
            target_epsilon=self.config.EPSILON / self.config.ROUNDS,
            target_delta=self.config.DELTA,
            max_grad_norm=self.config.MAX_GRAD_NORM,
        )
        
        return model, optimizer, train_loader
    
    def train_local_model(self, global_model):
        """Train with differential privacy"""
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()
        
        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.config.LR,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Apply differential privacy
        if self.config.USE_DIFFERENTIAL_PRIVACY:
            local_model, optimizer, private_loader = self.setup_privacy(local_model, optimizer)
        else:
            private_loader = self.train_loader
        
        criterion = nn.CrossEntropyLoss()
        
        # Local training with DP
        for epoch in range(self.config.LOCAL_EPOCHS):
            with BatchMemoryManager(
                data_loader=private_loader,
                max_physical_batch_size=self.config.BATCH_SIZE,
                optimizer=optimizer
            ) as memory_safe_loader:
                
                for X, y in memory_safe_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = local_model(X)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
        
        # Track privacy expenditure
        if self.privacy_engine:
            self.privacy_spent = self.privacy_engine.get_epsilon(self.config.DELTA)
        
        return local_model.state_dict()
    
    def evaluate(self, model):
        """Evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return correct / total
