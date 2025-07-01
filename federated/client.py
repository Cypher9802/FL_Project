import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from config import Config

class Client:
    def __init__(self, client_id, data):
        self.client_id = client_id
        self.device = Config.DEVICE
        X_train, y_train = data['train']['X'], data['train']['y']
        X_val, y_val = data['validate']['X'], data['validate']['y']
        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=Config.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
            batch_size=len(y_val), shuffle=False)
        self.privacy_spent = 0.0

    def local_train(self, global_model):
        # Debug: print shape of a batch
        sample = next(iter(self.train_loader))[0]
        print(f"Client {self.client_id} batch shape: {sample.shape}")
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=Config.LR,
            momentum=Config.MOMENTUM,
            weight_decay=Config.WEIGHT_DECAY)
        if Config.EPSILON > 0:
            privacy_engine = PrivacyEngine()
            model, optimizer, self.train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                epochs=Config.LOCAL_EPOCHS,
                target_epsilon=Config.EPSILON / Config.ROUNDS,
                target_delta=Config.DELTA,
                max_grad_norm=Config.MAX_GRAD_NORM)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(Config.LOCAL_EPOCHS):
            with BatchMemoryManager(
                data_loader=self.train_loader,
                max_physical_batch_size=Config.BATCH_SIZE,
                optimizer=optimizer) as memory_safe_loader:
                for X, y in memory_safe_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(X), y)
                    loss.backward()
                    optimizer.step()
        if Config.EPSILON > 0:
            self.privacy_spent = privacy_engine.get_epsilon(Config.DELTA)
        return model.state_dict()

    def evaluate(self, model):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total
