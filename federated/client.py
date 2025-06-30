import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

class Client:
    def __init__(self, cid, data, config):
        self.id = cid
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train, y_train = data['train']['X'], data['train']['y']
        X_val,   y_val   = data['validate']['X'], data['validate']['y']
        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
            batch_size=self.config['federated']['batch_size'], shuffle=True, drop_last=True)
        self.val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
            batch_size=len(y_val), shuffle=False)
        self.privacy_spent = 0.0

    def local_train(self, global_model):
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            momentum=self.config['training']['momentum'],
            weight_decay=self.config['training']['weight_decay'])
        if self.config['privacy']['epsilon'] > 0:
            engine = PrivacyEngine()
            model, optimizer, self.train_loader = engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                epochs=self.config['federated']['local_epochs'],
                target_epsilon=self.config['privacy']['epsilon'] / self.config['federated']['rounds'],
                target_delta=self.config['privacy']['delta'],
                max_grad_norm=self.config['privacy']['max_grad_norm'])
        for _ in range(self.config['federated']['local_epochs']):
            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(X), y)
                loss.backward()
                optimizer.step()
        if self.config['privacy']['epsilon'] > 0:
            self.privacy_spent = engine.get_epsilon(self.config['privacy']['delta'])
        return model.state_dict()
