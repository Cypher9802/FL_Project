import torch
import numpy as np
import random
from .aggregation import secure_agg

class Server:
    def __init__(self, clients, model, config):
        self.clients = clients
        self.model = model.to(config['training']['device'])
        self.config = config
        self.best_acc = 0.0

    def select(self, rnd):
        return random.sample(self.clients, 10)

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        for c in self.clients:
            for X, y in c.val_loader:
                X, y = X.to(self.model.device), y.to(self.model.device)
                preds = self.model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        return correct/total

    def train(self):
        for rnd in range(1, self.config['federated']['rounds']+1):
            participants = self.select(rnd)
            updates = [c.local_train(self.model) for c in participants]
            agg_state = secure_agg(
                self.model, updates,
                noise_scale=self.config['security']['aggregation_noise_scale'],
                use_secure=self.config['security']['use_secure_aggregation'])
            self.model.load_state_dict(agg_state)
            acc = self.evaluate()
            print(f"Round {rnd}: Global Acc = {acc:.4f}")
            if acc > self.best_acc:
                self.best_acc = acc
                torch.save(self.model.state_dict(), self.config['model']['save_path'])
            if acc >= self.config['model']['target_accuracy']:
                break
        print(f"Training Done: Best Acc = {self.best_acc:.4f}")
