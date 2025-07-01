import torch
import random
from .aggregation import secure_aggregate
from config import Config

class Server:
    def __init__(self, clients, model):
        self.clients = clients
        self.model = model.to(Config.DEVICE)
        self.best_acc = 0.0

    def select_clients(self, round_num):
        return random.sample(self.clients, 10)

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for client in self.clients:
                for X, y in client.val_loader:
                    X, y = X.to(self.model.device), y.to(self.model.device)
                    preds = self.model(X).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
        return correct / total

    def train(self):
        for rnd in range(1, Config.ROUNDS + 1):
            selected_clients = self.select_clients(rnd)
            updates = [c.local_train(self.model) for c in selected_clients]
            agg_state = secure_aggregate(
                self.model, updates,
                noise_scale=Config.AGGREGATION_NOISE_SCALE,
                use_secure=Config.USE_SECURE_AGGREGATION)
            self.model.load_state_dict(agg_state)
            acc = self.evaluate()
            print(f"Round {rnd}: Global Accuracy = {acc:.4f}")
            if acc > self.best_acc:
                self.best_acc = acc
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            if acc >= Config.TARGET_ACCURACY:
                print(f"Target accuracy {Config.TARGET_ACCURACY} reached.")
                break
        print(f"Training complete. Best accuracy: {self.best_acc:.4f}")
