import torch
import random
import numpy as np
from typing import List, Dict, Any
import yaml
from pathlib import Path
from .aggregation import secure_aggregate, federated_averaging
from .privacy import PrivacyTracker

# Load configuration
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

class Server:
    def __init__(self, clients, model, device=None):
        self.clients = clients
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.best_acc = 0.0
        self.rounds_completed = 0
        self.accuracy_history = []
        
        # Initialize privacy tracker
        self.privacy_tracker = PrivacyTracker(
            target_epsilon=config["privacy"]["epsilon"],
            target_delta=config["privacy"]["delta"]
        )
        
        # Training configuration
        self.rounds = config["federated"]["rounds"]
        self.clients_per_round = config["federated"]["clients_per_round"]
        self.target_accuracy = config["training"]["target_accuracy"]
        
        print(f"Server initialized with {len(self.clients)} clients")
        print(f"Target accuracy: {self.target_accuracy}")
        print(f"Device: {self.device}")

    def select_clients(self, round_num: int) -> List:
        """
        Select clients for the current round.
        Can implement various selection strategies.
        """
        # Simple random selection
        available_clients = [c for c in self.clients if not c.privacy_tracker.is_budget_exhausted()] if hasattr(self.clients[0], 'privacy_tracker') else self.clients
        
        if len(available_clients) < self.clients_per_round:
            print(f"Warning: Only {len(available_clients)} clients available (requested {self.clients_per_round})")
            return available_clients
        
        selected = random.sample(available_clients, self.clients_per_round)
        print(f"Round {round_num}: Selected {len(selected)} clients")
        return selected

    def evaluate_global_model(self) -> float:
        """
        Evaluate the global model on all clients' validation data.
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for client in self.clients:
                for X, y in client.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == y).sum().item()
                    total_samples += y.size(0)
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        return accuracy

    def evaluate_per_client(self) -> Dict[int, float]:
        """
        Evaluate model performance per client.
        """
        client_accuracies = {}
        
        for client in self.clients:
            accuracy = client.evaluate(self.model)
            client_accuracies[client.client_id] = accuracy
        
        return client_accuracies

    def aggregate_models(self, selected_clients: List, client_updates: List[Dict]) -> Dict:
        """
        Aggregate client model updates.
        """
        # Get client data sizes for weighted averaging
        client_sizes = [client.get_data_size() for client in selected_clients]
        total_size = sum(client_sizes)
        weights = [size / total_size for size in client_sizes]
        
        if config["privacy"]["use_secure_aggregation"]:
            # Use secure aggregation with noise
            aggregated_state = secure_aggregate(
                self.model,
                client_updates,
                noise_scale=config["privacy"]["aggregation_noise_scale"],
                use_secure=True
            )
        else:
            # Use standard FedAvg
            aggregated_state = federated_averaging(client_updates, weights)
        
        return aggregated_state

    def save_model(self, path: str = None):
        """Save the current global model."""
        if path is None:
            path = config["model_save_path"]
        
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict and additional info
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'accuracy_history': self.accuracy_history,
            'rounds_completed': self.rounds_completed,
            'best_accuracy': self.best_acc,
            'config': config
        }
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = None):
        """Load a previously saved model."""
        if path is None:
            path = config["model_save_path"]
        
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.accuracy_history = checkpoint.get('accuracy_history', [])
            self.rounds_completed = checkpoint.get('rounds_completed', 0)
            self.best_acc = checkpoint.get('best_accuracy', 0.0)
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def train(self):
        """
        Main federated training loop.
        """
        print("Starting federated training...")
        print(f"Configuration: {config['federated']}")
        
        for round_num in range(1, self.rounds + 1):
            print(f"\n--- Round {round_num}/{self.rounds} ---")
            
            # Select clients for this round
            selected_clients = self.select_clients(round_num)
            
            if not selected_clients:
                print("No clients available for training. Stopping.")
                break
            
            # Collect client updates
            client_updates = []
            for client in selected_clients:
                print(f"Training client {client.client_id}...")
                try:
                    update = client.local_train(self.model)
                    client_updates.append(update)
                except Exception as e:
                    print(f"Error training client {client.client_id}: {e}")
                    continue
            
            if not client_updates:
                print("No successful client updates. Skipping round.")
                continue
            
            # Aggregate updates
            try:
                aggregated_state = self.aggregate_models(selected_clients, client_updates)
                self.model.load_state_dict(aggregated_state)
            except Exception as e:
                print(f"Error aggregating models: {e}")
                continue
            
            # Evaluate global model
            global_acc = self.evaluate_global_model()
            self.accuracy_history.append(global_acc)
            
            # Evaluate per-client performance
            client_accs = self.evaluate_per_client()
            avg_client_acc = np.mean(list(client_accs.values()))
            
            print(f"Round {round_num} Results:")
            print(f"  Global Accuracy: {global_acc:.4f}")
            print(f"  Average Client Accuracy: {avg_client_acc:.4f}")
            print(f"  Client Accuracies: {client_accs}")
            
            # Update privacy tracking
            if config["privacy"]["epsilon"] > 0:
                sample_rate = self.clients_per_round / len(self.clients)
                self.privacy_tracker.add_round(
                    noise_multiplier=config["privacy"]["noise_multiplier"],
                    sample_rate=sample_rate
                )
                
                privacy_report = self.privacy_tracker.get_privacy_report()
                print(f"  Privacy Spent: ε={privacy_report['spent_epsilon']:.2f}/{privacy_report['target_epsilon']}")
            
            # Save best model
            if global_acc > self.best_acc:
                self.best_acc = global_acc
                self.save_model()
                print(f"  New best model saved! Accuracy: {self.best_acc:.4f}")
            
            # Check if target accuracy reached
            if global_acc >= self.target_accuracy:
                print(f"Target accuracy {self.target_accuracy} reached!")
                break
            
            # Check privacy budget
            if config["privacy"]["epsilon"] > 0 and self.privacy_tracker.is_budget_exhausted():
                print("Privacy budget exhausted. Stopping training.")
                break
            
            self.rounds_completed = round_num
        
        print(f"\nTraining complete!")
        print(f"Best accuracy: {self.best_acc:.4f}")
        print(f"Rounds completed: {self.rounds_completed}")
        
        if config["privacy"]["epsilon"] > 0:
            final_privacy_report = self.privacy_tracker.get_privacy_report()
            print(f"Final privacy report: {final_privacy_report}")
        
        return self.best_acc

    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training process."""
        return {
            'best_accuracy': self.best_acc,
            'rounds_completed': self.rounds_completed,
            'accuracy_history': self.accuracy_history,
            'num_clients': len(self.clients),
            'clients_per_round': self.clients_per_round,
            'target_accuracy': self.target_accuracy,
            'privacy_report': self.privacy_tracker.get_privacy_report() if config["privacy"]["epsilon"] > 0 else None
        }
