import torch
import numpy as np
import random
from collections import OrderedDict
import json
import os

class FederatedServer:
    def __init__(self, clients, global_model, config):
        self.clients = clients
        self.global_model = global_model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        
        self.best_accuracy = 0.0
        self.training_history = []
        
    def select_clients(self, round_num):
        """Select exactly 10/30 clients per round"""
        num_selected = 10  # YOUR REQUIREMENT: 10/30 per round
        selected = random.sample(self.clients, num_selected)
        print(f"Round {round_num}: Selected {len(selected)} clients")
        return selected
    
    def secure_aggregate(self, client_updates):
        """Secure aggregation with noise injection"""
        if not client_updates:
            return self.global_model.state_dict()
        
        global_dict = self.global_model.state_dict()
        aggregated_dict = OrderedDict()
        
        # FedAvg aggregation
        for key in global_dict.keys():
            # Average client updates
            averaged = torch.zeros_like(global_dict[key])
            for update in client_updates:
                averaged += update[key]
            averaged /= len(client_updates)
            
            # Add secure aggregation noise
            if self.config.USE_SECURE_AGGREGATION:
                noise = torch.randn_like(averaged) * 0.01
                averaged += noise
            
            aggregated_dict[key] = averaged
        
        return aggregated_dict
    
    def evaluate_global_model(self):
        """Evaluate on all clients"""
        self.global_model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for client in self.clients:
                for X, y in client.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.global_model(X)
                    _, predicted = torch.max(outputs, 1)
                    total_samples += y.size(0)
                    total_correct += (predicted == y).sum().item()
        
        accuracy = total_correct / total_samples
        return accuracy
    
    def run_federated_training(self):
        """Main FL training loop"""
        print(f"🚀 Starting FL Training: Target >{self.config.TARGET_ACCURACY:.1%} accuracy")
        print(f"Privacy: ε={self.config.EPSILON}, δ={self.config.DELTA}")
        
        for round_num in range(1, self.config.ROUNDS + 1):
            print(f"\n--- ROUND {round_num}/{self.config.ROUNDS} ---")
            
            # Select clients (10/30)
            selected_clients = self.select_clients(round_num)
            
            # Collect updates with DP
            client_updates = []
            privacy_costs = []
            
            for client in selected_clients:
                update = client.train_local_model(self.global_model)
                client_updates.append(update)
                privacy_costs.append(client.privacy_spent)
            
            # Secure aggregation
            aggregated_update = self.secure_aggregate(client_updates)
            self.global_model.load_state_dict(aggregated_update)
            
            # Evaluate
            accuracy = self.evaluate_global_model()
            avg_privacy = np.mean(privacy_costs) if privacy_costs else 0
            
            print(f"Accuracy: {accuracy:.4f} | Privacy: {avg_privacy:.2f}")
            
            # Save best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                torch.save(self.global_model.state_dict(), 
                          f"{self.config.MODEL_SAVE_PATH}best_model.pt")
                print(f"✓ New best: {accuracy:.4f}")
            
            # Check target accuracy
            if accuracy >= self.config.TARGET_ACCURACY:
                print(f"🎯 TARGET ACCURACY {self.config.TARGET_ACCURACY:.1%} REACHED!")
                break
            
            # Save progress
            self.training_history.append({
                'round': round_num,
                'accuracy': accuracy,
                'privacy_cost': avg_privacy
            })
        
        # Save final results
        with open("results/training_results.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"✅ Training completed! Best accuracy: {self.best_accuracy:.4f}")
        return self.global_model, self.best_accuracy
