#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    print("=== Federated Learning Privacy Analysis ===")
    
    # 1. Privacy budget analysis
    privacy_path = Path("server_privacy_analysis.json")
    if privacy_path.exists():
        privacy_data = json.loads(privacy_path.read_text())
        rounds = [r['round'] for r in privacy_data['round_by_round']]
        cumulative_eps = [r['cumulative_epsilon'] for r in privacy_data['round_by_round']]
        
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, cumulative_eps, 'b-o', linewidth=2)
        plt.axhline(y=privacy_data['target_epsilon'], color='r', linestyle='--', label='Target ε')
        plt.xlabel('Communication Round')
        plt.ylabel('Cumulative ε')
        plt.title('Privacy Budget Consumption')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('privacy_budget_analysis.png', dpi=300)
        print("- Generated privacy_budget_analysis.png")
    else:
        print("! No privacy data found (server_privacy_analysis.json missing)")
    
    # 2. Training progress
    training_path = Path("training_metrics.json")
    if training_path.exists():
        training_data = json.loads(training_path.read_text())
        rounds = [m['round'] for m in training_data]
        participants = [m['participants'] for m in training_data]
        losses = [m['average_loss'] for m in training_data]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Participation plot
        ax1.plot(rounds, participants, 'g-s', markersize=6)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Participants')
        ax1.set_title('Client Participation')
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(rounds, losses, 'r-o')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Average Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300)
        print("- Generated training_progress.png")
    else:
        print("! No training metrics found (training_metrics.json missing)")
    
    print("Analysis completed")

if __name__ == "__main__":
    main()
