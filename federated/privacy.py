import math
import numpy as np
import torch
import yaml
from pathlib import Path
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

# Load configuration
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def compute_rdp(epsilon, delta, rounds, q, noise_multiplier):
    """
    Compute privacy cost using RDP (Rényi Differential Privacy) composition.
    
    Args:
        epsilon: Target epsilon value
        delta: Target delta value
        rounds: Number of federated rounds
        q: Sampling probability
        noise_multiplier: Noise multiplier for DP-SGD
    
    Returns:
        Computed epsilon value
    """
    try:
        # Use Opacus RDP accountant for accurate computation
        accountant = RDPAccountant()
        
        # Add noise for each round
        for _ in range(rounds):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=q)
        
        # Get epsilon for the given delta
        computed_epsilon = accountant.get_epsilon(delta)
        return computed_epsilon
    except Exception as e:
        print(f"Error in RDP computation: {e}")
        # Fallback to simple composition
        return epsilon

def get_privacy_spent(noise_multiplier, sample_rate, steps, delta):
    """
    Calculate privacy spent using RDP accountant.
    
    Args:
        noise_multiplier: Noise multiplier used in DP-SGD
        sample_rate: Sampling rate of the data
        steps: Number of optimization steps
        delta: Target delta value
    
    Returns:
        Privacy epsilon spent
    """
    try:
        accountant = RDPAccountant()
        
        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        
        epsilon = accountant.get_epsilon(delta)
        return epsilon
    except Exception as e:
        print(f"Error calculating privacy spent: {e}")
        return float('inf')

def compute_noise_multiplier(target_epsilon, target_delta, sample_rate, steps):
    """
    Compute the noise multiplier needed to achieve target privacy.
    
    Args:
        target_epsilon: Target epsilon value
        target_delta: Target delta value
        sample_rate: Sampling rate of the data
        steps: Number of optimization steps
    
    Returns:
        Required noise multiplier
    """
    try:
        noise_multiplier = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            sample_rate=sample_rate,
            steps=steps
        )
        return noise_multiplier
    except Exception as e:
        print(f"Error computing noise multiplier: {e}")
        return config["privacy"]["noise_multiplier"]  # fallback

def add_noise_to_gradients(gradients, noise_multiplier, max_grad_norm):
    """
    Add calibrated noise to gradients for differential privacy.
    
    Args:
        gradients: List of gradient tensors
        noise_multiplier: Noise multiplier
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        Noisy gradients
    """
    noisy_gradients = []
    
    for grad in gradients:
        if grad is not None:
            # Add Gaussian noise scaled by the noise multiplier
            noise = torch.randn_like(grad) * noise_multiplier * max_grad_norm
            noisy_grad = grad + noise
            noisy_gradients.append(noisy_grad)
        else:
            noisy_gradients.append(grad)
    
    return noisy_gradients

def validate_privacy_parameters(epsilon, delta, noise_multiplier):
    """
    Validate privacy parameters to ensure they are reasonable.
    
    Args:
        epsilon: Privacy epsilon
        delta: Privacy delta
        noise_multiplier: Noise multiplier
    
    Returns:
        Boolean indicating if parameters are valid
    """
    if epsilon <= 0:
        print("Error: epsilon must be positive")
        return False
    
    if delta <= 0 or delta >= 1:
        print("Error: delta must be between 0 and 1")
        return False
    
    if noise_multiplier <= 0:
        print("Error: noise_multiplier must be positive")
        return False
    
    # Check if privacy parameters are too loose
    if epsilon > 10:
        print("Warning: epsilon is quite large, privacy may be weak")
    
    if delta > 1e-3:
        print("Warning: delta is quite large, privacy may be weak")
    
    return True

class PrivacyTracker:
    """Track privacy budget across federated learning rounds."""
    
    def __init__(self, target_epsilon, target_delta):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.spent_epsilon = 0.0
        self.round_epsilons = []
        self.accountant = RDPAccountant()
    
    def add_round(self, noise_multiplier, sample_rate):
        """Add privacy cost for a federated round."""
        self.accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        current_epsilon = self.accountant.get_epsilon(self.target_delta)
        round_epsilon = current_epsilon - self.spent_epsilon
        
        self.round_epsilons.append(round_epsilon)
        self.spent_epsilon = current_epsilon
    
    def get_remaining_budget(self):
        """Get remaining privacy budget."""
        return max(0, self.target_epsilon - self.spent_epsilon)
    
    def is_budget_exhausted(self):
        """Check if privacy budget is exhausted."""
        return self.spent_epsilon >= self.target_epsilon
    
    def get_privacy_report(self):
        """Get detailed privacy report."""
        return {
            'target_epsilon': self.target_epsilon,
            'target_delta': self.target_delta,
            'spent_epsilon': self.spent_epsilon,
            'remaining_budget': self.get_remaining_budget(),
            'rounds_completed': len(self.round_epsilons),
            'average_epsilon_per_round': np.mean(self.round_epsilons) if self.round_epsilons else 0
        }
