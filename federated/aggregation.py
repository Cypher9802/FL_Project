import torch
import yaml
from collections import OrderedDict
from pathlib import Path

# Load configuration
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

def secure_aggregate(global_model, client_states, noise_scale=None, use_secure=None):
    """
    Enhanced secure aggregation with configurable noise and privacy preservation.
    
    Args:
        global_model: Current global model
        client_states: List of state_dicts from client updates
        noise_scale: Noise level for privacy (from config if None)
        use_secure: Whether to use secure aggregation (from config if None)
    
    Returns:
        Aggregated state dictionary
    """
    if noise_scale is None:
        noise_scale = config["privacy"]["aggregation_noise_scale"]
    if use_secure is None:
        use_secure = config["privacy"]["use_secure_aggregation"]
    
    if not client_states:
        return global_model.state_dict()
    
    global_dict = global_model.state_dict()
    agg_dict = OrderedDict()
    n_clients = len(client_states)
    
    for key in global_dict:
        # Stack client parameters
        try:
            stacked = torch.stack([cs[key] for cs in client_states], dim=0)
            # FedAvg: weighted average (equal weights for now)
            avg = torch.mean(stacked, dim=0)
            
            # Add differential privacy noise if enabled
            if use_secure and noise_scale > 0:
                noise = torch.randn_like(avg) * noise_scale
                avg = avg + noise
            
            agg_dict[key] = avg
        except Exception as e:
            print(f"Warning: Could not aggregate parameter {key}: {e}")
            agg_dict[key] = global_dict[key]  # Keep original if aggregation fails
    
    return agg_dict

def federated_averaging(client_states, weights=None):
    """
    Standard FedAvg without privacy noise.
    
    Args:
        client_states: List of state_dicts from clients
        weights: List of weights for each client (equal if None)
    
    Returns:
        Averaged state dictionary
    """
    if not client_states:
        return None
    
    if weights is None:
        weights = [1.0 / len(client_states)] * len(client_states)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    agg_dict = OrderedDict()
    
    for key in client_states[0]:
        weighted_sum = None
        for i, state in enumerate(client_states):
            if weighted_sum is None:
                weighted_sum = weights[i] * state[key]
            else:
                weighted_sum += weights[i] * state[key]
        agg_dict[key] = weighted_sum
    
    return agg_dict
