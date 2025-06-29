import torch

def federated_average(client_updates):
    """Federated averaging algorithm"""
    if not client_updates:
        return {}
    
    avg_state_dict = {}
    total_clients = len(client_updates)
    
    # Initialize with first client's parameters
    for key in client_updates[0].keys():
        avg_state_dict[key] = torch.zeros_like(client_updates[0][key])
    
    # Sum all parameters
    for state_dict in client_updates:
        for key, param in state_dict.items():
            if key in avg_state_dict:
                avg_state_dict[key] += param / total_clients
    
    return avg_state_dict
