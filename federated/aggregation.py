import torch
from collections import OrderedDict

def secure_agg(global_model, client_states, noise_scale=0.1, use_secure=True):
    """
    Secure aggregation: FedAvg + noise injection.
    client_states: list of state_dicts from client updates
    """
    global_dict = global_model.state_dict()
    agg_dict = OrderedDict()
    n = len(client_states)
    for k in global_dict:
        # average
        stacked = torch.stack([cs[k] for cs in client_states], dim=0)
        avg = torch.mean(stacked, dim=0)
        # secure noise
        if use_secure:
            noise = torch.randn_like(avg) * noise_scale
            avg = avg + noise
        agg_dict[k] = avg
    return agg_dict
