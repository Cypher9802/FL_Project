# federated/privacy.py

import torch
import numpy as np
import math
import logging
import json
from pathlib import Path
from typing import Dict, Tuple

class PrivacyAccountant:
    """Tracks cumulative ε and δ over federated learning rounds."""
    
    def __init__(self, target_epsilon: float, target_delta: float):
        self.target_epsilon = float(target_epsilon)
        self.target_delta = float(target_delta)
        self.spent_epsilon = 0.0
        self.round_costs = []

    def compute_gaussian_cost(self, noise_multiplier: float, num_clients: int, sampling_probability: float) -> Tuple[float, float]:
        """
        Simplified RDP-based cost for Gaussian mechanism per round.
        noise_multiplier: standard deviation multiplier
        num_clients: number of clients participating
        sampling_probability: fraction of total clients
        """
        noise_multiplier = float(noise_multiplier)
        sampling_probability = float(sampling_probability)
        
        if noise_multiplier == 0:
            return float('inf'), 0.0
        
        # RDP cost for one round at order α=2
        rdp = (sampling_probability**2) / (2 * noise_multiplier**2)
        # Convert to (ε, δ)-DP using α=2
        epsilon = rdp + math.log(1 / self.target_delta)  # since α-1 = 1
        return float(epsilon), float(self.target_delta)

    def update(self, round_epsilon: float, round_delta: float, round_idx: int):
        """Record and accumulate privacy cost for a completed round."""
        round_epsilon = float(round_epsilon)
        round_delta = float(round_delta)
        round_idx = int(round_idx)
        
        self.spent_epsilon += round_epsilon
        self.round_costs.append({
            'round': round_idx,
            'round_epsilon': round_epsilon,
            'round_delta': round_delta,
            'cumulative_epsilon': self.spent_epsilon,
            'cumulative_delta': self.spent_delta
        })
        logging.info(f"PrivacyAccountant: Round {round_idx} ε={round_epsilon:.4f}, cumulative ε={self.spent_epsilon:.4f}")

    def is_exhausted(self) -> bool:
        """Return True if spent ε exceeds target."""
        return self.spent_epsilon >= self.target_epsilon

    def report(self) -> Dict:
        """Return a dict summarizing privacy usage."""
        return {
            'target_epsilon': self.target_epsilon,
            'target_delta': self.target_delta,
            'spent_epsilon': self.spent_epsilon,
            'round_by_round': self.round_costs,
            'budget_exhausted': self.is_exhausted()
        }

class DifferentialPrivacyManager:
    """
    Manages gradient clipping, noise addition, and privacy accounting
    once per federated communication round.
    """
    
    def __init__(self, config: Dict):
        # Save full config for access to federated parameters
        self.config = config
        p = config['privacy']
        
        self.noise_initial = float(p.get('noise_multiplier_initial', p.get('noise_multiplier', 0.5)))
        self.noise_final = float(p.get('noise_multiplier_final', self.noise_initial))
        self.clip_initial = float(p.get('max_grad_norm_initial', p.get('max_grad_norm', 1.0)))
        self.clip_final = float(p.get('max_grad_norm_final', self.clip_initial))
        
        self.target_epsilon = float(p['target_epsilon'])
        self.target_delta = float(p['target_delta'])
        self.accountant = PrivacyAccountant(self.target_epsilon, self.target_delta)
        
        self.total_rounds = int(config['federated']['num_rounds'])
        self.counted_rounds = set()

    def _interpolate(self, start: float, end: float, rnd: int) -> float:
        """Linear interpolation from start to end over rounds."""
        t = rnd / max(1, self.total_rounds - 1)
        return start * (1 - t) + end * t

    def apply(self, model: torch.nn.Module, round_idx: int, num_clients: int, final_batch: bool) -> Dict:
        """
        Clip gradients, add noise, and update privacy accountant once per round.
        
        model: the local model after backward()
        round_idx: zero-based index of current FL round
        num_clients: number of participating clients this round
        final_batch: True if this is the last local batch of the last epoch
        """
        # Determine current clipping threshold and noise multiplier
        clip_thresh = self._interpolate(self.clip_initial, self.clip_final, round_idx)
        noise_mul = self._interpolate(self.noise_initial, self.noise_final, round_idx)

        # Clip gradients
        total_norm = torch.norm(
            torch.stack([p.grad.norm(2) for p in model.parameters() if p.grad is not None]),
            2
        ).item()
        clip_coef = clip_thresh / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(clip_coef)

        # Add Gaussian noise
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.normal(0, noise_mul * clip_thresh, p.grad.shape, device=p.grad.device)
                p.grad.add_(noise)

        # Account privacy cost once per round
        privacy_info = {'round_epsilon': 0.0, 'cumulative_epsilon': self.accountant.spent_epsilon}
        if final_batch and round_idx not in self.counted_rounds:
            # Sampling probability q = clients / total_clients
            total_clients = float(self.config['federated']['num_clients'])
            q = float(num_clients) / total_clients
            eps, delta = self.accountant.compute_gaussian_cost(noise_mul, num_clients, q)
            self.accountant.update(eps, delta, round_idx + 1)
            self.counted_rounds.add(round_idx)
            privacy_info = {
                'round_epsilon': eps,
                'cumulative_epsilon': self.accountant.spent_epsilon
            }

        return privacy_info

    def save_analysis(self, filepath: str = "server_privacy_analysis.json"):
        """Write the accountant’s report to JSON."""
        report = self.accountant.report()
        Path(filepath).write_text(json.dumps(report, indent=2))
        logging.info(f"Saved privacy analysis to {filepath}")
