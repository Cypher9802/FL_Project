#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import csv
import itertools
import subprocess
import time
import os
import json
import numpy as np
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_base_config():
    """Load base configuration"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    return yaml.safe_load(open(config_path))

def create_hyperparameter_grid():
    """Define hyperparameter search space with privacy considerations"""
    return {
        'learning_rates': [0.0005, 0.001, 0.002],
        'local_epochs': [1, 2, 3],
        'hidden_layers': [[64, 32], [128, 64], [256, 128]],
        'noise_multipliers': [0.8, 1.2, 1.6],  # For differential privacy
        'max_grad_norms': [0.5, 1.0, 1.5],     # For gradient clipping
        'augmentation_noise': [0.02, 0.05, 0.08]  # For data augmentation
    }

def generate_privacy_aware_config(base_config: Dict, params: Dict) -> Dict:
    """Generate configuration with privacy and augmentation parameters"""
    config = dict(base_config)
    
    # Update federated learning parameters
    config['federated']['learning_rate'] = params['learning_rate']
    config['federated']['local_epochs'] = params['local_epochs']
    config['federated']['num_rounds'] = 8  # Reduced for faster tuning
    
    # Update model architecture
    config['model']['hidden_layers'] = params['hidden_layers']
    
    # Update privacy parameters
    config['privacy']['enable_differential_privacy'] = True
    config['privacy']['noise_multiplier'] = params['noise_multiplier']
    config['privacy']['max_grad_norm'] = params['max_grad_norm']
    
    # Update augmentation parameters
    config['dataset']['enable_augmentation'] = True
    config['dataset']['augmentation']['noise_std'] = params['augmentation_noise']
    
    return config

def run_training_trial(config: Dict, trial_id: int) -> Tuple[float, Dict]:
    """Run a single training trial and return accuracy with privacy metrics"""
    temp_config_path = Path(f"config/temp_hparam_trial_{trial_id}.yaml")
    temp_config_path.parent.mkdir(exist_ok=True)
    
    try:
        # Save temporary config
        with open(temp_config_path, 'w') as f:
            yaml.safe_dump(config, f)
        
        # Set environment variable for config path
        env = {**os.environ, "CONFIG_PATH": str(temp_config_path)}
        
        # Run training
        logging.info(f"Starting trial {trial_id} with privacy-enhanced training")
        start_time = time.time()
        
        training_result = subprocess.run([
            "python3", "scripts/train_federated.py"
        ], env=env, capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if training_result.returncode != 0:
            logging.error(f"Training failed for trial {trial_id}: {training_result.stderr}")
            return 0.0, {}
        
        # Run evaluation
        eval_result = subprocess.run([
            "python3", "scripts/evaluate_model.py"
        ], capture_output=True, text=True, timeout=120)  # 2 min timeout
        
        if eval_result.returncode != 0:
            logging.error(f"Evaluation failed for trial {trial_id}: {eval_result.stderr}")
            return 0.0, {}
        
        # Parse accuracy from evaluation output
        accuracy = 0.0
        for line in eval_result.stdout.splitlines():
            if "Test Accuracy:" in line:
                accuracy = float(line.split()[-1])
                break
        
        # Load privacy metrics if available
        privacy_metrics = {}
        privacy_file = Path("server_privacy_analysis.json")
        if privacy_file.exists():
            privacy_metrics = json.loads(privacy_file.read_text())
        
        training_time = time.time() - start_time
        
        return accuracy, {
            'privacy_epsilon_used': privacy_metrics.get('spent_privacy', {}).get('epsilon', 0),
            'privacy_budget_remaining': privacy_metrics.get('remaining_budget', {}).get('epsilon', 0),
            'training_time': training_time
        }
        
    except subprocess.TimeoutExpired:
        logging.warning(f"Trial {trial_id} timed out")
        return 0.0, {}
    except Exception as e:
        logging.error(f"Trial {trial_id} failed with error: {e}")
        return 0.0, {}
    finally:
        # Cleanup temporary config
        if temp_config_path.exists():
            temp_config_path.unlink()

def privacy_aware_hyperparameter_tuning():
    """Main hyperparameter tuning with privacy analysis"""
    base_config = load_base_config()
    param_grid = create_hyperparameter_grid()
    
    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        param_grid['learning_rates'],
        param_grid['local_epochs'],
        param_grid['hidden_layers'],
        param_grid['noise_multipliers'],
        param_grid['max_grad_norms'],
        param_grid['augmentation_noise']
    ))
    
    # Limit to manageable number of trials
    max_trials = min(20, len(param_combinations))
    if len(param_combinations) > max_trials:
        # Random sampling for efficiency
        import random
        random.seed(42)
        param_combinations = random.sample(param_combinations, max_trials)
    
    logging.info(f"Starting privacy-aware hyperparameter tuning with {len(param_combinations)} trials")
    
    # Results storage
    results_file = Path("privacy_aware_hparam_results.csv")
    detailed_results = []
    
    # CSV header
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_id", "learning_rate", "local_epochs", "hidden_layers",
            "noise_multiplier", "max_grad_norm", "augmentation_noise",
            "accuracy", "privacy_epsilon_used", "privacy_budget_remaining", 
            "training_time", "privacy_utility_ratio"
        ])
    
    best_accuracy = 0.0
    best_params = None
    best_privacy_utility = 0.0
    
    # Run trials
    for trial_id, params in enumerate(param_combinations, 1):
        lr, epochs, layers, noise_mult, grad_norm, aug_noise = params
        
        trial_params = {
            'learning_rate': lr,
            'local_epochs': epochs,
            'hidden_layers': layers,
            'noise_multiplier': noise_mult,
            'max_grad_norm': grad_norm,
            'augmentation_noise': aug_noise
        }
        
        logging.info(f"Trial {trial_id}/{len(param_combinations)}: {trial_params}")
        
        # Generate config and run trial
        trial_config = generate_privacy_aware_config(base_config, trial_params)
        accuracy, privacy_info = run_training_trial(trial_config, trial_id)
        
        # Calculate privacy-utility ratio (higher is better)
        epsilon_used = privacy_info.get('privacy_epsilon_used', 0)
        privacy_utility_ratio = accuracy / (epsilon_used + 0.001) if epsilon_used > 0 else accuracy
        
        # Record results
        result_row = [
            trial_id, lr, epochs, str(layers), noise_mult, grad_norm, aug_noise,
            accuracy, epsilon_used, privacy_info.get('privacy_budget_remaining', 0),
            privacy_info.get('training_time', 0), privacy_utility_ratio
        ]
        
        detailed_results.append({
            'trial_id': trial_id,
            'params': trial_params,
            'accuracy': accuracy,
            'privacy_info': privacy_info,
            'privacy_utility_ratio': privacy_utility_ratio
        })
        
        # Append to CSV
        with open(results_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(result_row)
        
        # Track best results
        if privacy_utility_ratio > best_privacy_utility:
            best_privacy_utility = privacy_utility_ratio
            best_params = trial_params
            best_accuracy = accuracy
        
        logging.info(f"Trial {trial_id} completed: Accuracy={accuracy:.4f}, "
                    f"ε_used={epsilon_used:.3f}, Privacy-Utility={privacy_utility_ratio:.4f}")
        
        # Brief pause between trials
        time.sleep(2)
    
    # Generate comprehensive analysis
    analysis_report = {
        'best_configuration': {
            'parameters': best_params,
            'accuracy': best_accuracy,
            'privacy_utility_ratio': best_privacy_utility
        },
        'trial_summary': {
            'total_trials': len(detailed_results),
            'avg_accuracy': np.mean([r['accuracy'] for r in detailed_results]),
            'std_accuracy': np.std([r['accuracy'] for r in detailed_results]),
            'avg_privacy_cost': np.mean([r['privacy_info'].get('privacy_epsilon_used', 0) for r in detailed_results])
        },
        'top_5_configurations': sorted(
            detailed_results, 
            key=lambda x: x['privacy_utility_ratio'], 
            reverse=True
        )[:5]
    }
    
    # Save detailed analysis
    analysis_path = Path("hparam_privacy_analysis.json")
    analysis_path.write_text(json.dumps(analysis_report, indent=2))
    
    print(f"\n=== Privacy-Aware Hyperparameter Tuning Complete ===")
    print(f"Best Configuration:")
    print(f"  Parameters: {best_params}")
    print(f"  Accuracy: {best_accuracy:.4f}")
    print(f"  Privacy-Utility Ratio: {best_privacy_utility:.4f}")
    print(f"\nResults saved to:")
    print(f"  CSV: {results_file}")
    print(f"  Analysis: {analysis_path}")
    
    return best_params, analysis_report

if __name__ == "__main__":
    privacy_aware_hyperparameter_tuning()
