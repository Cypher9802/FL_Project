#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import json
import random
import time
import torch
import torch.multiprocessing as mp
import numpy as np
import logging
from sklearn.metrics import accuracy_score
from torch.utils.data import ConcatDataset, DataLoader

from data.data_loader import UCIHARDataLoader
from models.neural_network import FeedForwardNN
from federated.server import FederatedServer
from federated.client import FederatedClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_config():
    return yaml.safe_load(open(Path(__file__).parent.parent / "config" / "config.yaml"))

def start_server_process(config):
    """Subprocess target: instantiate and start the FederatedServer with privacy"""
    import torch
    global_model = FeedForwardNN(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    )
    server = FederatedServer(config, global_model)
    server.start()

def run_client_process(client_id, config):
    """Subprocess target: run one privacy-enhanced client"""
    import torch
    mp.set_sharing_strategy('file_system')
    
    # Build data loaders
    dl = UCIHARDataLoader(config)
    client_loaders, _ = dl.get_data_loaders()
    loader = client_loaders[client_id]
    
    # Instantiate model
    model = FeedForwardNN(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    )
    client = FederatedClient(client_id, config, model, loader)
    client.run()

def run_privacy_enhanced_fold(train_clients, test_clients, config, fold_idx):
    """Run one cross-validation fold with privacy enhancement"""
    import torch
    from torch.utils.data import ConcatDataset, DataLoader
    
    logging.info(f"Fold {fold_idx}: Starting privacy-enhanced training with {len(train_clients)} clients")
    
    # Start server subprocess
    server_proc = mp.Process(
        target=start_server_process, 
        args=(config,), 
        daemon=True
    )
    server_proc.start()
    time.sleep(3)

    # Launch training clients
    client_procs = []
    for cid in train_clients:
        p = mp.Process(
            target=run_client_process,
            args=(cid, config),
            daemon=True
        )
        p.start()
        client_procs.append(p)
        time.sleep(0.2)

    # Wait for training to complete
    total_time = config['federated']['num_rounds'] * (config['federated']['round_timeout'] + 2)
    time.sleep(total_time)

    # Terminate processes
    for p in client_procs:
        if p.is_alive():
            p.terminate()
            p.join(5)
    if server_proc.is_alive():
        server_proc.terminate()
        server_proc.join(5)

    # Load privacy metrics for this fold
    fold_privacy_metrics = {}
    privacy_file = Path("server_privacy_analysis.json")
    if privacy_file.exists():
        fold_privacy_metrics = json.loads(privacy_file.read_text())
        # Backup fold-specific privacy metrics
        fold_privacy_file = Path(f"fold_{fold_idx}_privacy_analysis.json")
        fold_privacy_file.write_text(json.dumps(fold_privacy_metrics, indent=2))

    # Build test loader from test clients
    dl = UCIHARDataLoader(config)
    client_loaders, _ = dl.get_data_loaders()
    test_datasets = [client_loaders[cid].dataset for cid in test_clients]
    test_dataset = ConcatDataset(test_datasets)
    test_loader = DataLoader(test_dataset, batch_size=config['federated']['batch_size'])

    # Load and evaluate final model
    model = FeedForwardNN(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    model_path = Path("models/saved/federated_model.pth")
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Evaluate on test loader
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            preds = out.argmax(dim=1)
            y_pred.extend(preds.numpy().tolist())
            y_true.extend(y.numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    
    # Return both accuracy and privacy metrics
    return accuracy, fold_privacy_metrics

def privacy_enhanced_cross_validation():
    """Main cross-validation with comprehensive privacy analysis"""
    mp.set_start_method('spawn', force=True)
    config = load_config()
    
    # Ensure privacy and augmentation are enabled
    config['privacy']['enable_differential_privacy'] = True
    config['dataset']['enable_augmentation'] = True
    
    logging.info("🔒 Starting Privacy-Enhanced Cross-Validation")
    logging.info(f"Privacy Settings: ε={config['privacy']['target_epsilon']}, "
                f"noise_multiplier={config['privacy']['noise_multiplier']}")
    
    num_clients = config['federated']['num_clients']
    clients = list(range(num_clients))
    random.seed(config['dataset']['random_seed'])
    random.shuffle(clients)

    k = 5  # number of folds
    folds = [clients[i::k] for i in range(k)]
    results = {}
    privacy_analysis = {}

    for i in range(k):
        test_fold = folds[i]
        train_fold = [c for c in clients if c not in test_fold]
        
        logging.info(f"Fold {i+1}: Train={len(train_fold)} clients, Test={len(test_fold)} clients")
        
        acc, privacy_metrics = run_privacy_enhanced_fold(train_fold, test_fold, config, i+1)
        
        results[f"fold_{i+1}"] = acc
        privacy_analysis[f"fold_{i+1}"] = privacy_metrics
        
        logging.info(f"Fold {i+1} Accuracy: {acc:.4f}")
        if privacy_metrics.get('spent_privacy'):
            epsilon_used = privacy_metrics['spent_privacy']['epsilon']
            logging.info(f"Fold {i+1} Privacy Cost: ε={epsilon_used:.3f}")

    # Comprehensive cross-validation analysis
    accuracies = list(results.values())
    cv_analysis = {
        'cross_validation_results': results,
        'privacy_analysis_per_fold': privacy_analysis,
        'statistical_summary': {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
            '95_confidence_interval': [
                float(np.mean(accuracies) - 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))),
                float(np.mean(accuracies) + 1.96 * np.std(accuracies) / np.sqrt(len(accuracies)))
            ]
        },
        'privacy_summary': {
            'average_epsilon_per_fold': float(np.mean([
                p.get('spent_privacy', {}).get('epsilon', 0) 
                for p in privacy_analysis.values()
            ])),
            'total_privacy_budget_target': config['privacy']['target_epsilon'],
            'privacy_mechanism': config['privacy']['dp_mechanism']
        },
        'experimental_setup': {
            'num_folds': k,
            'clients_per_fold_train': len(train_fold),
            'clients_per_fold_test': len(test_fold),
            'differential_privacy_enabled': config['privacy']['enable_differential_privacy'],
            'data_augmentation_enabled': config['dataset']['enable_augmentation'],
            'secure_aggregation_enabled': config['privacy']['enable_secure_aggregation']
        }
    }

    # Save comprehensive results
    results_path = Path("privacy_enhanced_cv_results.json")
    results_path.write_text(json.dumps(cv_analysis, indent=2))
    
    # Generate privacy-utility analysis
    privacy_utility_analysis = {
        'privacy_cost_vs_accuracy': [
            {
                'fold': f"fold_{i+1}",
                'accuracy': results[f"fold_{i+1}"],
                'privacy_cost': privacy_analysis[f"fold_{i+1}"].get('spent_privacy', {}).get('epsilon', 0),
                'privacy_utility_ratio': results[f"fold_{i+1}"] / max(0.001, privacy_analysis[f"fold_{i+1}"].get('spent_privacy', {}).get('epsilon', 0.001))
            }
            for i in range(k)
        ]
    }

    print(f"\n=== Privacy-Enhanced Cross-Validation Results ===")
    print(f"Mean Accuracy: {cv_analysis['statistical_summary']['mean_accuracy']:.4f} ± {cv_analysis['statistical_summary']['std_accuracy']:.4f}")
    print(f"95% Confidence Interval: [{cv_analysis['statistical_summary']['95_confidence_interval'][0]:.4f}, {cv_analysis['statistical_summary']['95_confidence_interval'][1]:.4f}]")
    print(f"Average Privacy Cost: ε = {cv_analysis['privacy_summary']['average_epsilon_per_fold']:.3f}")
    print(f"Privacy Budget Target: ε = {cv_analysis['privacy_summary']['total_privacy_budget_target']:.3f}")
    
    # Save fold-by-fold results for backward compatibility
    simple_results = {f"fold_{i+1}": results[f"fold_{i+1}"] for i in range(k)}
    Path("cv_results.json").write_text(json.dumps(simple_results, indent=2))
    
    logging.info("🔒✅ Privacy-Enhanced Cross-Validation completed successfully!")
    logging.info(f"Detailed results saved to: {results_path}")
    
    return cv_analysis

if __name__ == "__main__":
    privacy_enhanced_cross_validation()
