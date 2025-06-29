#!/usr/bin/env python3
import sys
from pathlib import Path
import logging

# Ensure project root is on PYTHONPATH so we can import data.data_loader
sys.path.append(str(Path(__file__).parent.parent))

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.data_loader import UCIHARDataset, UCIHARDataLoader
from models.neural_network import FeedForwardNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load and preprocess data centrally
    data_loader = UCIHARDataLoader(config)
    client_loaders, test_loader = data_loader.get_data_loaders()

    # Combine all client training data into one DataLoader
    all_datasets = [loader.dataset for loader in client_loaders.values()]
    unified_dataset = torch.utils.data.ConcatDataset(all_datasets)
    train_loader = DataLoader(
        unified_dataset,
        batch_size=int(config['federated']['batch_size']),
        shuffle=True
    )

    # Initialize model
    model = FeedForwardNN(
        input_size=config['model']['input_size'],
        hidden_layers=config['model']['hidden_layers'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate']
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['federated']['learning_rate'])
    epochs = 10

    logging.info(f"Starting centralized pre-training for {epochs} epochs")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        logging.info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    # Save pre-trained weights
    pretrain_path = Path(config['model']['pretrain_path'])
    pretrain_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pretrain_path)
    logging.info(f"Saved pre-trained model to {pretrain_path}")

if __name__ == "__main__":
    main()
