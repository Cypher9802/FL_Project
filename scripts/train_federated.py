#!/usr/bin/env python3
import sys, time, signal, logging, random
from pathlib import Path
import torch, torch.multiprocessing as mp, yaml

sys.path.append(str(Path(__file__).parent.parent))
from data.data_loader import UCIHARDataLoader
from models.neural_network import FeedForwardNN
from federated.server import FederatedServer
from federated.client import FederatedClient
from models.mobile_optimized import MobileOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_config():
    return yaml.safe_load(open("config/config.yaml"))

def start_server(cfg, gm):
    sv = FederatedServer(cfg, gm)
    sv.start()
    sv.save_privacy_metrics()

def run_client(cfg, cid, ldr):
    mp.set_sharing_strategy('file_system')
    model_cfg = dict(cfg['model'])
    pretrain = model_cfg.pop('pretrain_path', None)
    model = FeedForwardNN(
        input_size=model_cfg['input_size'],
        hidden_layers=model_cfg['hidden_layers'],
        num_classes=model_cfg['num_classes'],
        dropout_rate=model_cfg['dropout_rate']
    )
    if pretrain:
        pt = Path(pretrain)
        if pt.exists():
            state = torch.load(pt, map_location='cpu')
            model.load_state_dict(state, strict=False)
            logging.info(f"Client {cid}: loaded pretrained weights from {pretrain}")
        else:
            logging.warning(f"Client {cid}: pretrain_path {pretrain} not found")
    client = FederatedClient(cfg, cid, model, ldr)
    client.run()

def signal_handler(sig, frame):
    sys.exit(0)

def main():
    mp.set_start_method('spawn')
    signal.signal(signal.SIGINT, signal_handler)
    cfg = load_config()
    logging.info("🔒 Starting DP-FL with Decay Schedules")
    dl = UCIHARDataLoader(cfg)
    cls, _ = dl.get_data_loaders()
    model_cfg = dict(cfg['model'])
    pretrain = model_cfg.pop('pretrain_path', None)
    gm = FeedForwardNN(
        input_size=model_cfg['input_size'],
        hidden_layers=model_cfg['hidden_layers'],
        num_classes=model_cfg['num_classes'],
        dropout_rate=model_cfg['dropout_rate']
    )
    if pretrain:
        pt = Path(pretrain)
        if pt.exists():
            state = torch.load(pt, map_location='cpu')
            gm.load_state_dict(state, strict=False)
            logging.info(f"Loaded pretrained global weights from {pretrain}")
        else:
            logging.warning(f"Global pretrain_path {pretrain} not found")
    logging.info(f"Global model size: {gm.get_model_size():.2f}MB")
    sp = mp.Process(target=start_server, args=(cfg, gm), daemon=True)
    sp.start()
    time.sleep(3)
    procs = []
    for cid, ldr in cls.items():
        p = mp.Process(target=run_client, args=(cfg, cid, ldr), daemon=True)
        p.start()
        procs.append(p)
        # Increased and randomized delay to avoid connection storm
        time.sleep(float(cfg['federated']['client_start_delay']) + random.uniform(1.0, 3.0))
        logging.info(f"Launched client {cid}")
    timeout = cfg['federated']['num_rounds'] * (cfg['federated']['round_timeout'] + 5)
    st = time.time()
    while time.time() - st < timeout and sp.is_alive():
        time.sleep(10)
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(5)
    if sp.is_alive():
        sp.terminate()
        sp.join(5)
    logging.info("🔒 Training complete; optimizing mobile model")
    mo = MobileOptimizer(cfg)
    opt = mo.optimize_for_mobile(gm, torch.randn(1, cfg['model']['input_size']))
    mo.save_mobile_model(opt, "models/saved/mobile_optimized_model.pt")
    torch.save(gm.state_dict(), "models/saved/federated_model.pth")

if __name__ == "__main__":
    main()
