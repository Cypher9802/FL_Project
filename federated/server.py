import socket, struct, threading, torch, io, time, logging, json
from pathlib import Path
from .aggregation import federated_average
from .privacy import DifferentialPrivacyManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class FederatedServer:
    def __init__(self, config, global_model):
        self.config = config
        self.global_model = global_model.to('cpu')
        self.client_updates = {}
        self.active_clients = set()
        self.lock = threading.Lock()
        self.current_round = 0
        self.server_running = True
        self.priv = DifferentialPrivacyManager(config)
        self.metrics_path = Path("training_metrics.json")
        self.privacy_path = Path("server_privacy_analysis.json")
        self.round_metrics = []

    def save_privacy_metrics(self):
        if hasattr(self.priv, 'accountant'):
            analysis = self.priv.accountant.report()
            self.privacy_path.write_text(json.dumps(analysis, indent=2))

    def save_training_metrics(self, round_idx, participants, avg_loss):
        metrics = []
        if self.metrics_path.exists():
            metrics = json.loads(self.metrics_path.read_text())
        metrics.append({
            'round': round_idx,
            'participants': participants,
            'average_loss': avg_loss
        })
        self.metrics_path.write_text(json.dumps(metrics, indent=2))

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.config['federated']['server_address'],
                  self.config['federated']['server_port']))
        sock.listen(128)  # Increased backlog
        logging.info(f"Server bound to {self.config['federated']['server_address']}:{self.config['federated']['server_port']}")
        logging.info("Listening with backlog size: 128")
        threading.Thread(target=self.run_federated_learning, daemon=True).start()

        while self.server_running:
            try:
                client_sock, _ = sock.accept()
                threading.Thread(target=self.handle_client,
                                 args=(client_sock,), daemon=True).start()
            except:
                break
        sock.close()
        logging.info("Server shut down")

    def handle_client(self, cs):
        cid = None
        try:
            cs.settimeout(30)
            while self.server_running:
                msg = cs.recv(1)
                if not msg: break
                if msg == b'R':
                    cid = self.register(cs)
                elif msg == b'G':
                    self.send_model(cs)
                elif msg == b'U':
                    self.receive_update(cs)
                elif msg == b'S':
                    self.send_status(cs)
                elif msg == b'H':
                    self.handle_heartbeat(cs)
        except:
            pass
        finally:
            if cid is not None:
                with self.lock:
                    self.active_clients.discard(cid)
            try: cs.close()
            except: pass

    # ... (register, handle_heartbeat, send_model, send_status, receive_update as before) ...

    def run_federated_learning(self):
        logging.info("Starting FL process")
        for r in range(self.config['federated']['num_rounds']):
            self.current_round = r
            logging.info(f"Round {r+1}/{self.config['federated']['num_rounds']}")
            start = time.time()
            minp = min(3, self.config['federated']['clients_per_round'])
            while time.time() - start < self.config['federated']['round_timeout']:
                with self.lock:
                    if len(self.client_updates) >= minp: break
                time.sleep(1)
            with self.lock:
                if self.client_updates:
                    ag = federated_average(list(self.client_updates.values()))
                    self.global_model.load_state_dict(ag)
                    avg_loss = 0.0  # You can calculate this if you pass losses from clients
                    self.save_training_metrics(r+1, len(self.client_updates), avg_loss)
                    self.save_privacy_metrics()
                    self.client_updates.clear()
                else:
                    logging.warning(f"No updates in round {r+1}")
            time.sleep(2)
        self.server_running = False
        self.save_privacy_metrics()
        logging.info("FL process complete")
