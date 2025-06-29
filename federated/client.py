import socket, struct, torch, io, time, random, logging
from .privacy import DifferentialPrivacyManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class FederatedClient:
    def __init__(self, config, client_id, model, train_loader):
        self.config = config
        self.client_id = int(client_id)
        self.model = model
        self.train_loader = train_loader
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.addr = (config['federated']['server_address'], int(config['federated']['server_port']))
        self.priv = DifferentialPrivacyManager(config)
        self.round = -1

    def connect(self):
        for i in range(5):  # 5 attempts with exponential backoff
            try:
                self.sock = socket.socket()
                self.sock.settimeout(5)
                self.sock.connect(self.addr)
                self.sock.send(b'R')
                self.sock.send(struct.pack('!I', self.client_id))
                if self.sock.recv(2) == b'OK':
                    logging.info(f"Client {self.client_id} connected")
                    return True
            except Exception as e:
                delay = (2 ** i) + random.uniform(0.5, 2.0)
                logging.warning(f"Client {self.client_id} connection attempt {i+1} failed: {str(e)}")
                time.sleep(delay)
        return False

    def run(self):
        if not self.connect(): return
        while True:
            try:
                # get status
                self.sock.send(b'S')
                l=struct.unpack('!I',self.sock.recv(4))[0]; data=b''
                while len(data)<l: data+=self.sock.recv(l-len(data))
                st=torch.load(io.BytesIO(data))
                if not st['server_running']: break
                if st['current_round']!=self.round:
                    self.round=st['current_round']
                    # get model
                    self.sock.send(b'G')
                    l=struct.unpack('!I',self.sock.recv(4))[0]; buf=b''
                    while len(buf)<l: buf+=self.sock.recv(l-len(buf))
                    sd=torch.load(io.BytesIO(buf),map_location=self.device)
                    self.model.load_state_dict(sd)
                    # local train
                    self.model.train()
                    opt=torch.optim.Adam(self.model.parameters(),lr=self.config['federated']['learning_rate'])
                    crit=torch.nn.CrossEntropyLoss()
                    for epoch in range(self.config['federated']['local_epochs']):
                        for i,(x,y) in enumerate(self.train_loader):
                            opt.zero_grad(); out=self.model(x); loss=crit(out,y); loss.backward()
                            final = (epoch==self.config['federated']['local_epochs']-1 and i==len(self.train_loader)-1)
                            self.priv.apply(self.model, self.round, 1, final)
                            opt.step()
                    # send update
                    buf=io.BytesIO(); torch.save(self.model.state_dict(),buf); b=buf.getvalue()
                    self.sock.send(b'U'); self.sock.send(struct.pack('!I',self.client_id))
                    self.sock.send(struct.pack('!I',self.round))
                    self.sock.send(struct.pack('!I',len(b))); self.sock.sendall(b)
                    self.sock.recv(3)
                time.sleep(random.uniform(1,3))
            except:
                break
        self.sock.close()
        logging.info(f"Client {self.client_id} done")
