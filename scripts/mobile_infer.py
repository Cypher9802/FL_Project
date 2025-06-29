#!/usr/bin/env python3
import sys,time,json,yaml,logging
from pathlib import Path
import torch, numpy as np, matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from models.neural_network import FeedForwardNN

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
torch.backends.quantized.engine="qnnpack"

def load_model():
    mp=Path("models/saved/mobile_optimized_model.pt")
    try:
        m=torch.jit.load(str(mp)); logging.info("TorchScript loaded"); return m
    except:
        cfg=yaml.safe_load(open("config/config.yaml"))
        m=FeedForwardNN(**cfg['model'])
        m.load_state_dict(torch.load(str(mp),map_location='cpu'))
        logging.info("State dict loaded"); return m

def main():
    m=load_model(); m.eval()
    # warm-up
    for _ in range(5): _=m(torch.randn(1,561))
    times=[]
    for _ in range(200):
        inp=torch.randn(1,561)
        s=time.time(); _=m(inp); times.append(time.time()-s)
    ms=np.array(times)*1000
    logging.info(f"Latency: {ms.mean():.2f}±{ms.std():.2f}ms")
    plt.hist(ms,bins=30); plt.title("Latency Distribution"); plt.xlabel("ms")
    plt.savefig("mobile_latency.png",dpi=300); plt.show()
    
    priv=Path("server_privacy_analysis.json")
    if priv.exists():
        pd=json.loads(priv.read_text())
        print("Privacy spent ε=",pd['spent']['ε'])

if __name__=="__main__":
    main()
