import torch, os, contextlib, torch.nn as nn, torch.nn.utils.prune as prune

class MobileOptimizer:
    def __init__(self,cfg): self.cfg=cfg

    def prune(self,m):
        if not self.cfg['mobile']['enable_pruning']: return m
        amt=float(self.cfg['mobile']['pruning_amount'])
        for _,mod in m.named_modules():
            if isinstance(mod,nn.Linear): prune.l1_unstructured(mod,'weight',amt)
        for _,mod in m.named_modules():
            if isinstance(mod,nn.Linear): prune.remove(mod,'weight')
        return m

    def quantize(self,m):
        if not self.cfg['mobile']['enable_quantization']: return m
        m=m.to('cpu')
        torch.backends.quantized.engine=self.cfg['mobile']['quantization_backend']
        with open(os.devnull,'w') as f, contextlib.redirect_stderr(f):
            return torch.quantization.quantize_dynamic(m,{nn.Linear},dtype=torch.qint8)

    def optimize_for_mobile(self,m,ex):
        m.eval(); m=self.prune(m); m=self.quantize(m)
        try:
            tm=torch.jit.trace(m,ex.to('cpu'))
            return torch.jit.optimize_for_inference(tm)
        except: return m

    def save_mobile_model(self,m,path):
        os.makedirs(os.path.dirname(path),exist_ok=True)
        if hasattr(m,'save'): m.save(path)
        else: torch.save(m.state_dict(),path)
