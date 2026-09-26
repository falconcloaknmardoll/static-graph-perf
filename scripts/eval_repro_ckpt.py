"""Score repro checkpoints with train.py's own evaluate() on nlp:default valid/."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import TPUGraphDataset
from model_phase3 import HardwareAwareDiffPool
from train import evaluate
dev = torch.device("cuda")
val = TPUGraphDataset(data_dir="data/predict-ai-model-runtime/npz_all/npz/layout/nlp/default/valid")
for p in sys.argv[1:]:
    ck = torch.load(p, map_location="cpu")
    m = HardwareAwareDiffPool(in_channels=158, hidden_channels=128, dropout=0.0).to(dev)
    m.load_state_dict(ck["model_state"]); t = time.time()
    with torch.no_grad(): r = evaluate(m, val, dev, 8)
    print(f"{p}  step={ck['step']}  val_tau={r['val_tau']:+.4f}  n_graphs={r['val_n_graphs']}  ({time.time()-t:.0f}s)", flush=True)
