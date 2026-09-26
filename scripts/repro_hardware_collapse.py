"""Repro for the nlp:default `hardware` collapse (2026-09-24).

Mirrors train.py's hardware-arm loop exactly (Adam w/ L2 wd, clip 0.1, lp 0.1, ent 0.01,
lr 1e-3, same seeding + sampler) but logs per-step health so we can see WHEN the network
dies.  Health = score_std, fraction of ~0 params, pre-clip grad norm, head-ReLU liveness.
Keeps a rolling last-healthy snapshot; on collapse it saves it + the collapsing batch id
so the failing step can be replayed.  Stops at collapse or --max-steps (default 1 epoch).

  python scripts/repro_hardware_collapse.py --seed 42
"""
from __future__ import annotations
import argparse, copy, os, random, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch_geometric.loader import DataLoader
from dataset import TPUGraphDataset
from loss import listmle_loss
from model_phase3 import HardwareAwareDiffPool
from train import SingleGraphBatchSampler, _forward

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--data-root", default="data/predict-ai-model-runtime/npz_all/npz/layout/nlp/default")
ap.add_argument("--max-steps", type=int, default=3200)
ap.add_argument("--log-every", type=int, default=20)
ap.add_argument("--out", default="logs/repro")
ap.add_argument("--lp-weight", type=float, default=0.1)
ap.add_argument("--entropy-weight", type=float, default=0.01)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--stop-on-collapse", type=int, default=1)
ap.add_argument("--tag", default="")
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
dev = torch.device("cuda")
ds = TPUGraphDataset(data_dir=os.path.join(a.data_root, "train"))
sam = SingleGraphBatchSampler(index=ds._index, configs_per_batch=8, shuffle=True, min_configs=2,
                              max_configs_per_graph=128, seed=a.seed)
dl = DataLoader(ds, batch_sampler=sam, num_workers=0)
model = HardwareAwareDiffPool(in_channels=158, hidden_channels=128, dropout=0.0).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
nparam = sum(p.numel() for p in model.parameters())
os.makedirs(a.out, exist_ok=True)
tag = f"seed{a.seed}{a.tag}"
logf = open(f"{a.out}/collapse_{tag}.log", "w")
def P(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

def zero_frac():
    return sum((p.detach().abs() < 1e-30).sum().item() for p in model.parameters()) / nparam

healthy, dead_streak, t0 = None, 0, time.time()
P(f"# repro seed={a.seed} lr={a.lr} lp_w={a.lp_weight} ent_w={a.entropy_weight} params={nparam}  sampler batches/epoch={len(sam)}")
for step, b in enumerate(dl):
    b = b.to(dev)
    model.train(); opt.zero_grad()
    scores, aux = _forward(model, b)
    rk = listmle_loss(scores, b.y)
    loss = rk + a.lp_weight * aux["lp_loss"] + a.entropy_weight * aux["entropy_loss"]
    if not torch.isfinite(loss):
        P(f"step {step} NON-FINITE loss rk={rk.item()} lp={aux['lp_loss'].item()} ent={aux['entropy_loss'].item()}"); continue
    loss.backward()
    gn = nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    opt.step()
    if step % a.log_every == 0 or step < 30:
        sstd = scores.std().item(); zf = zero_frac()
        P(f"step {step:5d} graph_nodes={b.x.shape[0]:>7} rank={rk.item():.4f} lp={aux['lp_loss'].item():.3f} "
          f"ent={aux['entropy_loss'].item():.3f} gnorm_preclip={gn.item():.2e} score_std={sstd:.2e} "
          f"zero_frac={zf:.3f} t={time.time()-t0:.0f}s")
        if zf < 0.01 and sstd > 1e-6:
            healthy = (step, copy.deepcopy(model.state_dict())); dead_streak = 0
        else:
            dead_streak += 1
        if a.stop_on_collapse and dead_streak >= 3:
            P(f"# COLLAPSED by step {step}; last healthy snapshot = step {healthy[0] if healthy else None}")
            break
    if step + 1 >= a.max_steps: P("# reached max-steps without collapse"); break
if healthy: torch.save({"step": healthy[0], "model_state": healthy[1], "seed": a.seed}, f"{a.out}/last_healthy_{tag}.pt")
torch.save({"step": step, "model_state": model.state_dict(), "seed": a.seed}, f"{a.out}/final_{tag}.pt")
P("# done")
