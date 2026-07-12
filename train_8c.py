#!/usr/bin/env python3
"""
train_8c.py — Phase 8c training entry point (MXU-feature co-design arms).

NEW FILE by design: train.py must stay byte-identical while other capped runs
execute. This wrapper IMPORTS train.py's machinery (sampler, train_epoch,
evaluate, arg parser) — zero duplication of the science — and changes exactly
three things, per the pre-registration:

  1. dataset  : MXUGraphDataset (dataset_mxu.py) with --mxu-arm
                {baseline | channel | spatial | full}
  2. model    : BaselineGraphSAGE_Large with in_channels = 158 + k
                (input-proj grows ≤ +2.8k params, ≤1.2% — prereg amendment A1)
  3. loop     : lean re-statement of train.py's main loop (same optimizer,
                cosine schedule, grad-clip, best-τ checkpointing).

Everything else (regime flags) is identical to the Phase 2c/3/6 protocol and
is supplied by run_phase8c.sh through the capped harness.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from dataset_mxu import MXUGraphDataset, ARM_COLUMNS
from model import BaselineGraphSAGE_Large
from train import SingleGraphBatchSampler, evaluate, train_epoch, parse_args
from torch_geometric.loader import DataLoader


def main() -> None:
    import argparse
    # Extract the one 8c flag, hand the rest to train.py's full parser.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--mxu-arm", default="full", choices=list(ARM_COLUMNS))
    arm_args, rest = p.parse_known_args()
    import sys
    sys.argv = [sys.argv[0]] + rest
    args = parse_args()
    arm = arm_args.mxu_arm

    # ── Reproducibility (identical to train.py) ──────────────────────────
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}\nPhase8c : arm={arm} (+{len(ARM_COLUMNS[arm])} MXU cols)")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = MXUGraphDataset(os.path.join(args.data_root, "train"), mxu_arm=arm)
    val_ds = MXUGraphDataset(os.path.join(args.data_root, "valid"), mxu_arm=arm)
    in_dim = train_ds.num_node_features
    print(f"  train : {train_ds.num_files} graphs, {len(train_ds):,} pairs   in_dim={in_dim}")

    sampler = SingleGraphBatchSampler(
        index=train_ds._index,
        configs_per_batch=args.configs_per_batch,
        shuffle=True,
        min_configs=2,
        max_configs_per_graph=args.max_configs_per_graph,
        seed=args.seed,
    )
    loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ── Model: wide flat baseline, input dim per arm ──────────────────────
    model = BaselineGraphSAGE_Large(in_channels=in_dim, dropout=args.dropout).to(device)
    n_par = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"Model   : wide-baseline in={in_dim} — {n_par:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best_model.pt")
    best_tau, best_epoch = -float("inf"), 0

    print("─" * 64 + f"\nTraining {args.epochs} epoch(s)\n" + "─" * 64)
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        t0 = time.perf_counter()
        stats = train_epoch(model, loader, optimizer, device, args.grad_clip,
                            lp_weight=args.lp_weight, entropy_weight=args.entropy_weight)
        scheduler.step()
        dt = time.perf_counter() - t0

        line = (f"Epoch {epoch:>3}/{args.epochs}  loss={stats['train_loss']:.5f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  ({dt:.1f}s)")
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val = evaluate(model, val_ds, device, eval_batch_size=args.eval_batch)
            tau = val["val_tau"]
            mark = ""
            if tau > best_tau:
                best_tau, best_epoch = tau, epoch
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "val_tau": tau, "mxu_arm": arm, "in_dim": in_dim,
                            "args": vars(args)}, best_path)
                mark = " ← saved"
            line += f"  val_τ={tau:+.4f}  best={best_tau:+.4f}@{best_epoch}{mark}"
        print(line, flush=True)

    print(f"Done.  Best val_τ = {best_tau:+.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
