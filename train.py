"""
train.py
--------
Phase 1 Training Loop — GraphSAGE Baseline with ListMLE and Kendall's Tau.

Critical design decisions
~~~~~~~~~~~~~~~~~~~~~~~~~
(1) SingleGraphBatchSampler
    Ensures every training batch contains configurations of exactly ONE
    computation graph.  Standard DataLoader with shuffle=True would silently
    mix configs from different graphs, making ListMLE compute a meaningless
    cross-graph ranking.  See the class docstring for the full explanation.

(2) num_workers=0
    macOS uses spawn-based multiprocessing.  With spawn, each DataLoader
    worker re-imports the module and calls TPUGraphDataset.__init__ from
    scratch, reopening all mmap file handles in every worker simultaneously.
    For BERT-scale graphs this can exhaust file descriptors and memory.

(3) Gradient clipping (max_norm=1.0)
    ListMLE gradients can be large early in training when scores are far from
    the correct ordering.  Clipping prevents the NaN cascade that follows an
    exploding gradient step.

Usage
~~~~~
    # Minimal run (no wandb)
    python train.py --no-wandb

    # Full run with wandb (install first: pip install wandb)
    python train.py --epochs 50 --configs-per-batch 32

    # Memory-limited: reduce configs per batch for BERT-scale graphs
    python train.py --configs-per-batch 8 --max-configs-per-graph 512 --no-wandb

    # Resume from checkpoint
    python train.py --resume checkpoints/best_model.pt --no-wandb
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import scipy.stats as sp_stats
import torch
import torch.nn as nn
from torch.utils.data import Sampler
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from dataset import TPUGraphDataset
from loss import listmle_loss
from model import BaselineGraphSAGE


_DATA_ROOT = os.path.join(
    "data", "predict-ai-model-runtime", "npz_all", "npz", "layout", "xla", "random"
)


# ─────────────────────────────────────────────────────────────────────────────
# SingleGraphBatchSampler — The Critical Batching Fix
# ─────────────────────────────────────────────────────────────────────────────

class SingleGraphBatchSampler(Sampler):
    """
    Yields batches where every index belongs to the same computation graph.

    WHY THIS EXISTS
    ───────────────
    TPUGraphDataset._index is a flat list of (file_idx, cfg_idx) pairs.
    Indices 0 through 29,143 are all configurations of graph 0 (AlexNet);
    index 29,144 is the first configuration of graph 1 (BERT).

    A standard DataLoader(dataset, batch_size=32, shuffle=True) will draw 32
    random indices, mixing configurations from different graphs into the same
    batch.  ListMLE applied to this batch ranks an AlexNet configuration against
    a BERT configuration — quantities measured in different computation graphs
    with incomparable runtimes.  The loss decreases (the model overfits the
    cross-graph ordering) but Kendall's Tau on validation will be near zero,
    because the model learns nothing about within-graph configuration quality.

    This class groups indices by file_idx and yields chunks of
    `configs_per_batch` that all share the same file_idx.

    INVARIANT guaranteed for every yielded batch B:
        len(B) >= min_configs
        dataset._index[B[i]][0] == dataset._index[B[j]][0]   ∀ i, j ∈ B

    Parameters
    ----------
    index : list of (file_idx, cfg_idx)
        The dataset._index from a TPUGraphDataset instance.
    configs_per_batch : int
        Number of configurations per yielded batch.  Trailing chunks with
        fewer than `min_configs` items are silently dropped.
    shuffle : bool
        If True, randomise (a) the order graphs are visited and (b) which
        configurations are drawn from each graph on each epoch.
        Call sampler.set_epoch(epoch) before each epoch for distinct,
        reproducible shuffling per epoch.
    min_configs : int
        Minimum batch size.  Must be >= 2 (ListMLE is degenerate with one
        item: the loss is zero and gradients vanish regardless of the score).
        Graphs with fewer than min_configs total configurations are dropped.
    max_configs_per_graph : int or None
        Cap the number of configurations sampled per graph per epoch.
        For BERT-pretraining (18,776–22,944 configs) this prevents one graph
        from dominating the epoch.  None = use all configurations.
    seed : int
        Base random seed.  Epoch i uses seed + i, giving distinct but
        reproducible shuffling across epochs without a global RNG side-effect.
    """

    def __init__(
        self,
        index: List[Tuple[int, int]],
        configs_per_batch: int = 32,
        shuffle: bool = True,
        min_configs: int = 2,
        max_configs_per_graph: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if min_configs < 2:
            raise ValueError("min_configs must be >= 2; ListMLE needs at least 2 items.")
        if configs_per_batch < min_configs:
            raise ValueError(
                f"configs_per_batch ({configs_per_batch}) must be >= min_configs ({min_configs})."
            )

        self.configs_per_batch      = configs_per_batch
        self.shuffle                = shuffle
        self.min_configs            = min_configs
        self.max_configs_per_graph  = max_configs_per_graph
        self._base_seed             = seed
        self._epoch: int            = 0

        # Group flat dataset indices by file_idx
        raw: Dict[int, List[int]] = defaultdict(list)
        for flat_idx, (file_idx, _) in enumerate(index):
            raw[file_idx].append(flat_idx)

        # Drop graphs that can never form a valid batch
        self._groups: List[List[int]] = [
            g for g in raw.values() if len(g) >= min_configs
        ]

        n_dropped = len(raw) - len(self._groups)
        if n_dropped:
            print(
                f"[Sampler] Dropped {n_dropped} graph(s) with < {min_configs} configs.",
                file=sys.stderr,
            )
        if not self._groups:
            raise RuntimeError(
                "No graphs have enough configurations to form a valid batch. "
                f"Lower min_configs (currently {min_configs}) or check your dataset."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Call at the top of each training epoch for per-epoch shuffling."""
        self._epoch = epoch

    @property
    def num_graphs(self) -> int:
        """Number of graphs contributing to training batches."""
        return len(self._groups)

    # ── Sampler protocol ──────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self._base_seed + self._epoch)

        graph_order = np.arange(len(self._groups))
        if self.shuffle:
            rng.shuffle(graph_order)

        for gidx in graph_order:
            g = self._groups[int(gidx)].copy()

            if self.shuffle:
                rng.shuffle(g)

            if self.max_configs_per_graph is not None:
                g = g[: self.max_configs_per_graph]

            # Yield non-overlapping chunks of configs_per_batch
            for start in range(0, len(g), self.configs_per_batch):
                chunk = g[start : start + self.configs_per_batch]
                if len(chunk) >= self.min_configs:
                    yield chunk

    def __len__(self) -> int:
        """Approximate number of batches per epoch."""
        total = 0
        for g in self._groups:
            n = min(len(g), self.max_configs_per_graph or len(g))
            full = n // self.configs_per_batch
            tail = n % self.configs_per_batch
            total += full + (1 if tail >= self.min_configs else 0)
        return total


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_dataset: TPUGraphDataset,
    device: torch.device,
    eval_batch_size: int = 32,
) -> Dict[str, float]:
    """
    Compute mean Kendall's Tau across all validation graphs.

    For each graph, we:
      1. Retrieve true runtimes directly from the mmap'd npz (fast, no Early-Join).
      2. Run model inference in mini-batches of `eval_batch_size` configurations.
      3. Compute scipy.stats.kendalltau(predicted_scores, -true_runtimes).
         Sign convention: higher predicted score = predicted faster config,
         lower runtime = faster config; tau > 0 for a useful model.
      4. Average tau across all graphs with >= 2 configurations.

    Parameters
    ----------
    model : nn.Module
        The model in eval mode (called externally with model.eval() by caller).
    val_dataset : TPUGraphDataset
        Validation split dataset.
    device : torch.device
        Device for model inference.
    eval_batch_size : int
        Number of configurations processed per forward pass during evaluation.
        Reduce if BERT-scale graphs cause OOM; does not affect tau calculation.

    Returns
    -------
    dict with keys:
        "val_tau"      : mean Kendall's Tau (float)
        "val_n_graphs" : number of graphs evaluated (int)
    """
    model.eval()

    # Group validation flat indices by file_idx
    groups: Dict[int, List[int]] = defaultdict(list)
    for flat_idx, (file_idx, _) in enumerate(val_dataset._index):
        groups[file_idx].append(flat_idx)

    taus: List[float] = []

    for file_idx, flat_indices in groups.items():
        if len(flat_indices) < 2:
            continue

        # ── True runtimes: read directly from mmap (no full Early-Join) ────
        # _index[i] = (file_idx, cfg_idx); we need cfg_idx per flat index.
        cfg_indices   = [val_dataset._index[i][1] for i in flat_indices]
        npz           = val_dataset._npz_handles[file_idx]
        true_runtimes = np.array(npz["config_runtime"], dtype=np.float64)[cfg_indices]

        # ── Predicted scores: mini-batched forward passes ──────────────────
        all_preds: List[torch.Tensor] = []

        for start in range(0, len(flat_indices), eval_batch_size):
            chunk     = flat_indices[start : start + eval_batch_size]
            data_list = [val_dataset[i] for i in chunk]
            batch     = Batch.from_data_list(data_list).to(device)

            preds = model(batch.x, batch.edge_index, batch.batch)  # (len(chunk),)
            all_preds.append(preds.cpu())

        preds_np = torch.cat(all_preds).numpy()  # (n_configs,)

        # ── Kendall's Tau for this graph ───────────────────────────────────
        # tau > 0 ↔ high predicted score correlates with low (fast) runtime
        tau, _ = sp_stats.kendalltau(preds_np, -true_runtimes)
        if not np.isnan(tau):
            taus.append(float(tau))

    mean_tau = float(np.mean(taus)) if taus else 0.0
    return {"val_tau": mean_tau, "val_n_graphs": len(taus)}


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> Dict[str, float]:
    """
    Run one full training epoch.

    Each batch from `loader` is guaranteed by SingleGraphBatchSampler to
    contain K configurations of the same computation graph.  The model
    produces K scores; listmle_loss computes the ranking loss.

    Returns
    -------
    dict with keys:
        "train_loss"      : mean per-batch loss for this epoch
        "skipped_batches" : number of non-finite batches that were skipped
    """
    model.train()
    total_loss    = 0.0
    total_batches = 0
    skipped       = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        scores = model(batch.x, batch.edge_index, batch.batch)  # (K,)
        loss   = listmle_loss(scores, batch.y)

        # Guard: a non-finite loss indicates a degenerate batch or
        # scores that have already diverged.  Skip rather than propagate NaN.
        if not torch.isfinite(loss):
            skipped += 1
            continue

        loss.backward()

        # Clip gradients before the optimizer step.  ListMLE gradients can
        # spike early in training when scores are far from the correct order.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss    += loss.item()
        total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)
    return {"train_loss": mean_loss, "skipped_batches": skipped}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 — GraphSAGE Baseline Training (TPUGraphs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Data")
    g.add_argument("--data-root", default=_DATA_ROOT,
                   help="Root directory containing train/valid/test subdirs.")
    g.add_argument("--configs-per-batch", type=int, default=32,
                   help="Configurations per training batch (all from one graph). "
                        "Reduce to 8–16 for BERT-scale graphs if OOM.")
    g.add_argument("--max-configs-per-graph", type=int, default=None,
                   help="Cap configs sampled per graph per epoch. "
                        "Useful to balance epoch time when one graph dominates.")
    g.add_argument("--eval-batch", type=int, default=32,
                   help="Mini-batch size for validation forward passes.")

    # ── Model ─────────────────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument("--hidden",   type=int,   default=128,
                   help="GNN and first MLP hidden dimension.")
    g.add_argument("--dropout",  type=float, default=0.1)

    # ── Training ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",       type=int,   default=50)
    g.add_argument("--lr",           type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--grad-clip",    type=float, default=1.0)
    g.add_argument("--eval-every",   type=int,   default=5,
                   help="Run validation every N epochs.")
    g.add_argument("--seed",         type=int,   default=42)
    g.add_argument("--resume",       type=str,   default=None,
                   help="Path to a checkpoint file to resume training from.")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    g = p.add_argument_group("Checkpointing")
    g.add_argument("--ckpt-dir", default="checkpoints",
                   help="Directory for saved checkpoints.")

    # ── Logging (WandB) ───────────────────────────────────────────────────────
    g = p.add_argument_group("Logging")
    g.add_argument("--project",  default="tpugraphs-baseline",
                   help="WandB project name.")
    g.add_argument("--run-name", default=None,
                   help="WandB run name (auto-generated if None).")
    g.add_argument("--no-wandb", action="store_true",
                   help="Disable WandB logging even if wandb is installed.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device  : {device}")
    print(f"PyTorch : {torch.__version__}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dir = os.path.join(args.data_root, "train")
    valid_dir = os.path.join(args.data_root, "valid")

    print("\nLoading datasets …")
    train_ds = TPUGraphDataset(data_dir=train_dir)
    print(f"  train : {train_ds.num_files} graphs, {len(train_ds):,} (graph, config) pairs")
    val_ds = TPUGraphDataset(data_dir=valid_dir)
    print(f"  valid : {val_ds.num_files} graphs, {len(val_ds):,} (graph, config) pairs")

    # ── Sampler & DataLoader ──────────────────────────────────────────────────
    sampler = SingleGraphBatchSampler(
        index=train_ds._index,
        configs_per_batch=args.configs_per_batch,
        shuffle=True,
        min_configs=2,
        max_configs_per_graph=args.max_configs_per_graph,
        seed=args.seed,
    )
    print(
        f"\nSampler : {sampler.num_graphs} graphs active, "
        f"~{len(sampler)} batches/epoch, "
        f"{args.configs_per_batch} configs/batch"
    )

    # batch_sampler= replaces batch_size/shuffle/drop_last in DataLoader.
    # num_workers=0: required on macOS (see module docstring, Point 2).
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BaselineGraphSAGE(
        in_channels=158,
        hidden_channels=args.hidden,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model   : {n_params:,} trainable parameters")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Cosine decay from lr → lr/100 over the full training horizon.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 1
    best_tau    = -float("inf")
    best_epoch  = 0

    if args.resume:
        if not os.path.isfile(args.resume):
            sys.exit(f"ERROR: checkpoint not found at {args.resume!r}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1
        best_tau    = ckpt.get("val_tau", -float("inf"))
        best_epoch  = ckpt["epoch"]
        print(f"\nResumed from epoch {ckpt['epoch']} (best_tau={best_tau:+.4f})")

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = _WANDB_AVAILABLE and not args.no_wandb
    if use_wandb:
        wandb.init(project=args.project, name=args.run_name, config=vars(args))
        wandb.watch(model, log="gradients", log_freq=100)
        print(f"WandB   : project={args.project}")
    elif not args.no_wandb and not _WANDB_AVAILABLE:
        print(
            "\n[WARN] wandb not installed — logging disabled.\n"
            "       Install with:  pip install wandb\n"
            "       Then run:      wandb login\n"
        )

    # ── Checkpoint directory ───────────────────────────────────────────────────
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.ckpt_dir, "best_model.pt")

    # ── Training Loop ─────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print(f"Training for {args.epochs} epoch(s), starting at epoch {start_epoch}")
    print("─" * 64)

    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        t0 = time.perf_counter()

        train_stats = train_epoch(model, loader, optimizer, device, args.grad_clip)
        scheduler.step()

        elapsed = time.perf_counter() - t0

        if train_stats["skipped_batches"]:
            print(
                f"  [WARN] epoch {epoch}: {train_stats['skipped_batches']} "
                "non-finite batch(es) skipped — check for score divergence."
            )

        log: Dict[str, float] = {
            "epoch":         float(epoch),
            "train_loss":    train_stats["train_loss"],
            "lr":            scheduler.get_last_lr()[0],
            "epoch_time_s":  elapsed,
            "skipped":       float(train_stats["skipped_batches"]),
        }

        # ── Validation ────────────────────────────────────────────────────────
        val_str = ""
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_stats = evaluate(model, val_ds, device, args.eval_batch)
            log.update(val_stats)

            if val_stats["val_tau"] > best_tau:
                best_tau   = val_stats["val_tau"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch":       epoch,
                        "model_state": model.state_dict(),
                        "opt_state":   optimizer.state_dict(),
                        "val_tau":     best_tau,
                        "args":        vars(args),
                    },
                    best_ckpt_path,
                )
                saved_marker = " ← saved"
            else:
                saved_marker = ""

            val_str = (
                f"  val_τ={val_stats['val_tau']:+.4f}"
                f"  best={best_tau:+.4f}@{best_epoch}{saved_marker}"
            )

        print(
            f"Epoch {epoch:>3}/{args.epochs}"
            f"  loss={train_stats['train_loss']:.5f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
            f"  ({elapsed:.1f}s)"
            f"{val_str}"
        )

        if use_wandb:
            wandb.log(log)

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print(f"Done.  Best val_τ = {best_tau:+.4f} at epoch {best_epoch}.")
    if os.path.isfile(best_ckpt_path):
        print(f"Best checkpoint : {best_ckpt_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
