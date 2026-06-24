# GNN Cost Models for TPU Compiler Layout Ranking — A Rigorous Negative Result

Learned graph-neural-network cost models for ranking XLA layout configurations on the
[TPUGraphs](https://github.com/google-research-datasets/tpu_graphs) `layout:xla:random`
benchmark. The goal is to predict which compiler layout runs fastest (Kendall's τ;
paper baseline ≈ 0.34).

**What this project actually found** (see [`VERIFICATION.md`](VERIFICATION.md) for the
honest, claim-by-claim ledger):

1. **Two architectural interventions fail their controlled tests.** At a matched ~236k
   parameters, neither hierarchical pooling (DiffPool) nor hardware-contention edge
   augmentation beats a flat GraphSAGE — on global τ *or* the downstream Top-K metric.
2. **Seed variance dominates.** Paired-difference std (≈0.04–0.07) is ~6× any architectural
   effect. Early single-run "wins" were parameter-count + seed-luck artifacts. **Any τ
   comparison below ~0.05 is uninterpretable without multi-seed error bars.**
3. **A clean exploratory lead.** Error analysis shows the models fail bimodally: ~5% off
   optimal on transformer/UNet graphs, ~47% off on convolution graphs — a hypothesis (not
   a finding; n=7, post-hoc) about MXU convolution layout sensitivity.

This is a negative-results / "mythbuster" + noise-floor characterization, with every number
traceable and not a single p-hacked result.

> ⚠️ Read [`VERIFICATION.md`](VERIFICATION.md) before citing any number — it tags each
> claim 🔴 AI-asserted / 🟡 partial / 🟢 human-verified and says what each still needs.

---

## Repository layout

```
dataset.py            Early-Join features + structural (hardware-contention) edges
model.py              BaselineGraphSAGE (flat) + BaselineGraphSAGE_Large (param-matched)
model_phase2.py       HierarchicalGraphSAGE (memory-safe DiffPool)
model_phase3.py       HardwareAwareDiffPool (GINEConv + S^T E^c S edge coarsening)
train.py              training loop, SingleGraphBatchSampler, evaluate()
loss.py               ListMLE ranking loss
analyze_opcodes.py    opcode / compute-vs-memory analysis (corrected TPUGraphs enum)
analyze_topology.py   spectral gap (λ₁), diameter
phase4_evaluate_topk.py   downstream Top-K autotuner metric
phase5_error_analysis.py  per-graph error attribution vs graph structure
run_ablation_seeds.sh / run_phase3_seeds.sh   multi-seed experiment drivers
scripts/              extract_opcode_map.py, plotting
experiments/          phase-by-phase write-ups (tutorial depth) + figures/
VERIFICATION.md       claim-by-claim human-verification ledger
checkpoints/          trained weights (.pt) + results_summary.txt per experiment
```

The phase write-ups in [`experiments/`](experiments/) are the readable narrative:
Phase 1/2 (baseline + DiffPool), 2c (parameter-matched ablation), 3 (hardware edges),
4 (Top-K), 5 (error analysis).

---

## Setup

Two external dependencies are **not** included in this repo (size / licensing):

1. **The dataset** (`data/`, ~7 GB) — the public TPUGraphs `layout:xla:random` split.
   Download via the official repo
   ([google-research-datasets/tpu_graphs](https://github.com/google-research-datasets/tpu_graphs))
   and place it at:
   ```
   data/predict-ai-model-runtime/npz_all/npz/layout/xla/random/{train,valid,test}/
   ```
2. **The reference repo** (`reference/`) — a clone of Google's `tpu_graphs`, used *only* by
   `scripts/extract_opcode_map.py` to derive the opcode→id map from
   `reference/tpu_graphs/process_data/xla/hlo_opcode.h`. The corrected ids are already
   hard-coded in `dataset.py`/`analyze_opcodes.py`, so the pipeline runs without it; clone
   it only to re-verify the enum:
   ```
   git clone https://github.com/google-research-datasets/tpu_graphs reference
   ```

Then:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # see file header re: CUDA-specific torch/torch_scatter
pytest test_dataset.py             # 7 tests should pass
```

---

## Reproducing the results

```bash
# Phase 1/2 baselines (see experiments/phase1_phase2_results.md for exact commands)
python train.py --model baseline --epochs 50 --configs-per-batch 32 \
    --max-configs-per-graph 512 --num-workers 4

# Phase 2c — parameter-matched ablation (flat vs DiffPool), 5 seeds
bash run_ablation_seeds.sh          # → checkpoints/ablation/results_summary.txt

# Phase 3 — hardware-aware edges, 5 seeds
bash run_phase3_seeds.sh            # → checkpoints/phase3/results_summary.txt

# Phase 4 — downstream Top-K metric
python phase4_evaluate_topk.py --all-seeds

# Phase 5 — error analysis (which graphs are mispredicted, and why)
python phase5_error_analysis.py

# Figures
python scripts/plot_ablation_results.py
python scripts/plot_training_results.py
```

> **Note on `--max-configs-per-graph` and `--num-workers`:** always set both. Without the
> cap an epoch processes ~1M configs (4+ hr epochs); without workers the GPU starves on
> single-threaded data loading. Details in `experiments/phase1_phase2_results.md`.

---

## References (cited, not redistributed)

- Phothilimthana et al., *TPUGraphs* (NeurIPS 2023) — dataset & baseline.
- Ying et al., *Hierarchical Graph Representation Learning with Differentiable Pooling*
  (DiffPool, NeurIPS 2018).
- Hu et al., *Strategies for Pre-training Graph Neural Networks* (GINE, ICLR 2020).
- Jouppi et al., *A Domain-Specific Supercomputer for Training Deep Neural Networks*
  (TPU v2/v3, CACM 2020).

(PDFs are kept locally in `papers/`, not committed — copyright.)
