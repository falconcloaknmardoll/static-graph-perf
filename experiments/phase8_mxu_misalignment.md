# Phase 8 — MXU misalignment vs the convolution failure

## Test set 1: the 7 Phase-5 validation graphs

graph                     family         slow mxu_frac overhead cout_u   k_u  rt_cv rt_p90/10
unet_3d.4x4.bf16          memory-bound  1.010   0.0168    1.524   0.76  0.99   0.00      1.01
tf2_bert_pretrain_dynamic transformer   1.056   0.0000      n/a    n/a   n/a   0.03      1.05
bert_pretraining.4x4.fp16 transformer   1.067   0.0000      n/a    n/a   n/a   0.03      1.03
mlperf_bert_batch_24_2x2  transformer   1.072   0.0000      n/a    n/a   n/a   0.03      1.03
inception_v3_batch_128_tr convolution   1.441   0.0243    1.690   0.72  0.95   0.11      1.05
resnet_v1_50_official_bat convolution   1.452   0.0258    1.171   0.94  0.98   0.04      1.08
resnet50.4x4.fp16         convolution   1.512   0.0279    1.171   0.94  0.98   0.15      1.10

## T1 (confirmatory): MXU signal vs Phase-5 slowdown across 7 graphs
  mxu_frac     vs slowdown: Spearman ρ=+0.79  (n=7)
  runtime CV   vs slowdown: Spearman ρ=+0.96  (n=7)  ← real-runtime layout sensitivity, model-independent
  (mxu_frac/overhead are only defined where conv exists, so T1 partly
   restates the conv-vs-not split; the load-bearing tests are T2/T3.)

## T2 (mechanism, within the 3 convolution graphs)
  inception_v3_batch_128_train        slowdown=1.441  MXU overhead=1.690  cout_util=0.72  k_util=0.95
  resnet_v1_50_official_batch_128_bf  slowdown=1.452  MXU overhead=1.171  cout_util=0.94  k_util=0.98
  resnet50.4x4.fp16                   slowdown=1.512  MXU overhead=1.171  cout_util=0.94  k_util=0.98
  → does higher MXU overhead track higher slowdown? NO (orders differ)

## T3 (the non-circular test): layout sensitivity vs MXU across ALL xla graphs
  scanned 160 xla graphs (train+valid, default+random).
  layout sensitivity (runtime CV)  — median:
     conv-bearing graphs (n=105): 0.239
     conv-free graphs    (n=55): 0.086
  mxu_frac vs runtime CV across all graphs: Spearman ρ=+0.54 (n=144)
  MXU overhead vs runtime CV (conv graphs only): Spearman ρ=-0.06 (n=95)

## Honest caveats
  • T1/T2 are n=7 / n=3 — directional, not significance. Same post-hoc
    grouping caveat as Phase 5 (C10 is exploratory).
  • MXU overhead is a STATIC property of conv shapes; it explains why conv
    graphs are a distinct regime, but whether ENCODING it helps the model
    is the Phase-8c retrain question — not settled here.
  • T3 uses real runtimes (no model), so it is the most defensible: it asks
    whether convolution-heavy graphs are genuinely more layout-sensitive.

---

## Addendum (2026-07-07): roofline characterization + 3-D kernel-volume fix

**Bug fix (matters for 8c):** `mxu_features.py` read only `window_size_0/1`, undercounting
kernel volume for **3-D convolutions** (unet_3d: 3×3×3 counted as 3×3). Fixed to the full
spatial volume (all `window_size_*` slots); 2-D graphs verified bit-identical (ResNet
K=147/576 unchanged). All §-above headline numbers re-run and unchanged. Caught BEFORE the
Phase-8c retrain would have trained on the undercounted feature.

**Roofline figure (`figures/phase8_roofline.png`, `scripts/plot_roofline.py`):** static
roofline position (arithmetic intensity vs MXU packing utilization, MAC-weighted) of every
conv op in the 4 conv-bearing val graphs, overlaid with Phase-5 model failure.
- **All convs are compute-bound** (AI ≈ 90–2,900, well past the ≈137 FLOP/B TPU-v3 ridge —
  ridge constant VERIFIED 2026-08-25: 123 TFLOP/s bf16 peak ÷ 900 GB/s HBM, both per chip, per
  Google's official Cloud TPU v3 spec page, the architecture Jouppi et al. 2020 describes; see
  `scripts/plot_roofline.py` header for the full citation note).
- **The most MXU-intense graph is the one the model handles BEST:** UNet-3D's convs have
  the highest AI (MAC-wtd ≈2,857) at util 0.75, slowdown 1.01; the failing ResNets sit at
  LOWER AI (≈700) and HIGHER packing utilization (0.92).
- **Conclusion: static roofline position does not order model failure** — the third
  independent static-feature negative (after channel-padding ρ=−0.06 and T2 order
  mismatch). Each one sharpens the representational-floor thesis: the difficulty is in
  the layout/tiling *search dynamics*, which no static shape-derived feature captures.
- Paper use: architect-coded headline visual for §7–8; also pre-registers our skepticism
  (P1/P2 in `phase8c_preregistration.md`) with visual evidence.

---

## Erratum (2026-10-11, framing audit) — T3 counts and ρ

The T3 glob (`scripts/analyze_mxu_misalignment.py:160`, `xla/*/*/*.npz`) also picks up the 16
`test/` graphs, whose `config_runtime` is all zeros, so their runtime CV is NaN. They are dropped
from the medians but counted in the printed n. Correct counts: **conv-bearing n = 95, conv-free
n = 49, total with runtimes 144.** Medians 0.239 / 0.086 are unchanged (2.77×; Mann-Whitney
p = 1.5e-7, a test this report did not originally run). The script's `spearman()` ranks with argsort
and does not average ties (mxu_frac has 49 tied zeros); the tie-corrected ρ is **+0.52**
(p = 2.5e-11), not +0.54. The overhead ρ = −0.06 (n = 95) holds (scipy −0.065, p = .53). The
output above is left as generated. See `framing_audit_2026-10-11.md`.

## Erratum (2026-10-13) — "all convs are compute-bound" and the n = 7 ρ

The roofline addendum's "All convs are compute-bound (AI ≈ 90–2,900)" is wrong per op. Recomputed with
`scripts/plot_roofline.py`'s loader: below-ridge (AI < 137) conv ops are UNet-3D 6/53 (0.8% of MACs),
Inception-v3 53/293 (5.4%), ResNet-v1-50 41/158 (15.7%), ResNet-50 41/158 (15.7%); per-op AI spans
3.8–11,810. Correct claim: conv work is **predominantly** compute-bound (74–89% of ops, 84–99% of MACs);
MAC-weighted AI 688 / 696 / 732 / 2,857. The figure itself was always right.

T1's mxu_frac vs slowdown ρ = +0.79 has the same no-tie-averaging bug as T3 (3 tied zeros among 7
graphs): scipy gives **ρ = +0.74, p = .057**. The runtime-CV ρ = +0.96 likely has the same problem, but it
can't be recomputed from the 2-decimal table above; rerun before citing it.
