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
