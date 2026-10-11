"""Build the R4 data root (experiments/r4_heldout_conv_declaration.md §2-3) as symlinks.

layout/xla/random_r4/train = xla:random train minus the 11 held-out conv graphs (58 kept)
layout/xla/random_r4/valid = xla:random valid, unchanged (checkpoint selection as in 8c)
Idempotent; reads names only (no runtimes).
  python scripts/make_r4_split.py
"""
import os
from pathlib import Path

BASE = Path("data/predict-ai-model-runtime/npz_all/npz/layout/xla")
HELD_OUT = [
    "alexnet_train_batch_32",
    "mlperf_maskrcnn_1_shard_batch_4", "mlperf_maskrcnn_batch_2", "mlperf_maskrcnn_batch_4",
    "mlperf_ssd_1_shard_batch_8_fast_epoch", "mlperf_ssd_2_shard_batch_8_fast_epoch",
    "mnasnet_a1_batch_128",
    "retinanet.2x2.fp32", "retinanet.4x4.bf16.performance", "retinanet.4x4.fp32",
    "xception_imagenet",
]

if __name__ == "__main__":
    src_tr, src_va = BASE / "random" / "train", BASE / "random" / "valid"
    for split, src, skip in (("train", src_tr, set(HELD_OUT)), ("valid", src_va, set())):
        dst = BASE / "random_r4" / split
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob("*.npz")):
            if f.stem in skip:
                continue
            link = dst / f.name
            if not link.exists():
                os.symlink(f.resolve(), link)
    missing = [g for g in HELD_OUT if not (src_tr / f"{g}.npz").exists()]
    assert not missing, missing
    n_tr = len(list((BASE / "random_r4" / "train").glob("*.npz")))
    n_va = len(list((BASE / "random_r4" / "valid").glob("*.npz")))
    assert n_tr == 58 and n_va == 7, (n_tr, n_va)
    print(f"random_r4: train {n_tr} (held out {len(HELD_OUT)}), valid {n_va}")
