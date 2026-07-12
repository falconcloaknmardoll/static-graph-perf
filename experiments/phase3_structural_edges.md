# Phase 3 — Hardware-Aware Graph Augmentation (Granite)
## TPUGraphs GNN Compiler Optimization

**Status:** dataset augmentation complete & verified; `model_phase3.py` built, trained,
and **evaluated → null result** (hardware edges do not beat the flat baseline; see
[`phase3_results.md`](phase3_results.md), Δτ=−0.0116, paired p=0.72).
**Baseline to beat:** the **parameter-matched flat GraphSAGE** (`BaselineGraphSAGE_Large`,
h=172, ~236k params), τ = 0.2914 ± 0.034 over 5 seeds at the 128-config cap
(see [`phase2c_ablation_results.md`](phase2c_ablation_results.md)). Best *headline* model
overall is still GraphSAGE warm start (τ = 0.3676, single-run — C4).

> This document is written to be **re-readable from cold** — it explains the *why*
> behind each idea, not just the *what*. If you are returning to this project after
> a break, read this top-to-bottom and you should be able to rebuild the full mental
> model of Phase 3.

> **⚠️ Context update (2026-06-22) — why Phase 3 matters more now.** Phase 2's hierarchy
> approach (DiffPool) **failed its parameter-matched ablation** ([Phase 2c](phase2c_ablation_results.md)):
> at equal capacity it gave no significant gain over a flat GNN. So the "fix over-squashing by
> coarsening" lever did **not** pan out. Phase 3 is the *other* bet — a **targeted hardware
> inductive bias** rather than generic capacity or pooling. It must be evaluated against the
> honest flat baseline above (5 seeds, same cap), not against the confounded Phase 2b numbers.

---

## 0. One-paragraph summary

Our graph so far only encodes **data flow** (which operation feeds which). But TPU
latency is also shaped by **hardware resource contention** — operations that fight
over the same physical unit (the matrix multiply unit, or memory bandwidth) slow each
other down even when there is no data dependency between them. In Phase 3 we *add new
edges* to the graph to represent this contention, so the GNN can "see" it. We chose to
do this with a single edge list plus an **edge type label** (rather than a fancier
multi-graph structure), and we discovered along the way that the dataset's opcode
integers use a **non-standard custom numbering** that we had to extract from Google's
source code to get right.

---

## 1. Motivation — why add edges at all?

### 1.1 The over-squashing problem (recap from Phase 1/2)

A GNN passes messages along edges. After `L` layers, a node has only "heard from"
neighbors up to `L` hops away. XLA computation graphs are **long and stringy** (low
spectral gap, large diameter), so information from a distant but important operation
gets **squashed** into a fixed-size vector before it can travel far enough. This is
*over-squashing*, and it was our core bottleneck in Phase 1. Phase 2 (DiffPool)
attacked it by **coarsening** the graph (shrinking the diameter) — but that approach
**did not survive its parameter-matched ablation** ([Phase 2c](phase2c_ablation_results.md)):
hierarchy gave no significant gain over a same-size flat model, which is itself evidence that
plain diameter-reduction is not the operative lever here. Phase 3 attacks the problem from a
different angle: instead of generic coarsening, **add *targeted* shortcut edges** that encode a
specific physical phenomenon (hardware contention), so relevant operations become direct
neighbors *for a reason the hardware actually cares about*.

### 1.2 What data-flow edges miss

The original `edge_index` is a list of `(producer → consumer)` pairs: op A feeds its
output into op B. That captures *correctness* dependencies, but not *performance*
ones. Two matrix multiplications scheduled close together in time will **compete for
the single systolic MXU** even if neither uses the other's output. The data-flow graph
has no edge between them, so a plain GNN can never reason about that contention.

**Phase 3's hypothesis:** if we add edges that connect operations competing for the
same hardware resource, the GNN gains an *inductive bias* about latency — a built-in
hint that "these ops interfere." This is the idea behind Google's **Granite** work,
hence the phase name.

---

## 2. The design decision — Option A vs Option B

We need to attach a *type* to each edge (is it a data-flow edge, or an MXU-contention
edge, or a memory-contention edge?). There are two standard ways to do this in PyTorch
Geometric:

### Option A — Heterogeneous graph (`HeteroData`)
Store three completely separate edge lists, one per relation:
`(node, data_flow, node)`, `(node, mxu_contention, node)`, `(node, mem_contention, node)`.
The model would then use a heterogeneous convolution (e.g. `HeteroConv`, `RGCNConv`)
that runs a *separate* message pass per relation and sums the results.

### Option B — Homogeneous graph + edge attributes ✅ CHOSEN
Keep **one** `edge_index`, but attach an `edge_attr` tensor that labels each edge with
a **one-hot vector**:
```
data-flow edge        → [1, 0, 0]
MXU-contention edge   → [0, 1, 0]
memory-contention edge→ [0, 0, 1]
```

### Why we chose B

| Reason | Explanation |
|---|---|
| **Protects DiffPool** | Our Phase 2b pooling relies on a *single* `edge_index` and a uniform-node-count trick (`x.view(B, N, K)`, plus a `scatter`-based sparse `A@S`). HeteroData gives three edge lists and forces a heterogeneous conv that **materializes messages per relation → 3× the activation memory**. On BERT-scale graphs we are already near the 16 GB limit on the RTX 5080. Option A would mean rewriting (and re-validating) the DiffPool pipeline we'd already built and tested. |
| **Protects the pipeline** | The `SingleGraphBatchSampler`, the 158-dim Early-Join `x`, and `config_mask` all assume a homogeneous `Data` object. HeteroData ripples changes through the whole stack. |
| **Memory is trivial** | `edge_attr` is `(E', 3)` floats — a rounding error. Option A stores 3 edge structures *and* does 3 message passes. |
| **Model-agnostic** | Even a model that ignores `edge_attr` still benefits, because the extra edges shorten paths (more diameter reduction — the same lever that won Phase 2b). |

> **Important caveat (the catch for later):** the *semantic* distinction between edge
> types only pays off once the **model actually reads `edge_attr`**. A plain `SAGEConv`
> ignores it — it would just see extra undifferentiated neighbors. So the Phase 3 model
> step is to swap `SAGEConv → GINEConv` (or `GENConv`), convs that accept `edge_dim=3`
> and fold the edge type into the message. The dataset work below is the *same*
> regardless of that choice, which is why we did it first.

---

## 3. The edge-construction heuristic — how to add edges without exploding

### 3.1 The trap: O(N²) cliques

The naive way to model "all MXU ops contend" is to connect every MXU op to every other
MXU op. On a 40,000-node graph that is a **dense clique = ~1.6 billion edges → instant
out-of-memory.** We must add edges *locally*.

### 3.2 The key reframing: contention is *temporal*

Two matrix multiplies only fight over the MXU if they run at **roughly the same time**.
So we don't need all-to-all edges — we only need to connect ops that are **scheduled
near each other**.

**How do we know the schedule?** We use **node-ID order as a proxy for schedule order.**
TPUGraphs stores nodes in roughly HLO instruction order, which is already close to
topological/execution order. Using node-ID order avoids running an expensive per-graph
topological sort (which would also need special handling for `while`-loop cycles) and
is robust.

### 3.3 The sliding-window contention chain

For each resource (MXU, MEM):
1. Take the list of nodes of that resource type, **in node-ID order**.
2. Connect each op to the **next `W` same-resource ops** in that list.
3. Make the edges **bidirectional** (contention is mutual — A interferes with B *and*
   B with A).

This produces a "band" of edges, not a clique. Picture the MXU ops laid out in a line;
each one links to its `W` nearest same-type neighbors on each side. The result models
"ops queued close together compete," at a bounded cost.

**Edge-count bound:** for a resource with `M` nodes and window `W`, this adds at most
`2·W·M` edges. Total augmented graph:
```
E' ≤ E_original + 2·W_mxu·N_mxu + 2·W_mem·N_mem
```
This is **linear in N**, never quadratic. (This exact inequality is asserted in the
test suite — see §6.)

### 3.4 Window sizes: why 5 for MXU but 15 for MEM

The two windows are **deliberately asymmetric**, for a microarchitectural reason:

- **`mxu_window = 5`** — a TPU has essentially **one** big systolic matrix unit. Very
  few matmuls can be "in flight" at once, so contention is short-range: an op realistically
  only contends with a handful of temporal neighbors.
- **`mem_window = 15`** — High-Bandwidth Memory (HBM) can service **many** concurrent
  loads/stores. Bandwidth *saturation* builds up across a **wider** group of memory ops,
  not just adjacent pairs. A wider window captures that broader contention.

(These are reasonable priors, not tuned values. If Phase 3 modeling stalls, the windows
are a knob to sweep.)

---

## 4. The opcode-enum-shift bug (the important story)

This is the part most likely to confuse future-you, so it gets full detail.

### 4.1 What is an "opcode"?

Every node in the graph is an XLA operation — `convolution`, `dot` (matrix multiply),
`add`, `reshape`, etc. The dataset doesn't store the *name*; it stores an **integer
code** in the `node_opcode` array (e.g. node 5 has opcode `34`). To define `MXU_OPS`
we need to know **which integer means `dot`**, which integer means `convolution`, and
so on.

### 4.2 How we got it wrong at first

Our earlier script `analyze_opcodes.py` assumed `node_opcode` followed the **standard
XLA HLO enum** — the alphabetical 0-indexed list straight from XLA's source
(`hlo_opcode.h` in TensorFlow). Under that assumption, `convolution = 30` and
`dot = 39`. We built `MXU_OPS = {30, 39}` on that basis.

### 4.3 The tell: BERT had zero Dots

When we ran the distribution check, the MXU lane was nearly empty:

| (WRONG enum) | count |
|---|---|
| Convolution (30) | 4 total, across the whole 69-graph split |
| Dot (39) | **0** |

**A BERT model with zero matrix multiplications is physically impossible** — BERT is
*made* of matmuls. That impossibility was the red flag that something fundamental was
miscoded. (We also saw `logistic` apparently being the single most common op, another
absurd signal.)

### 4.4 Root cause: a custom, non-standard numbering

TPUGraphs did **not** use XLA's standard enum. When Google generated the dataset they
assigned their **own integer indices** to opcodes. The ground truth lives in our cloned
copy of their repo:
```
reference/tpu_graphs/process_data/xla/hlo_opcode.h
```
Inside, two macro lists define the mapping as explicit `V("name", id)` pairs, e.g.
`V("convolution", 26)` and `V("dot", 34)`. Their `StringToOpcodeID()` function builds
its lookup table from exactly these. So the *real* codes are **convolution = 26,
dot = 34** — shifted from the 30/39 we'd assumed.

> **"Enum shift" in plain terms:** two different pieces of software numbered the same
> list of operations differently. We were reading the integers with the wrong legend.

### 4.5 The fix: extract the mapping from source

We wrote **`scripts/extract_opcode_map.py`**, which parses every `V("name", id)` entry
out of `hlo_opcode.h` and builds the authoritative `name → integer` dictionary
(121 opcodes, ids 0–120). Run it any time to regenerate or inspect:
```
python scripts/extract_opcode_map.py          # print the full map
python scripts/extract_opcode_map.py --check    # assert convolution=26, dot=34
```

### 4.6 The corrected opcode sets (now in `dataset.py`)

```python
MXU_OPS = {26, 34}      # convolution, dot
                        # (einsum is ABSENT — XLA lowers einsum → dot)

MEM_OPS = {13, 22, 27, 35, 36, 42, 62, 75, 76, 82, 92, 98}
#          broadcast, concatenate, copy, dynamic-slice, dynamic-update-slice,
#          gather, pad, reshape, reverse, scatter, slice, transpose
```

---

## 5. Verification run (Task 3 result)

Re-running the distribution on all 69 train graphs with the **corrected** sets:

| Metric | OLD (wrong enum) | CORRECTED enum |
|---|---:|---:|
| Total Dot ops | 0 | **8,113** |
| Total Convolution ops | 4 | **11,113** |
| Graphs with ≥1 MXU op | 3 / 69 | **67 / 69** |
| BERT Dots (per graph) | 0 | **296 – 581** |

BERT now has hundreds of matmuls (sane); ResNet / MaskRCNN show hundreds of convolutions
(sane). The two graphs with zero MXU ops are tiny pure-elementwise graphs (plausible).

**Decision:** the MXU lane now carries real signal, so we **keep Option B with all three
edge types** ( `[data_flow, mxu, mem]` ) and proceed to build `model_phase3.py`.

### Example augmentation on real graphs

| Graph | N nodes | data-flow edges | → augmented edges | MXU nodes | MEM nodes |
|---|---:|---:|---:|---:|---:|
| alexnet | 372 | 597 | 1,257+ | 23 | 39 |
| bert_classifier (perf) | 20,665 | 36,659 | ~47k | 296 | 1,856 |

The edge count grows by a bounded, modest amount — exactly as the O(W·N) heuristic
promises.

---

## 6. What's in the code (file-by-file)

| File | What changed |
|---|---|
| `dataset.py` | Added `MXU_OPS`, `MEM_OPS`, `MXU_WINDOW=5`, `MEM_WINDOW=15`, `_contention_edges()`, and `add_structural_edges()`. The augmentation runs **once per graph in the cache** (it depends only on `node_opcode`, not on the config), then `edge_index` (augmented) and `edge_attr` (one-hot) are returned in every `Data` object. PyG's `Batch` stacks `edge_attr` automatically. |
| `scripts/extract_opcode_map.py` | Parses the ground-truth opcode map out of the reference repo (see §4.5). |
| `test_dataset.py` | New `test_structural_edges` asserts: (a) `edge_attr` is `(E', 3)` and row count matches `edge_index`; (b) every row is a valid one-hot summing to exactly 1.0; (c) the O(W·N) edge-count bound holds (no clique). **All 7 tests pass.** |

---

## 7. Known technical debt (must address before publication)

⚠️ **`analyze_opcodes.py` still uses the OLD, WRONG enum.** Its `COMPUTE_BOUND_OPCODES`
/ `MEMORY_BOUND_OPCODES` sets, and therefore the **Phase 0.5 "~63% compute-dominated"
figure**, are built on mislabeled opcodes. That number should **not** go into the paper
until it is re-derived with the corrected map from `scripts/extract_opcode_map.py`.
This does not block Phase 3 modeling, but it is a correctness issue for the paper's
motivation section. Tracked as **V5 / C5 in [`../VERIFICATION.md`](../VERIFICATION.md)** —
the project-wide ledger of what is human-verified vs. still AI-asserted. Read that file
before citing any result from this project.

---

## 8. Outcome (done)

**`model_phase3.py` (`HardwareAwareDiffPool`) was built and evaluated.** It swaps
`SAGEConv → GINEConv`, projects the 3-D edge types via a shared `Linear(3, hidden)`,
and coarsens edge attributes through pooling via a sparse per-channel `Sᵀ Eᶜ S`
(verified against a dense reference to 1e-7). At a matched ~234k params and 5 seeds,
it **did not beat the flat baseline** (Δτ = −0.0116, paired t-test p = 0.72). The data
foundation in this document is sound and correct; the *modeling* result is a clean
null. Full analysis: [`phase3_results.md`](phase3_results.md). The augmentation here is
**not deleted** — it remains available, and Phase 4 checks whether it matters for the
Top-K autotuner metric even though it doesn't move global τ.
