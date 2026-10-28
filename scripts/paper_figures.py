"""Paper figures and LaTeX tables, generated only from committed result files.

  python scripts/paper_figures.py

Outputs (paper/figures/, paper/tables/): FIG-K1 forest plot (lr 1e-3 and LR-tuned groups), FIG-R4 LR curves
(6e + 6f), FIG-R1 xla:default staircase and FIG-R2 xla:random paired plot (both eval batch 8), 8c arm table,
§7 robustness table, appendix eval-batch table, MDE table,
audit axis table (recomputed from the 20 per-paper score tables, then checked against the
ledger), hyperparameter table. Rerun after any result file changes; never edit outputs by hand.
"""
import glob, inspect, json, os, re, sys
import numpy as np
import scipy.stats as st
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from power_analysis import min_detectable_effect
import phase8c_evaluate

FIG, TAB = "paper/figures", "paper/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
INK, MUTED, GRID, ACCENT = "#1f2328", "#6e7781", "#d0d7de", "#c2410c"
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})


def load(path):
    return json.load(open(path))


def ci95(x):
    x = np.asarray(x)
    h = st.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


# ── FIG-K1: CONV Top-1 slowdown per arm, mean + 95% CI (t, df = n−1) + the seeds ─────────────
# Two groups: the shared lr 1e-3 arms (8c, P4 control, 8d), and the LR-tuned arms at 3e-5
# (8e channel, R5 baseline and oracle). All at eval batch 32, the registered batch (O19).
c8 = load("experiments/phase8c_results.json")
d8 = load("experiments/phase8d_results.json")
ctl = load("experiments/phase8c_control_results.json")
e8 = load("experiments/phase8e_results.json")
r5 = load("experiments/r5_results.json")
for arm in ("baseline", "channel"):   # the same runs appear in all three files
    assert c8["per_arm_seed_means"][arm]["conv"] == d8["per_arm_seed_means"][arm]["conv"] \
        == ctl["per_arm_seed_means"][arm]["conv"], arm
OLD_SEEDS = ["42", "100", "123", "456", "789"]                  # 8e's seeds, in 8e's per-seed order
r5_base = [r5["per_seed"]["b32"]["baseline"][s] for s in map(str, r5["seeds"])]
r5_orc = [r5["per_seed"]["b32"]["oracle"][s] for s in map(str, r5["seeds"])]
# 8e's channel@3e-5 per seed = its baseline (R5 reads the same 8e cache) + 8e's Q1 per-seed delta
ch_tuned = [r5["per_seed"]["b32"]["baseline"][s] + dl
            for s, dl in zip(OLD_SEEDS, e8["contrasts"]["b32:Q1:conv"]["per_seed_delta"])]
assert abs(np.mean(r5_base) - r5["means"]["b32"]["baseline"]["conv"]) < 1e-3
assert abs(np.mean(r5_orc) - r5["means"]["b32"]["oracle"]["conv"]) < 1e-3
assert abs(np.mean(ch_tuned) - np.mean([r5["per_seed"]["b32"]["baseline"][s] for s in OLD_SEEDS])
           - e8["contrasts"]["b32:Q1:conv"]["delta"]) < 1e-3
groups = [("Shared lr 1e-3 (n = 5)", [
              ("Baseline", c8["per_arm_seed_means"]["baseline"]["conv"], "control"),
              ("Scramble control", ctl["per_arm_seed_means"]["scramble"]["conv"], "control"),
              ("Full MXU", c8["per_arm_seed_means"]["full"]["conv"], "enc"),
              ("Spatial MXU", c8["per_arm_seed_means"]["spatial"]["conv"], "enc"),
              ("Channel MXU", c8["per_arm_seed_means"]["channel"]["conv"], "primary"),
              ("Oracle", d8["per_arm_seed_means"]["oracle-bucket"]["conv"], "oracle")]),
          ("Both LR-tuned, lr 3e-5", [
              ("Baseline (n = 10)", r5_base, "control"),
              ("Channel MXU (n = 5)", ch_tuned, "primary"),
              ("Oracle (n = 10)", r5_orc, "oracle")])]

n_rows = sum(len(g) + 1 for _, g in groups)
fig, ax = plt.subplots(figsize=(3.4, 3.1))
ax.axvline(1.0, color=MUTED, lw=0.6, ls=(0, (2, 2)))
y, ticks, labels = n_rows - 1, [], []
for gname, rows in groups:
    ax.text(0.952, y, gname, fontsize=7, fontweight="bold", color=INK, ha="left", va="center")
    y_top = y - 0.5
    y -= 1
    base_mean = np.mean(rows[0][1])
    ax.plot([base_mean, base_mean], [y - len(rows) + 0.6, y_top], color=GRID, lw=0.8, zorder=1)
    for label, seeds, role in rows:
        m, lo, hi = ci95(seeds)
        col = ACCENT if role == "primary" else INK
        ax.scatter(seeds, [y] * len(seeds), s=9, facecolors="none", edgecolors=MUTED, lw=0.5, zorder=2)
        ax.plot([lo, hi], [y, y], color=col, lw=1.2, solid_capstyle="round", zorder=3)
        marker = "D" if role == "oracle" else "o"
        face = "white" if role == "control" else col
        ax.scatter([m], [y], s=26, marker=marker, facecolors=face, edgecolors=col, lw=1.0, zorder=4)
        ticks.append(y); labels.append(label)
        y -= 1
ax.set_yticks(ticks)
ax.set_yticklabels(labels)
ax.set_xlabel("CONV Top-1 slowdown, eval batch 32")
ax.set_xlim(0.95, 1.56)
ax.text(1.005, -0.75, "optimal", color=MUTED, fontsize=6.5, ha="left", va="center")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_ylim(-1.0, n_rows - 0.4)
fig.tight_layout(pad=0.3)
fig.savefig(f"{FIG}/fig_k1_forest.pdf", metadata={"CreationDate": None}); fig.savefig(f"{FIG}/fig_k1_forest.png", dpi=250)
plt.close(fig)

# ── 8c arm table ───────────────────────────────────────────────────────────────────────────
def pm(x):
    return f"{np.mean(x):.4f} $\\pm$ {np.std(x, ddof=1):.4f}"


lines = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Arm & CONV slowdown & OTHER slowdown & $\Delta$ CONV & BH-$p$ & $d$ \\", r"\midrule"]
for arm in ("baseline", "channel", "spatial", "full"):
    s = c8["per_arm_seed_means"][arm]
    t = c8["primary"].get(arm)
    lines.append(f"{arm} & {pm(s['conv'])} & {pm(s['other'])} & "
                 + (f"{t['delta']:+.4f} & {t['p_bh']:.4f} & {t['d']:+.2f}" if t else "-- & -- & --") + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/phase8c_arms.tex", "w").write(
    "% Generated by scripts/paper_figures.py from experiments/phase8c_results.json\n" + "\n".join(lines) + "\n")

# ── §7 robustness table: every check the 8c result faced, registered batch 32 + batch 8 ─────
o19 = load("experiments/o19_8c8d_batch8_results.json")
r3 = load("experiments/r3_results.json")
r4 = load("experiments/r4_results.json")
r0 = load("experiments/r0_conv_top1_results.json")
CONV = phase8c_evaluate.CONV_GRAPHS


def fp(p):
    return "<.001" if p < 0.001 else (f"{p:.4f}" if p < 0.01 else f"{p:.3f}").lstrip("0")


def loo(raw):
    """Leave-one-CONV-graph-out channel − baseline (both collections dropped), per drop: (Δ, p)."""
    out = []
    for drop in sorted(CONV):
        keys = [k for k in raw["baseline"] if k.split("::")[1] in CONV and k.split("::")[1] != drop]
        b = np.array([np.mean([raw["baseline"][k][s] for k in keys]) for s in OLD_SEEDS])
        c = np.array([np.mean([raw["channel"][k][s] for k in keys]) for s in OLD_SEEDS])
        out.append(((c - b).mean(), st.ttest_rel(c, b).pvalue))
    return out


def cell(c, bh=False):
    p = c["p_bh"] if bh else c["p_t"]
    return f"{c['delta']:+.3f}", fp(p), (f"{c['mde']:.3f}" if c.get("mde") else "--")


loo32, loo8 = loo(c8["raw"]), loo({a: o19["raw"]["8"][a] for a in ("baseline", "channel")})
assert abs(min(d for d, _ in loo32) + 0.3783) < 1e-3 and abs(max(d for d, _ in loo32) + 0.2597) < 1e-3
O = lambda b, k: o19["contrasts"][f"b{b}:{k}"]
ROB = [  # (check, contrast, lr, n, b32 cell, b8 cell, status)
    ("8c primary", "channel $-$ baseline", "1e-3", "5",
     cell(O(32, "8c primary: channel − baseline:conv"), True), cell(O(8, "8c primary: channel − baseline:conv"), True), "prereg."),
    ("Leave one graph out", "channel $-$ baseline", "1e-3", "5",
     (f"{min(d for d, _ in loo32):+.3f} to {max(d for d, _ in loo32):+.3f}", r"$\le$" + fp(max(p for _, p in loo32)), "--"),
     (f"{min(d for d, _ in loo8):+.3f} to {max(d for d, _ in loo8):+.3f}", r"$\le$" + fp(max(p for _, p in loo8)), "--"), "post hoc"),
    ("P4 scramble control", "channel $-$ scramble (CPU)", "1e-3", "5",
     cell(O(32, "P4: channel − scramble:conv")), cell(O(8, "P4: channel − scramble:conv")), "prereg."),
    ("R3 matched hardware", "channel $-$ scramble (GPU)", "1e-3", "5",
     cell(r3["contrasts"]["b32:R3-P4: channel − scramble-GPU:conv"]), cell(r3["contrasts"]["b8:R3-P4: channel − scramble-GPU:conv"]), "declared"),
    ("8e LR tuning (Q1)", "channel $-$ baseline", "3e-5", "5",
     cell(e8["contrasts"]["b32:Q1:conv"]), cell(e8["contrasts"]["b8:Q1:conv"]), "prereg."),
    ("8e LR tuning (Q2)", "channel@1e-3 $-$ baseline@3e-5", "mixed", "5",
     cell(e8["contrasts"]["b32:Q2:conv"]), cell(e8["contrasts"]["b8:Q2:conv"]) if isinstance(e8["contrasts"]["b8:Q2:conv"], dict) else ("--", "--", "--"), "prereg."),
    ("R4 held-out families", "channel $-$ baseline", "3e-5", "5",
     cell(r4["contrasts"]["b32:primary"]), cell(r4["contrasts"]["b8:primary"]), "declared, expl."),
    ("8d oracle (P5)", "oracle $-$ channel", "1e-3", "5",
     cell(O(32, "8d primary: oracle − channel:conv")), cell(O(8, "8d primary: oracle − channel:conv")), "prereg."),
    ("R5 tuned oracle", "oracle $-$ baseline", "3e-5", "10",
     cell(r5["contrasts"]["b32:primary"]), cell(r5["contrasts"]["b8:primary"]), "declared"),
]
lines = [r"\begin{tabular}{llcccccccl}", r"\toprule",
         r" & & & & \multicolumn{3}{c}{Eval batch 32 (registered)} & \multicolumn{2}{c}{Eval batch 8} & \\",
         r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
         r"Check & Contrast & lr & $n$ & $\Delta$ & $p$ & Realised MDE & $\Delta$ & $p$ & Status \\", r"\midrule"]
for chk, con, lr, n, a, b, stat in ROB:
    lines.append(f"{chk} & {con} & {lr} & {n} & {a[0]} & {a[1]} & {a[2]} & {b[0]} & {b[1]} & {stat} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/capstone_robustness.tex", "w").write(
    "% Generated by scripts/paper_figures.py from o19_8c8d_batch8, phase8c, r3, phase8e, r4, r5 results JSON.\n"
    "% CONV Top-1 slowdown, paired by seed; Delta < 0 = first arm closer to optimal. 8c p is BH-adjusted\n"
    "% over the three 8c arms; others are paired t. Leave-one-graph-out: range over the 3 drops, max p.\n"
    "% Realised MDE = from each contrast's observed paired SD (alpha .05, power .8), not a preregistered MDE.\n"
    + "\n".join(lines) + "\n")

# ── Appendix: every headline contrast at eval batch 8 and 32 (O1, O8, O19, O21) ─────────────
ebd = load("experiments/eval_batch_sensitivity_xla_default.json")["contrasts"]
ebr = load("experiments/eval_batch_sensitivity_xla_random.json")["contrasts"]
e6, f6 = load("experiments/phase6e_results.json"), load("experiments/phase6f_results.json")
d6 = load("experiments/phase6d_amended_results.json")
sens = lambda c, k, b: (f"{c[f'{k}@b{b}']['mean']:+.3f}", fp(c[f"{k}@b{b}"]["p"]))
reg = lambda c: (f"{c['mean_delta']:+.3f}", fp(c["p"]))
two = lambda c: (f"{c['delta']:+.3f}", fp(c.get("p_bh", c["p_t"])))
NA = ("--", "--")
EB = [  # (section, contrast, metric, b8, b32, registered batch)
    (r"\S5", "contention-edge $-$ flat, xla:default, lr 1e-3", r"$\tau$", sens(ebd, "hardware-vs-flat", 8), sens(ebd, "hardware-vs-flat", 32), "8"),
    (r"\S5", "hierarchical $-$ flat, xla:default, lr 1e-3", r"$\tau$", sens(ebd, "hierarchical-vs-flat", 8), sens(ebd, "hierarchical-vs-flat", 32), "8"),
    (r"\S5", "hierarchical $-$ flat, xla:random, lr 1e-3", r"$\tau$", sens(ebr, "hierarchical-vs-flat", 8), sens(ebr, "hierarchical-vs-flat", 32), "8"),
    (r"\S5", "contention-edge $-$ flat, xla:random, lr 1e-3", r"$\tau$", sens(ebr, "hardware-vs-flat", 8), sens(ebr, "hardware-vs-flat", 32), "8"),
    (r"\S5", "contention-edge@1e-3 $-$ flat@LR*, xla:default (6e P1)", r"$\tau$", reg(e6["P1_hardware_minus_flat_at_lr_star"]), NA, "8"),
    (r"\S5", "contention-edge $-$ flat, both LR-tuned, xla:default (6f)", r"$\tau$", reg(f6["Q1_hardware_at_lr_star_hw_minus_flat_at_lr_star_flat"]), NA, "8"),
    (r"\S5", "contention-edge $-$ flat, lr 1e-4, nlp:default (6d amended)", r"$\tau$", reg(d6["A_P1_hardware1e-4_minus_flat1e-4"]), NA, "8"),
    (r"\S5", "hierarchical $-$ flat, CONV, lr 1e-3 (R0)", "Top-1", two(r0["contrasts"]["b8:hierarchical-flat:conv"]), two(r0["contrasts"]["b32:hierarchical-flat:conv"]), "32"),
    (r"\S5", "contention-edge $-$ flat, CONV, lr 1e-3 (R0)", "Top-1", two(r0["contrasts"]["b8:hardware-flat:conv"]), two(r0["contrasts"]["b32:hardware-flat:conv"]), "32"),
] + [(r"\S7", f"{chk}: {con}", "Top-1", b[:2], a[:2], "32") for chk, con, lr, n, a, b, stat in ROB]
lines = [r"\begin{tabular}{lllcccccc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Eval batch 8} & \multicolumn{2}{c}{Eval batch 32} & \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Section & Contrast & Metric & $\Delta$ & $p$ & $\Delta$ & $p$ & Primary \\", r"\midrule"]
for sec, con, met, b8, b32, rb in EB:
    lines.append(f"{sec} & {con} & {met} & {b8[0]} & {b8[1]} & {b32[0]} & {b32[1]} & {rb} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/appendix_eval_batch.tex", "w").write(
    "% Generated by scripts/paper_figures.py from the eval_batch_sensitivity, phase6e/6f/6d, r0 and §7 sources.\n"
    "% -- = not scored at that batch (6e/6f/6d checkpoints were evaluated at their registered batch only).\n"
    "% Primary = the batch the main text reports (registered for every preregistered or declared contrast).\n"
    + "\n".join(lines) + "\n")

# ── MDE table ──────────────────────────────────────────────────────────────────────────────
NS, SIG = [3, 5, 7, 10, 15, 20], [0.040, 0.055, 0.070]
lines = [r"\begin{tabular}{l" + "c" * len(NS) + "}", r"\toprule",
         r"$\sigma$ (paired) & " + " & ".join(f"$n={n}$" for n in NS) + r" \\", r"\midrule"]
for s in SIG:
    lines.append(f"{s:.3f} & " + " & ".join(f"{min_detectable_effect(s, n):.3f}" for n in NS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/mde.tex", "w").write(
    "% Generated by scripts/paper_figures.py via power_analysis.min_detectable_effect\n"
    "% (paired t, alpha=0.05, power=0.8)\n" + "\n".join(lines) + "\n")

# ── Audit axis table, recomputed from the per-paper score tables ──────────────────────────
doc = open("experiments/phase7_meta_analysis.md", encoding="utf-8").read()
papers = re.split(r"\n### A\d+\.", doc)[1:]
papers = [p.split("\n## ")[0] for p in papers]
scores = []
for p in papers:
    row = {}
    for m in re.finditer(r"^\| ([1-5]) [^|]*\| ([^|]*)\|", p, re.M):
        marks = re.findall(r"\*\*(P|∂|F)\*\*", m.group(2))
        row[int(m.group(1))] = marks[-1]          # a corrected cell lists the final score last
    scores.append(row)
assert len(scores) == 20 and all(len(r) == 5 for r in scores), [len(r) for r in scores]
AX = {1: "Seeds / variance", 2: "Parameter-matched", 3: "Significance test",
      4: "Eval-set integrity", 5: "Effect vs noise"}
LEDGER_F = {1: 80, 2: 85, 3: 95, 4: 20, 5: 30}   # Axis 3: 100 → 95 after the 2026-10-05 human check (A2 F → ∂)
lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Axis & Fail & Partial & Pass \\", r"\midrule"]
mismatch = []
for a, name in AX.items():
    col = [r[a] for r in scores]
    f, pa, ps = col.count("F"), col.count("∂"), col.count("P")
    if 100 * f // 20 != LEDGER_F[a]:
        mismatch.append((a, 100 * f // 20, LEDGER_F[a]))
    lines.append(f"{a}. {name} & {f}/20 & {pa}/20 & {ps}/20 \\\\")
fails3 = sum(1 for r in scores if sum(v == "F" for v in r.values()) >= 3)
pass13 = sum(1 for r in scores if all(r[a] == "P" for a in (1, 2, 3)))
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/audit_axes.tex", "w").write(
    "% Generated by scripts/paper_figures.py from the per-paper tables in experiments/phase7_meta_analysis.md\n"
    f"% papers failing >=3 axes: {fails3}/20; passing axes 1-3: {pass13}/20\n" + "\n".join(lines) + "\n")

# ── Hyperparameter table, read from the saved checkpoints' own args ───────────────────────
ck = torch.load("checkpoints/expansion/xla_default/hardware-seed42/best_model.pt", map_location="cpu", weights_only=False)
a = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
keep = ["epochs", "lr", "weight_decay", "grad_clip", "dropout", "configs_per_batch",
        "max_configs_per_graph", "eval_batch", "eval_every", "lp_weight", "entropy_weight", "hidden"]
lines = [r"\begin{tabular}{ll}", r"\toprule", r"Setting & Value \\", r"\midrule"]
label = {"eval_batch": r"eval\_batch (checkpoint selection, $\tau$)"}
lines += [f"{label.get(k, k.replace('_', r'\_'))} & {a[k]} \\\\" for k in keep if k in a]
# Eval-batch exceptions (O8, O19), read from their sources rather than typed in
sel_2c3 = {(c["args"] if isinstance(c["args"], dict) else vars(c["args"]))["eval_batch"]
           for c in (torch.load(p, map_location="cpu", weights_only=False)
                     for p in sorted(glob.glob("checkpoints/ablation/*-seed*/best_model.pt"))
                     + sorted(glob.glob("checkpoints/phase3/hardware-seed*/best_model.pt")))}
top1 = inspect.signature(phase8c_evaluate.arm_collection_slowdowns).parameters["batch_size"].default
assert len(sel_2c3) == 1, sel_2c3
lines += [r"\midrule",
          rf"eval\_batch, xla:random Phase 2c/3 selection & {sel_2c3.pop()} \\",
          rf"eval batch, CONV Top-1 scoring (\S7) & {top1} \\"]
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{TAB}/hyperparameters.tex", "w").write(
    "% Generated by scripts/paper_figures.py from checkpoints/expansion/xla_default/hardware-seed42 args;\n"
    "% eval-batch exceptions from the Phase 2c/3 checkpoint args and phase8c_evaluate.py (O8, O19)\n"
    + "\n".join(lines) + "\n")

print("figure: fig_k1_forest.pdf/png | tables: phase8c_arms, capstone_robustness, appendix_eval_batch, mde, audit_axes, hyperparameters")
print(f"audit: fail>=3 axes {fails3}/20 (ledger 17), pass axes 1-3 {pass13}/20 (ledger 0)")
print("audit %F vs ledger mismatches:", mismatch or "none")
print("hyperparameter keys missing:", [k for k in keep if k not in a] or "none")

# ── DIAG-1: a convolution's output channels on the 128-column MXU (schematic, no data) ────
# Weight matrix of a lowered conv = (reduction dim) x C_out; C_out maps to the MXU's 128 columns.
# C_out=64 fills half the columns (the rest are zero-padded, idle); C_out=256 fills two full tiles.
from matplotlib.patches import Rectangle
fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.75), gridspec_kw={"width_ratios": [1, 2]})
def tile(ax, x0, used_cols, label):
    ax.add_patch(Rectangle((x0, 0), 128, 128, fill=False, ec=INK, lw=0.8))
    ax.add_patch(Rectangle((x0, 0), used_cols, 128, fc="#57606a", ec="none"))
    if used_cols < 128:
        ax.add_patch(Rectangle((x0 + used_cols, 0), 128 - used_cols, 128, fc="white",
                               ec=MUTED, lw=0, hatch="////"))
        ax.text(x0 + used_cols + (128 - used_cols) / 2, 64, "padded\n(idle)", ha="center",
                va="center", fontsize=6.5, color=INK, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.text(x0 + 64, -12, label, ha="center", va="top", fontsize=6.5, color=MUTED)
tile(axes[0], 0, 64, "cols 0–127")
axes[0].set_title("64 output channels:\n50% of columns used", fontsize=7)
tile(axes[1], 0, 128, "cols 0–127")
tile(axes[1], 140, 128, "cols 128–255")
axes[1].set_title("256 output channels:\ntwo full tiles, 100% used", fontsize=7)
for ax, w in ((axes[0], 128), (axes[1], 268)):
    ax.set_xlim(-6, w + 6); ax.set_ylim(-34, 134); ax.set_aspect("equal"); ax.axis("off")
axes[0].annotate("", xy=(-4, 128), xytext=(-4, 0), arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
axes[0].text(-9, 64, "reduction dim", rotation=90, ha="right", va="center", fontsize=6, color=MUTED)
fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.02, wspace=0.12)
fig.savefig(f"{FIG}/diag1_mxu_tiling.pdf", metadata={"CreationDate": None}); fig.savefig(f"{FIG}/diag1_mxu_tiling.png", dpi=250)
plt.close(fig)
print("figure: diag1_mxu_tiling.pdf/png")

# ── FIG-R4: flat vs hardware val τ across the LR grid, xla:default (6e + 6f, registered §5) ─
# Mean ± sd over the 5 seeds at every LR, seeds as small dots; dashed line = the shared lr 1e-3.
e6, f6 = load("experiments/phase6e_results.json"), load("experiments/phase6f_results.json")
lrs = ["3e-5", "1e-4", "3e-4", "1e-3", "3e-3"]
assert e6["lr_star"] == f6["lr_star_flat"]
fig, ax = plt.subplots(figsize=(3.3, 2.0))
ax.axvline(1e-3, color=MUTED, lw=0.6, ls=(0, (2, 2)))
ax.text(1.08e-3, 0.005, "shared lr", color=MUTED, fontsize=6.5, ha="left", va="bottom")
for label, curve, col, mk, dx in (("Flat", e6["flat_curve"], INK, "o", 0.94),
                                   ("Contention-edge", f6["hardware_curve"], ACCENT, "s", 1.06)):
    x = np.array([float(l) for l in lrs])
    m = np.array([curve[l]["mean"] for l in lrs]); s = np.array([curve[l]["sd"] for l in lrs])
    for xi, l in zip(x, lrs):
        ax.scatter([xi * dx] * 5, curve[l]["per_seed"], s=6, facecolors="none", edgecolors=col, lw=0.4, alpha=0.6, zorder=2)
    ax.errorbar(x * dx, m, yerr=s, color=col, marker=mk, ms=3.5, lw=1.0, capsize=1.5, elinewidth=0.7, label=label, zorder=3)
ax.set_xscale("log")
ax.set_xticks([float(l) for l in lrs]); ax.set_xticklabels(lrs)
ax.minorticks_off()
ax.set_xlabel("learning rate"); ax.set_ylabel("validation Kendall's τ")
ax.set_ylim(0, 0.3)
ax.legend(frameon=False, fontsize=7, loc="upper right")
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(f"{FIG}/fig_r4_lr_curves.pdf", metadata={"CreationDate": None}); fig.savefig(f"{FIG}/fig_r4_lr_curves.png", dpi=250)
plt.close(fig)
print("figure: fig_r4_lr_curves.pdf/png")


def paired_panel(ax, arms, taus, labels):
    """Seeds as grey connected dots (one line per seed), arm mean ± sd as black diamonds."""
    vals = [np.asarray(taus[a]["8"]) for a in arms]
    x = np.arange(len(arms))
    for seed_vals in np.column_stack(vals):
        ax.plot(x, seed_vals, color=MUTED, lw=0.5, alpha=0.6, marker="o", ms=2.2, zorder=1)
    ax.errorbar(x, [v.mean() for v in vals], yerr=[v.std(ddof=1) for v in vals], fmt="D", color=INK,
                ms=4, capsize=2, elinewidth=0.8, lw=0, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.4, len(arms) - 0.6)
    ax.set_ylabel("validation Kendall's τ")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return vals


# ── FIG-R1: the xla:default staircase at the shared lr 1e-3, eval batch 8 (§5 ¶1/¶5) ─────────
# Left panel of the old phase6_xla_default_discovery.png, without its titles; the 6b
# decomposition panel moved to the appendix (option A). ¶6 / FIG-R4 test this result.
xd = load("experiments/eval_batch_sensitivity_xla_default.json")["taus"]
fig, ax = plt.subplots(figsize=(3.3, 2.0))
v = paired_panel(ax, ["wide-baseline", "hierarchical", "hardware"], xd, ["Flat", "+ Hierarchy", "+ Contention\nedges"])
assert abs(v[0].mean() - 0.0327) < 5e-4 and abs(v[2].mean() - 0.1977) < 5e-4  # Desk §5 ¶5
ax.text(0.02, 0.98, "xla:default, shared lr 1e-3", transform=ax.transAxes, color=MUTED, fontsize=6.5, ha="left", va="top")
ax.set_ylim(0, 0.25)
fig.tight_layout(pad=0.3)
fig.savefig(f"{FIG}/fig_r1_xla_default_staircase.pdf", metadata={"CreationDate": None})
fig.savefig(f"{FIG}/fig_r1_xla_default_staircase.png", dpi=250)
plt.close(fig)
print("figure: fig_r1_xla_default_staircase.pdf/png")

# ── FIG-R2: xla:random, parameter-matched flat vs DiffPool at eval batch 8 (§5 ¶2) ───────────
# Replaces experiments/figures/ablation_paired.png, which showed the batch-32 numbers (p = .22).
xr = load("experiments/eval_batch_sensitivity_xla_random.json")
fig, ax = plt.subplots(figsize=(2.4, 2.0))
v = paired_panel(ax, ["wide-baseline", "hierarchical"], xr["taus"], ["Flat", "Hierarchy\n(DiffPool)"])
c = xr["contrasts"]["hierarchical-vs-flat@b8"]
assert abs((v[1] - v[0]).mean() - c["mean"]) < 1e-9 and abs(c["p"] - 0.0702) < 5e-4  # Desk §5 ¶2
ax.text(0.02, 0.02, "xla:random", transform=ax.transAxes, color=MUTED, fontsize=6.5, ha="left", va="bottom")
ax.set_ylim(0.22, 0.335)
fig.tight_layout(pad=0.3)
fig.savefig(f"{FIG}/fig_r2_xla_random_paired.pdf", metadata={"CreationDate": None})
fig.savefig(f"{FIG}/fig_r2_xla_random_paired.png", dpi=250)
plt.close(fig)
print("figure: fig_r2_xla_random_paired.pdf/png")
