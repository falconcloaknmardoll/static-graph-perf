"""
loss.py
-------
Numerically stable ListMLE ranking loss for TPUGraphs compiler configuration ranking.

Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
ListMLE (Xia et al., 2008) maximises the log-likelihood of observing the
ground-truth permutation π* under the Plackett-Luce model:

    P(π* | s) = ∏_{k=0}^{K-1}  exp(s_{π*(k)}) / Σ_{j=k}^{K-1} exp(s_{π*(j)})

    Loss = -log P(π* | s) / K     (normalised by K)

where π* ranks configurations by ascending runtime (fastest first = rank 0),
and s ∈ R^K are the model's predicted scores (higher = predicted faster).

Numerical stability
~~~~~~~~~~~~~~~~~~~
The denominator Σ exp(s_j) overflows float32 for any s_j > ~88.  The fix is
the log-sum-exp identity:

    log Σ exp(s_j)  =  max(s) + log Σ exp(s_j - max(s))

PyTorch exposes this as torch.logcumsumexp, which we apply on the reversed
score sequence to obtain the *suffix* log-sum-exp at each rank position.
"""

from __future__ import annotations

import torch


def listmle_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute the normalised ListMLE loss.

    Parameters
    ----------
    y_pred : Tensor, shape (K,)
        Model's predicted scores for K configurations.
        Higher score → model predicts a faster (lower-runtime) configuration.
    y_true : Tensor, shape (K,)
        Ground-truth execution times in nanoseconds (lower = faster).
        Used only to determine the optimal permutation π* — gradients do
        NOT flow through y_true.
    temperature : float
        Divides predicted scores before computing the loss.
        Values < 1.0 sharpen the Plackett-Luce distribution (the model is
        penalised more harshly for ranking errors); values > 1.0 soften it.
        Default 1.0 (no scaling).  Useful for warm-up: start at 2.0 and
        anneal toward 1.0 to prevent large early gradients.

    Returns
    -------
    loss : scalar Tensor
        Per-configuration average ListMLE loss (loss / K).
        Differentiable with respect to y_pred.

    Examples
    --------
    >>> scores  = torch.tensor([2.0, 0.5, 1.5])    # model predicts: config0 > config2 > config1
    >>> runtimes = torch.tensor([100.0, 50.0, 75.0]) # truth: config1 < config2 < config0
    >>> loss = listmle_loss(scores, runtimes)
    >>> loss.item()  # non-zero: model has wrong order
    """
    K = y_pred.shape[0]
    if K < 2:
        # A single configuration cannot be ranked; return a zero loss with
        # a valid gradient attachment so the optimizer step is a no-op.
        return y_pred.sum() * 0.0

    scores = y_pred / temperature  # (K,)

    # ── Step 1: Determine the ground-truth permutation π* ─────────────────
    # Sort by ascending runtime: π*(0) = fastest config (rank-1).
    # stable=True breaks ties by original position, giving a deterministic π*.
    _, order = torch.sort(y_true, descending=False, stable=True)  # (K,)
    sorted_scores = scores[order]  # (K,) — scores in optimal ranking order

    # ── Step 2: Suffix log-sum-exp via logcumsumexp on reversed sequence ───
    #
    # We need: suffix_lse[k] = log Σ_{j=k}^{K-1} exp(sorted_scores[j])
    #
    # Key identity:
    #   flip(sorted_scores)[i]           = sorted_scores[K-1-i]
    #   logcumsumexp(flipped)[i]         = log Σ_{j=0}^{i} exp(flipped[j])
    #                                    = log Σ_{j=K-1-i}^{K-1} exp(sorted_scores[j])
    #   flip(logcumsumexp(flipped))[k]   = log Σ_{j=k}^{K-1} exp(sorted_scores[j])
    #                                    = suffix_lse[k]  ✓
    #
    # torch.logcumsumexp is numerically stable by construction (it applies
    # the max-subtraction trick internally), so exp() never overflows.

    flipped    = sorted_scores.flip(0)                          # (K,)
    suffix_lse = torch.logcumsumexp(flipped, dim=0).flip(0)    # (K,)

    # ── Step 3: Assemble loss ──────────────────────────────────────────────
    # -log P(π* | s) = Σ_k [ suffix_lse[k] - sorted_scores[k] ]
    # Normalised by K so the loss magnitude is comparable across batch sizes.

    loss = (suffix_lse - sorted_scores).sum() / K
    return loss
