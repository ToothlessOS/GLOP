"""Worst-window picker for heatmap-guided decomposition.

Given the current closed-loop tour (as node IDs in tour order) and a
fixed AGFN edge-probability heatmap, return the start position of the
length-`L` window that is least aligned with the heatmap.

Score (per tour edge, then per window):

    m[b, t]     = 1.0 - H[ pi[b, t], pi[b, (t+1) mod N] ]
    score[b, s] = sum_{t=s}^{s+L-1} m[b, t mod N]      # cyclic wrap
    s[b]        = argmax_s score[b, s]                 # over non-excluded windows

The argmax direction is `score` (not `-score`) because `m` is the
"misalignment" and bigger `m` = worse.

All operations are vectorised over the batch dimension `B` and run in
O(B * N) per call via a cyclic cumulative-sum trick.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

# Score formula. `1 - H` is the default. `neg_log_H` emphasises edges
# the heatmap is most confidently sure are *not* in the optimal tour.
AlignMetric = Literal["1_minus_H", "neg_log_H"]


def edge_mismatch(
    pi: Tensor,
    H: Tensor,
    metric: AlignMetric = "1_minus_H",
) -> Tensor:
    """Per-edge misalignment score.

    Args:
        pi: (B, N) int64 — node IDs in tour order, closed loop.
        H:  (N, N) float — edge-probability heatmap (e.g. AGFN output).
        metric: "1_minus_H" -> m = 1 - H[edge]
                "neg_log_H" -> m = -log(H[edge])
    Returns:
        (B, N) float — `m[b, t]` is the misalignment of the edge
        `(pi[b, t], pi[b, (t+1) mod N])`.
    """
    B, N = pi.shape
    pi_next = torch.roll(pi, shifts=-1, dims=1)  # (B, N)
    h_edge = H[pi, pi_next]                      # (B, N), fancy indexing
    if metric == "1_minus_H":
        return 1.0 - h_edge
    if metric == "neg_log_H":
        return -torch.log(h_edge)
    raise ValueError(f"Unknown align metric: {metric!r}")


def worst_window(
    pi: Tensor,
    H: Tensor,
    L: int,
    exclude_visited: Tensor,
    metric: AlignMetric = "1_minus_H",
) -> Tensor:
    """Pick the least-aligned length-`L` window per instance.

    Args:
        pi: (B, N) int64 — node IDs in tour order, closed loop.
        H:  (N, N) float — edge-probability heatmap.
        L: int — window size (= reviser segment size).
        exclude_visited: (B, N) bool — True at tour positions that
            have already been revised in the current cascade stage.
            A window is excluded if it overlaps *any* visited position.
        metric: see `edge_mismatch`.
    Returns:
        (B,) int64 — the start position `s` of the worst length-`L`
        window per instance. Ties broken by the smallest `s`.
    """
    if L < 1:
        raise ValueError(f"window size L must be >= 1, got {L}")
    B, N = pi.shape
    if L > N:
        raise ValueError(f"window size L={L} larger than tour length N={N}")

    m = edge_mismatch(pi, H, metric=metric)  # (B, N)

    # Cyclic cumulative sum: pad the first L entries at the end so
    # any length-L window in [0, N) is a contiguous slice in the
    # padded array. torch.cumsum is *inclusive*, so we prepend a
    # zero column to get an exclusive cumsum where cs_excl[i] = sum
    # of m_pad[0..i-1]. Then for s in [0, N) the window covers edges
    # m[s..s+L-1] (mod N), which equals cs_excl[s+L] - cs_excl[s].
    m_pad = torch.cat([m, m[:, :L]], dim=1)                # (B, N+L)
    m_cs_incl = m_pad.cumsum(dim=1)                         # (B, N+L) inclusive
    m_cs = torch.cat(
        [torch.zeros_like(m_cs_incl[:, :1]), m_cs_incl[:, :-1]], dim=1
    )                                                       # (B, N+L) exclusive
    # window_score[b, s] = sum of m[b, s:s+L] (mod N)
    window_score = m_cs[:, L : N + L] - m_cs[:, 0:N]        # (B, N)

    # Exclude any window that overlaps a visited position.
    ex_pad = torch.cat(
        [exclude_visited, exclude_visited[:, :L]], dim=1
    )                                                       # (B, N+L)
    ex_cs_incl = ex_pad.cumsum(dim=1)                       # (B, N+L) inclusive
    ex_cs = torch.cat(
        [torch.zeros_like(ex_cs_incl[:, :1]), ex_cs_incl[:, :-1]], dim=1
    )                                                       # (B, N+L) exclusive
    window_overlap = ex_cs[:, L : N + L] - ex_cs[:, 0:N]    # (B, N) int
    window_score = window_score.masked_fill(
        window_overlap > 0, float("-inf")
    )

    # Worst (highest score) per instance. argmax returns the smallest
    # index on ties, which matches the documented behaviour.
    return window_score.argmax(dim=1)                        # (B,) int64
