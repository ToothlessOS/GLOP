"""Revise a length-L window and its two half-overlaps with the SHPP reviser.

The main (worst-aligning) window `[s, s+L)` is accompanied by two
half-overlapping windows of the same size:

    main:  [s,         s + L)
    left:  [s - L/2,   s + L/2)
    right: [s + L/2,   s + 3L/2)

The boundary edges of the main window (`prev -> s` and `s+L -> next`)
are inside the half-overlap windows, so the reviser naturally
re-stitches them by changing which nodes land at the SHPP endpoints.

The caller maintains `(seed, pi)` in lockstep — we permute both with
the same `sub_tour` returned by `utils.functions.revision`.

Memory notes
------------
There are two memory concerns, addressed by separate mechanisms:

1. **Cross-window peak.** Calling the reviser once on `(B*3, L, 2)`
   pins a ~3× larger working set than three separate `(B, L, 2)`
   calls. We call the reviser **three times in sequence**, one per
   window type, so each call's intermediates are released before the
   next runs.

2. **Within-window peak.** With B = width × val_size = 1280 tours,
   L = 100, and 4× coordinate augmentation inside `revision`, a
   single reviser forward at L=100 allocates attention scores of
   shape `(4*B, n_heads=8, L, L) = (5120, 8, 100, 100)` per encoder
   layer — ~1.6 GB per layer, ~10 GB across the 6-layer encoder
   alone. On an 8 GB GPU this OOMs. We further **chunk over the
   batch dimension** with size `batch_chunk` so each forward only
   sees `batch_chunk` tours at a time. Default 64 matches the
   reviser's training-time `eval_batch_size` and keeps peak VRAM
   under ~1.5 GB.

The caller passes `opts.eval_batch_size` for `batch_chunk`; pass
`batch_chunk=None` to disable chunking.
"""

from __future__ import annotations

import contextlib
from argparse import Namespace
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from utils.functions import revision  # type: ignore[import-not-found]


def _autocast_ctx(opts: Namespace):
    """bf16 autocast around the reviser forward, mirroring the
    baseline's `_autocast_ctx` in `run_benchmark.py`. No-op on CPU
    or when `opts.use_amp` is False.
    """
    if not getattr(opts, "use_amp", True):
        return contextlib.nullcontext()
    if not str(getattr(opts, "device", "cpu")).startswith("cuda"):
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def _revise_one_window(
    seed: Tensor,
    pi: Tensor,
    s: Tensor,
    L: int,
    reviser,
    opts: Namespace,
    cost_func: Callable[[Tensor, Tensor], Tensor],
    window_offset: int,
    half: int,
    N: int,
    original_subtour: Tensor,
) -> None:
    """Run the reviser on a single window type for the full batch.

    The caller is responsible for chunking the batch if B is large.
    This helper expects the *whole* `(B, L, 2)` slice.
    """
    B = seed.shape[0]
    device = seed.device
    b_idx_l = (
        torch.arange(B, device=device).unsqueeze(-1).expand(-1, L)
    )  # (B, L)

    # 1) Build the (B, L) index of tour positions for this window type.
    starts = (s + window_offset) % N  # (B,)
    offsets = torch.arange(L, device=device)  # (L,)
    idx = (starts.unsqueeze(-1) + offsets) % N  # (B, L)

    # 2) Gather original coords and pi for every window.
    coords = seed[b_idx_l, idx]  # (B, L, 2)
    pi_old = pi[b_idx_l, idx]    # (B, L)

    # 3) Run the reviser on the (B, L, 2) batch.
    with _autocast_ctx(opts):
        new_window, _ = revision(
            opts, cost_func, reviser, coords, original_subtour
        )  # (B, L, 2) — post-gather, in the original (un-transformed) coords

    # 4) Recover the per-window permutation.
    dist = torch.cdist(new_window, coords, p=2)  # (B, L, L)
    sub_tour = dist.argmin(dim=2)  # (B, L)

    # 5) Apply the same permutation to pi.
    pi_new = pi_old.gather(1, sub_tour)

    # 6) Scatter back.
    seed[b_idx_l, idx] = new_window
    pi[b_idx_l, idx] = pi_new

    del dist, sub_tour, pi_new, new_window, coords, pi_old, idx


def revise_main_with_overlaps(
    seed: Tensor,
    pi: Tensor,
    s: Tensor,
    L: int,
    reviser,
    opts: Namespace,
    cost_func: Callable[[Tensor, Tensor], Tensor],
    batch_chunk: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Revise the main window and its two half-overlaps.

    Processes the three windows sequentially with one reviser call per
    window type. Within each window type, the batch is split into
    chunks of `batch_chunk` tours so the per-call peak memory is
    bounded by the reviser's training-time footprint.

    Args:
        seed: (B, N, 2) float — coords in tour order, closed loop.
        pi:   (B, N) int64  — node IDs in tour order, parallel to seed.
        s:    (B,) int64    — start position of the main window per instance.
        L:    int           — window size.
        reviser: nn.Module  — a loaded AttentionModel (size L).
        opts: Namespace     — argparse namespace (used for `opts.no_aug`).
        cost_func: callable — `cost_func(coords, subtour) -> Tensor`.
        batch_chunk: max tours per reviser forward call. If None or
            >= B, no chunking. Defaults to `opts.eval_batch_size` in
            the caller.
    Returns:
        (seed, pi) — both updated in place.
    """
    B, N, _ = seed.shape
    half = L // 2
    device = seed.device

    if batch_chunk is None or batch_chunk >= B:
        # Single reviser call per window type.
        original_subtour = torch.arange(L, device=device, dtype=torch.long)
        for window_offset in (0, -half, half):
            _revise_one_window(
                seed, pi, s, L, reviser, opts, cost_func,
                window_offset, half, N, original_subtour,
            )
        return seed, pi

    # Chunked path. Build one `original_subtour` and reuse for every
    # chunk + window-type combination.
    original_subtour = torch.arange(L, device=device, dtype=torch.long)
    for window_offset in (0, -half, half):
        for chunk_start in range(0, B, batch_chunk):
            chunk_end = min(chunk_start + batch_chunk, B)
            chunk_slice = slice(chunk_start, chunk_end)
            s_chunk = s[chunk_slice]
            # `_revise_one_window` mutates `seed` / `pi` in place via
            # advanced indexing — slicing gives it a view onto the
            # parent storage, so the writes are visible on the caller.
            _revise_one_window(
                seed[chunk_slice], pi[chunk_slice], s_chunk,
                L, reviser, opts, cost_func,
                window_offset, half, N, original_subtour,
            )
    return seed, pi