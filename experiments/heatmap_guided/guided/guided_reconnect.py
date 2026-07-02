"""Top-level heatmap-guided cascade driver.

Mirrors the structure of `utils/functions.py:reconnect` but replaces
the even-chunk `LCP_TSP` loop with the heatmap-guided window picker.
The key invariant: `pi` is maintained in lockstep with `seed` — whenever
we permute a window of `seed` we apply the same permutation to `pi`, so
the heatmap lookup stays correct on the next iteration.
"""

from __future__ import annotations

import time
from argparse import Namespace
from typing import Callable, List, Tuple

import torch
from torch import Tensor

from .align import worst_window
from .heatmap import infer_heatmap
from .overlapping_windows import revise_main_with_overlaps


def closed_loop_cost(seed: Tensor) -> Tensor:
    """Closed-loop tour length. (B,)."""
    seg = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1)
    closing = (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
    return seg + closing


def guided_reconnect(
    get_cost_func: Callable[[Tensor, Tensor], Tensor],
    batch: Tensor,
    pi: Tensor,
    opts: Namespace,
    revisers: list,
    agfn_model,
) -> Tuple[Tensor, Tensor, Tensor, List[dict]]:
    """Run the full heatmap-guided cascade on a batch of tours.

    Args:
        get_cost_func: callable (coords, subtour) -> cost (used by
            `utils.functions.revision`).
        batch: (B, N, 2) float — coords in tour order (warm start).
        pi:    (B, N) int64    — node IDs in tour order, parallel to batch.
        opts: argparse namespace. Reads `opts.revision_lens` and
            `opts.revision_iters`; uses `opts.no_aug` and `opts.eval_batch_size`.
        revisers: list of loaded AttentionModel instances, one per
            entry in `opts.revision_lens`.
        agfn_model: a loaded AGFN Net.
    Returns:
        (seed, pi, cost, history) where:
            seed    : (B, N, 2) — revised tour coords
            pi      : (B, N)    — revised tour node IDs
            cost    : (B,)      — closed-loop tour cost per instance
            history : list of {iter, stage, cost_mean, cost_std, t}
    """
    seed = batch
    B, N, _ = seed.shape
    history: List[dict] = []

    # 1) heatmap is fixed per instance — compute once. AGFN is
    #    invariant under tour permutation, so we can re-use it across
    #    every iteration.
    #    We pass seed[0] (the first instance's coords) — every instance
    #    in a batch has the same N but may have different coords; for
    #    the experiment we run one instance at a time so this is fine.
    #    For a real batched run, we'd need one AGFN forward per
    #    instance or to stack them with PyG batching.
    H = infer_heatmap(agfn_model, seed[0])  # (N, N)

    for stage_id, (reviser_size, n_iter) in enumerate(
        zip(opts.revision_lens, opts.revision_iters)
    ):
        # The whole cascade is pure inference — no gradients needed.
        # Disabling autograd is the dominant memory fix: every reviser
        # forward + gather would otherwise stay attached to the autograd
        # graph and accumulate ~O(iters * L * embed_dim) of graph nodes.
        print(
            f"    [cascade] stage {stage_id}/{len(opts.revision_lens)-1} "
            f"start: L={reviser_size}, n_iter={n_iter}",
            flush=True,
        )
        with torch.no_grad():
            exclude = torch.zeros(
                B, N, dtype=torch.bool, device=seed.device
            )

            for it in range(n_iter):
                t0 = time.time()

                # 2) pick the worst (least-aligned) length-`reviser_size` window
                s = worst_window(
                    pi, H, L=reviser_size, exclude_visited=exclude
                )

                # 3) revise [s, s+L) + half-overlap [s-L/2, s+L/2) + [s+L/2, s+3L/2)
                #    Chunk the reviser forward over the batch so peak
                #    VRAM stays bounded (the reviser allocates ~1.6 GB
                #    of attention scores per encoder layer at the
                #    benchmark's B=1280 / L=100).
                seed, pi = revise_main_with_overlaps(
                    seed, pi, s=s, L=reviser_size,
                    reviser=revisers[stage_id], opts=opts, cost_func=get_cost_func,
                    batch_chunk=opts.eval_batch_size,
                )

                # 4) mark the three windows as visited for this cascade stage
                #    Vectorised: build (B, 3, L) index tensor of the three
                #    windows and scatter into `exclude` in one shot. Avoids
                #    the per-instance `int(s[bi].item())` CPU sync.
                half = reviser_size // 2
                starts = torch.stack(
                    [s % N, (s - half) % N, (s + half) % N], dim=1
                )  # (B, 3)
                offsets = torch.arange(reviser_size, device=seed.device)
                idx = (starts.unsqueeze(-1) + offsets) % N  # (B, 3, L)
                b_idx = (
                    torch.arange(B, device=seed.device)
                    .view(-1, 1, 1)
                    .expand(-1, 3, reviser_size)
                )  # (B, 3, L)
                exclude[b_idx, idx] = True

                cost = closed_loop_cost(seed)
                history.append(
                    {
                        "iter": len(history),
                        "stage": stage_id,
                        "reviser_size": reviser_size,
                        "cost": cost.detach().cpu().tolist(),
                        "t": time.time() - t0,
                    }
                )
                # Per-iter progress to match `glop_reconnect_with_history`
                # byte-for-byte (including the `flush=True` requirement
                # when stdout is piped via nohup/tee). `done` is
                # 1-indexed across the entire cascade (matches the
                # baseline), and `eta` is a running-mean-per-stage so
                # the first iter doesn't inflate the estimate.
                #
                # `cost_ori` reports best-of-width per instance (averaged),
                # but the cascade runs on all (width*val_size) tours so a
                # naive `cost.mean()` mixes bad warm-starts with good ones.
                # Show both: `cost` (best, comparable to `cost_ori`) and
                # `mean` (true mean over all warm-starts).
                done = len(history)
                total = sum(opts.revision_iters)
                mean_cost = cost.mean().item()
                if opts.width > 1:
                    n_inst = cost.size(0) // opts.width
                    cost_best = (
                        cost.reshape(n_inst, opts.width).min(1).values.mean().item()
                    )
                else:
                    cost_best = mean_cost
                dt = history[-1]["t"]
                avg_dt = sum(
                    h["t"] for h in history if h["stage"] == stage_id
                ) / (it + 1)
                eta = avg_dt * (n_iter - (it + 1))
                print(
                    f"    [cascade] stage {stage_id}/{len(opts.revision_lens)-1} "
                    f"(L={reviser_size}) iter {it+1}/{n_iter}  "
                    f"global {done}/{total}  "
                    f"cost={cost_best:.4f}  mean={mean_cost:.4f}  "
                    f"dt={dt:.2f}s  eta={eta:.0f}s",
                    flush=True,
                )
            # End of stage: drop the exclude mask and any stage-local
            # tensors so the next stage starts with a clean slate.
            # Without this, the PyTorch CPU caching allocator keeps
            # the (B, N) bool tensor and the per-iteration cost slices
            # around for re-use, which compounds across the three
            # stages.
            del exclude, cost
        # Force Python GC + an empty-cache between stages. CPU tensors
        # don't have a CUDA caching allocator to drain, but `gc.collect`
        # releases intermediate references that the reviser forwards
        # leave behind via local closures.
        import gc  # local import: gc is only needed here
        gc.collect()

    # final cost
    cost = closed_loop_cost(seed)
    return seed, pi, cost, history
