"""Tests for the heatmap-guided decomposition modules.

These tests do NOT require a GPU, AGFN checkpoints, or GLOP revisers.
They exercise:

  - edge_mismatch   : the per-tour-edge scoring formula
  - worst_window    : the cyclic-cumsum picker + exclude mask
  - revise_main_with_overlaps : the (seed, pi) lockstep invariant
  - guided_reconnect: end-to-end with a mock reviser
  - memory regression: cascade iteration does not accumulate the
    autograd graph (the bug that caused the benchmark to OOM)

The mock reviser is a tiny nn.Module that, given a window of L
coordinates, returns the same coordinates in a known permutation.
This is enough to test the algorithmic plumbing without any pretrained
weights.
"""

from __future__ import annotations

import argparse
import gc
import resource
from typing import Tuple

import torch
from torch import Tensor, nn

from experiments.heatmap_guided.guided.align import edge_mismatch, worst_window


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

class MockReviser(nn.Module):
    """A minimal stand-in for `AttentionModel` that just flips each window.

    On a call `model(window, return_pi=True)` it returns:
        cost   : (1,) — always 0
        pi1    : (1, L) — identity permutation
        cost2  : (1,) — always 0
        pi2    : (1, L) — reversed permutation
    The `revision` wrapper inside `utils.functions.revision` takes the
    minimum-cost of these and returns a `gather` over the input coords.
    Since both permutations produce a valid window, the cost is 0
    either way and the `original_subtour` fallback does not apply.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor, return_pi: bool = False, **kwargs):
        L = x.size(1)
        # The reviser interface used in `revision` returns 4 values
        # (cost, pi, cost2, pi2) when return_pi=True.
        cost1 = torch.zeros(x.size(0), device=x.device)
        pi1 = torch.arange(L, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        cost2 = torch.zeros(x.size(0), device=x.device)
        pi2 = torch.arange(L - 1, -1, -1, device=x.device).unsqueeze(0).expand(
            x.size(0), -1
        )
        return cost1, pi1, cost2, pi2


class _MockProblem:
    NAME = "local"

    @staticmethod
    def get_costs(dataset, pi, return_local: bool = False):
        # Return zeros — the cost fallback in `revision` only checks
        # the sign of `reduced_cost`, so any constant works.
        B = dataset.size(0)
        return torch.zeros(B, device=dataset.device), None


class _MockModel(nn.Module):
    """Mock `AttentionModel` with the attributes `revision` expects."""

    def __init__(self):
        super().__init__()
        self.problem = _MockProblem()
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def forward(self, x, return_pi=False, return_embedding=False, embeddings=None, **kwargs):
        return MockReviser().forward(x, return_pi=return_pi)


def _make_opts(no_aug: bool = True) -> argparse.Namespace:
    return argparse.Namespace(no_aug=no_aug, eval_batch_size=1, revision_lens=[20], revision_iters=[1])


# ---------------------------------------------------------------------------
# align: edge_mismatch
# ---------------------------------------------------------------------------

def test_edge_mismatch_shape_and_sign():
    B, N = 2, 6
    pi = torch.arange(N).unsqueeze(0).repeat(B, 1)  # (B, N) identity
    # H is symmetric with H[i, j] = 0.5 for all i != j, 0 on diag
    H = torch.full((N, N), 0.5)
    H.fill_diagonal_(0.0)
    m = edge_mismatch(pi, H)
    assert m.shape == (B, N)
    # All m should equal 1 - 0.5 = 0.5
    assert torch.allclose(m, torch.full((B, N), 0.5))


def test_edge_mismatch_picks_low_heat_edges():
    """If one edge has lower H than its neighbours, its mismatch is higher."""
    _B, N = 1, 5
    pi = torch.arange(N).unsqueeze(0)  # edges are (0,1),(1,2),(2,3),(3,4),(4,0)
    H = torch.full((N, N), 0.5)
    H.fill_diagonal_(0.0)
    # Make the edge (2,3) very low-heat
    H[2, 3] = 0.05
    H[3, 2] = 0.05
    m = edge_mismatch(pi, H)
    # The edge (2,3) is at position 2 in the tour
    assert m[0, 2] > m[0, 0]


# ---------------------------------------------------------------------------
# align: worst_window
# ---------------------------------------------------------------------------

def test_worst_window_picks_min_alignment():
    B, N, L = 1, 10, 3
    pi = torch.arange(N).unsqueeze(0)
    H = torch.full((N, N), 0.9)
    H.fill_diagonal_(0.0)
    # Make window starting at s=4 (edges 4,5,6) the worst
    for a, b in [(4, 5), (5, 6), (6, 7)]:
        H[a, b] = 0.1
        H[b, a] = 0.1
    exclude = torch.zeros(B, N, dtype=torch.bool)
    s = worst_window(pi, H, L=L, exclude_visited=exclude)
    assert s.tolist() == [4]


def test_worst_window_cyclic_wrap():
    """A misaligned window that wraps around the tour boundary should be picked."""
    B, N, L = 1, 8, 3
    pi = torch.arange(N).unsqueeze(0)  # edges: (0,1),...,(7,0)
    H = torch.full((N, N), 0.9)
    H.fill_diagonal_(0.0)
    # Make window starting at s=6 (edges 6,7,0) the worst
    for a, b in [(6, 7), (7, 0), (0, 1)]:
        H[a, b] = 0.1
        H[b, a] = 0.1
    exclude = torch.zeros(B, N, dtype=torch.bool)
    s = worst_window(pi, H, L=L, exclude_visited=exclude)
    assert s.tolist() == [6]


def test_exclude_visited_skips_used():
    B, N, L = 1, 10, 3
    pi = torch.arange(N).unsqueeze(0)
    H = torch.full((N, N), 0.9)
    H.fill_diagonal_(0.0)
    # Make window starting at s=4 the worst
    for a, b in [(4, 5), (5, 6), (6, 7)]:
        H[a, b] = 0.1
        H[b, a] = 0.1
    exclude = torch.zeros(B, N, dtype=torch.bool)

    s1 = worst_window(pi, H, L=L, exclude_visited=exclude)
    assert s1.tolist() == [4]

    # Mark positions 4..6 as visited
    exclude[0, 4:7] = True
    # Now the same window would still be selected if not for the mask.
    # Add a second-bad window at s=7 so the algorithm has somewhere
    # else to go.
    for a, b in [(7, 8), (8, 9), (9, 0)]:
        H[a, b] = 0.2
        H[b, a] = 0.2
    s2 = worst_window(pi, H, L=L, exclude_visited=exclude)
    assert s2.tolist() != [4]  # masked off
    assert s2.tolist() == [7]


# ---------------------------------------------------------------------------
# overlapping_windows: pi lockstep + boundary coverage
# ---------------------------------------------------------------------------

def _make_seed_and_pi(N: int) -> Tuple[Tensor, Tensor]:
    coords = torch.rand(N, 2)
    pi = torch.arange(N)
    return coords.unsqueeze(0), pi.unsqueeze(0)  # (1, N, 2), (1, N)


def test_pi_lockstep_after_revision():
    """After a (mock) revision, every position's (coord, node_id) pair
    must match a position that was either at that slot before or
    inside the revised window."""
    from experiments.heatmap_guided.guided.overlapping_windows import (
        revise_main_with_overlaps,
    )

    N, L = 30, 10
    seed, pi = _make_seed_and_pi(N)
    s = torch.tensor([5])
    opts = _make_opts()
    reviser = _MockModel()
    def get_cost_func(input, pi):
        return torch.zeros(input.size(0), device=input.device)

    seed_new, pi_new = revise_main_with_overlaps(
        seed.clone(), pi.clone(), s=s, L=L, reviser=reviser, opts=opts, cost_func=get_cost_func
    )

    # The multiset of (coord, node_id) pairs must be preserved
    before_pairs = set(
        (round(c[0].item(), 6), round(c[1].item(), 6), int(n.item()))
        for c, n in zip(seed[0], pi[0])
    )
    after_pairs = set(
        (round(c[0].item(), 6), round(c[1].item(), 6), int(n.item()))
        for c, n in zip(seed_new[0], pi_new[0])
    )
    assert before_pairs == after_pairs


def test_overlapping_windows_covers_boundary():
    """Given a known tour and main-window start s=10 with L=10, the
    three windows are [10,20), [5,15), [15,25). The two boundary
    edges of the main (9->10 and 20->21) are inside the half-overlaps
    at the last and first positions respectively.

    `revision` is called three times — once per window type
    (main, left half-overlap, right half-overlap) — with batch shape
    (B, L, 2). We assert the calls happened in that order and each
    call's input starts at the expected tour position."""
    from experiments.heatmap_guided.guided.overlapping_windows import (
        revise_main_with_overlaps,
    )

    N, L = 30, 10
    seed, pi = _make_seed_and_pi(N)
    s = torch.tensor([10])
    opts = _make_opts()
    reviser = _MockModel()
    def get_cost_func(input, pi):
        return torch.zeros(input.size(0), device=input.device)

    # Inspect the windows the function would have revised by
    # monkey-patching `revision` to record its inputs.
    called_inputs = []

    import experiments.heatmap_guided.guided.overlapping_windows as ow

    real_revision = ow.revision

    def spy(opts, cost_func, reviser, decomposed_seeds, original_subtour, **kwargs):
        called_inputs.append(decomposed_seeds.clone())
        return real_revision(opts, cost_func, reviser, decomposed_seeds, original_subtour, **kwargs)

    ow.revision = spy
    try:
        revise_main_with_overlaps(
            seed.clone(), pi.clone(), s=s, L=L, reviser=reviser, opts=opts, cost_func=get_cost_func
        )
    finally:
        ow.revision = real_revision

    # Three sequential `revision` calls, one per window type. Each
    # call's batch shape is (B, L, 2) = (1, L, 2) when B=1.
    assert len(called_inputs) == 3
    for c in called_inputs:
        assert c.shape == (1, L, 2)

    # Expected window starts in order:
    #   main  : 10          (window_offset = 0)
    #   left  : 10 - L//2 = 5    (window_offset = -L//2)
    #   right : 10 + L//2 = 15   (window_offset = +L//2)
    expected_starts = [10, 5, 15]
    for call_idx, expected_pos in enumerate(expected_starts):
        # Each call's coords are (1, L, 2); the first coord of row 0
        # must equal the coord at `expected_pos` in the original seed.
        assert torch.allclose(
            called_inputs[call_idx][0, 0], seed[0, expected_pos]
        ), f"call {call_idx} should start at tour position {expected_pos}"


def test_revise_main_with_overlaps_chunked_matches_unchunked():
    """The chunked path must produce the same result as the unchunked
    path. Without this guarantee, the memory fix would silently change
    the algorithm whenever B > batch_chunk."""
    from experiments.heatmap_guided.guided.overlapping_windows import (
        revise_main_with_overlaps,
    )

    torch.manual_seed(0)
    B, N, L = 12, 30, 10
    coords = torch.rand(B, N, 2)
    pi0 = torch.randperm(N).unsqueeze(0).repeat(B, 1)

    s = torch.tensor([5, 12, 7, 25, 1, 8, 18, 3, 22, 15, 28, 11])
    opts = _make_opts()
    reviser = _MockModel()

    def get_cost_func(input, pi):
        return torch.zeros(input.size(0), device=input.device)

    seed_a, pi_a = revise_main_with_overlaps(
        coords.clone(), pi0.clone(), s=s, L=L,
        reviser=reviser, opts=opts, cost_func=get_cost_func,
        batch_chunk=None,
    )
    seed_b, pi_b = revise_main_with_overlaps(
        coords.clone(), pi0.clone(), s=s, L=L,
        reviser=reviser, opts=opts, cost_func=get_cost_func,
        batch_chunk=4,  # smaller than B → forces chunking
    )

    assert torch.allclose(seed_a, seed_b), "chunked coords differ from unchunked"
    assert torch.equal(pi_a, pi_b), "chunked pi differs from unchunked"


# ---------------------------------------------------------------------------
# guided_reconnect: end-to-end with a mock reviser
# ---------------------------------------------------------------------------

def test_guided_reconnect_improves_or_equals():
    """Running guided_reconnect on a single instance with a mock reviser
    must produce a tour whose cost is <= the warm-start cost (because
    `revision` falls back to the original when the reviser is worse)."""
    from experiments.heatmap_guided.guided.guided_reconnect import guided_reconnect

    torch.manual_seed(0)
    N = 50
    coords = torch.rand(1, N, 2)
    pi0 = torch.arange(N).unsqueeze(0)
    cost_ori = (
        (coords[:, 1:] - coords[:, :-1]).norm(p=2, dim=2).sum(1)
        + (coords[:, 0] - coords[:, -1]).norm(p=2, dim=1)
    )

    reviser = _MockModel()
    opts = argparse.Namespace(
        revision_lens=[20],
        revision_iters=[2],
        no_aug=True,
        eval_batch_size=1,
    )

    # AGFN is a no-op for this test: we just need it to return a
    # matrix with H[i, j] in (0, 1] for the k_sparse nearest neighbours
    # of i. Use a uniform heatmap.
    class MockAGFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.N = N

        def forward(self, pyg):
            k = pyg.edge_index.size(1)
            return torch.full((k,), 0.5, device=pyg.x.device)

    agfn = MockAGFN()

    # Monkey-patch `infer_heatmap` to return a uniform matrix directly
    import experiments.heatmap_guided.guided.guided_reconnect as gr

    real_infer = gr.infer_heatmap

    def fake_infer(model, coords, k_sparse=None):
        # Dense (N, N) uniform
        H = torch.full((N, N), 0.5)
        H.fill_diagonal_(0.0)
        return H

    gr.infer_heatmap = fake_infer
    try:
        def get_cost_func(input, pi):
            return torch.zeros(input.size(0), device=input.device)
        # `revision` calls `reviser.problem.get_costs(decomposed_seeds, sub_tour)`
        # — make sure the mock problem returns zero cost.
        seed, pi, cost, history = guided_reconnect(
            get_cost_func=get_cost_func,
            batch=coords.clone(),
            pi=pi0.clone(),
            opts=opts,
            revisers=[reviser],
            agfn_model=agfn,
        )
    finally:
        gr.infer_heatmap = real_infer

    # The cost is the closed-loop cost; the mock reviser returns the
    # same coords (either identity or reverse), so the cost may stay
    # the same or drop (reverse on random coords is unlikely to be
    # better, but the fallback to original_subtour keeps it the same
    # at worst). In any case, the lockstep invariant is preserved.
    final_cost = (
        (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1)
        + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
    )
    assert final_cost.item() <= cost_ori.item() + 1e-5
    # History was recorded
    assert len(history) == 2


def test_worst_window_tie_breaks_smallest_s():
    B, N, L = 1, 8, 2
    pi = torch.arange(N).unsqueeze(0)
    H = torch.full((N, N), 0.5)
    H.fill_diagonal_(0.0)
    # Tie: every window has the same score (m = 0.5 for every edge)
    exclude = torch.zeros(B, N, dtype=torch.bool)
    s = worst_window(pi, H, L=L, exclude_visited=exclude)
    assert s.tolist() == [0]  # argmax returns the smallest index on ties


# ---------------------------------------------------------------------------
# AMP wiring: bf16 mixed precision is enabled by default on CUDA, but is a
# no-op on CPU and when `opts.use_amp=False`. These tests guard the
# `_autocast_ctx` helper in `run_benchmark.py` and the matching helper in
# `guided/overlapping_windows.py`.
# ---------------------------------------------------------------------------

def test_autocast_ctx_is_nullcontext_on_cpu():
    """CPU runs must not enter bf16 autocast — there's no CUDA bf16
    matmul kernel on CPU."""
    from experiments.heatmap_guided.run_benchmark import _autocast_ctx

    opts = argparse.Namespace(use_amp=True, device="cpu")
    ctx = _autocast_ctx(opts)
    assert ctx.__class__.__name__ == "nullcontext"


def test_autocast_ctx_respects_use_amp_flag():
    """`--no_amp` must disable autocast regardless of device."""
    from experiments.heatmap_guided.run_benchmark import _autocast_ctx

    opts = argparse.Namespace(use_amp=False, device="cuda")
    ctx = _autocast_ctx(opts)
    assert ctx.__class__.__name__ == "nullcontext"


# ---------------------------------------------------------------------------
# Memory regression: the autograd graph must NOT accumulate across cascade
# iterations. Before the fix, every reviser forward + gather stayed attached
# to the graph, so RSS grew ~linearly with iteration count and the
# benchmark OOM'd at TSP-500 / width=10 / val_size=128 within a few stages.
# ---------------------------------------------------------------------------

def _rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def test_guided_reconnect_does_not_accumulate_graph():
    """Run guided_reconnect for many iterations and verify that:
       1. RSS does not grow unboundedly (within a generous tolerance).
       2. The reviser's parameter grads are None (we never recorded them).
    """
    from experiments.heatmap_guided.guided.guided_reconnect import guided_reconnect

    torch.manual_seed(0)
    N = 60
    B = 4
    coords = torch.rand(B, N, 2)
    pi0 = torch.arange(N).unsqueeze(0).repeat(B, 1)

    reviser = _MockModel()
    # 25 iterations: enough to expose the autograd-graph leak if it
    # is still present, while staying fast.
    opts = argparse.Namespace(
        revision_lens=[20],
        revision_iters=[25],
        no_aug=True,
        eval_batch_size=B,
    )

    class MockAGFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.N = N

        def forward(self, pyg):
            k = pyg.edge_index.size(1)
            return torch.full((k,), 0.5, device=pyg.x.device)

    agfn = MockAGFN()

    # Force a GC + RSS baseline.
    gc.collect()
    rss_before = _rss_kb()

    # Monkey-patch `infer_heatmap` to skip the AGFN plumbing.
    import experiments.heatmap_guided.guided.guided_reconnect as gr

    real_infer = gr.infer_heatmap

    def fake_infer(model, coords, k_sparse=None):
        H = torch.full((N, N), 0.5)
        H.fill_diagonal_(0.0)
        return H

    gr.infer_heatmap = fake_infer
    try:
        def get_cost_func(input, pi):
            return torch.zeros(input.size(0), device=input.device)
        seed, pi, cost, history = guided_reconnect(
            get_cost_func=get_cost_func,
            batch=coords.clone(),
            pi=pi0.clone(),
            opts=opts,
            revisers=[reviser],
            agfn_model=agfn,
        )
    finally:
        gr.infer_heatmap = real_infer

    gc.collect()
    rss_after = _rss_kb()

    # The dominant memory fix is `torch.no_grad()` around the cascade.
    # If the leak came back, RSS would grow ~megabytes per iteration.
    # We allow generous headroom for transient allocations but assert
    # the growth is bounded.
    growth_kb = rss_after - rss_before
    # 25 iterations of B=4 should easily fit under 64 MB of growth
    # once the graph no longer accumulates. Before the fix this would
    # climb well past 200 MB.
    assert growth_kb < 64 * 1024, (
        f"guided_reconnect RSS grew by {growth_kb} KB over 25 iterations — "
        f"autograd graph may be accumulating. (before={rss_before}, "
        f"after={rss_after})"
    )

    # The history should have one entry per iteration.
    assert len(history) == 25

    # Final tour cost must be <= the warm-start cost (the reviser
    # falls back when its proposal is worse).
    cost_ori = (
        (coords[:, 1:] - coords[:, :-1]).norm(p=2, dim=2).sum(1)
        + (coords[:, 0] - coords[:, -1]).norm(p=2, dim=1)
    )
    cost_final = (
        (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1)
        + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
    )
    assert (cost_final <= cost_ori + 1e-5).all()


def test_glop_reconnect_with_history_does_not_accumulate_graph():
    """Run the instrumented baseline cascade for many iterations and verify
    that RSS does not grow unboundedly. This guards against the same
    autograd-graph-leak class of bug the guided test covers, but for
    the baseline arm of `experiments/heatmap_guided/run_benchmark.py`.

    Before the fix, every reviser forward + gather stayed attached to
    the autograd graph, so RSS grew ~linearly with iteration count and
    the benchmark OOM'd at TSP-500 / width=10 / val_size=128 within a
    few stages. The fix wraps the cascade in `torch.no_grad()`.
    """
    from experiments.heatmap_guided.run_benchmark import (
        closed_loop_cost,
        glop_reconnect_with_history,
    )

    torch.manual_seed(0)
    N = 60
    B = 4
    coords = torch.rand(B, N, 2)

    reviser = _MockModel()
    # 25 iterations: enough to expose the autograd-graph leak if it
    # is still present, while staying fast.
    opts = argparse.Namespace(
        revision_lens=[20],
        revision_iters=[25],
        no_aug=True,
        no_prune=True,
        eval_batch_size=B,
        device="cpu",
    )

    # Force a GC + RSS baseline.
    gc.collect()
    rss_before = _rss_kb()

    def get_cost_func(input, pi):
        return torch.zeros(input.size(0), device=input.device)

    seed_b_final, cost_final, history = glop_reconnect_with_history(
        get_cost_func=get_cost_func,
        batch=coords.clone(),
        opts=opts,
        revisers=[reviser],
    )

    gc.collect()
    rss_after = _rss_kb()

    # The dominant memory fix is `torch.no_grad()` around the cascade.
    # If the leak came back, RSS would grow ~megabytes per iteration.
    # We allow generous headroom for transient allocations but assert
    # the growth is bounded.
    growth_kb = rss_after - rss_before
    # 25 iterations of B=4 should easily fit under 64 MB of growth
    # once the graph no longer accumulates. Before the fix this would
    # climb well past 200 MB.
    assert growth_kb < 64 * 1024, (
        f"glop_reconnect_with_history RSS grew by {growth_kb} KB over "
        f"25 iterations — autograd graph may be accumulating. "
        f"(before={rss_before}, after={rss_after})"
    )

    # The history should have one entry per iteration.
    assert len(history) == 25

    # Final tour cost must be <= the warm-start cost (the reviser
    # falls back when its proposal is worse).
    cost_ori = closed_loop_cost(coords)
    cost_final_closed = closed_loop_cost(seed_b_final)
    assert (cost_final_closed <= cost_ori + 1e-5).all()


def test_glop_reconnect_with_history_width_prune_with_width_gt_one():
    """The width-prune at the end of the cascade must group by
    `width` (one row per instance, all warm-starts in a row), not
    by `eval_batch_size`. The earlier code used `eval_batch_size`
    as the reshape row size, which only works when
    `eval_batch_size == val_size` (the original main.py
    DataLoader-chunked context). With the new default
    `eval_batch_size=640` and `val_size=128, width=10` the reshape
    would fail; with `width=1` the prune was a no-op and the bug
    was hidden. This test uses `width=2, val_size=3, eval_batch_size=8`
    so that `B = 6` and the prune reshape is non-trivial but
    `B % eval_batch_size != 0`, which is the failure mode the
    earlier code had.

    The cascade itself is a no-op (MockReviser identity), so the
    only thing under test is the prune shape logic.
    """
    from experiments.heatmap_guided.run_benchmark import (
        glop_reconnect_with_history,
    )

    torch.manual_seed(0)
    width, val_size, N = 2, 3, 30
    B = width * val_size  # 6
    coords = torch.rand(B, N, 2)

    reviser = _MockModel()
    opts = argparse.Namespace(
        revision_lens=[20],
        revision_iters=[1],
        width=width,
        val_size=val_size,
        no_aug=True,
        no_prune=False,  # enable the prune
        eval_batch_size=8,  # 6 % 8 != 0 — the earlier bug would fail here
        device="cpu",
    )

    def get_cost_func(input, pi):
        return torch.zeros(input.size(0), device=input.device)

    seed_b_final, cost_final, history = glop_reconnect_with_history(
        get_cost_func=get_cost_func,
        batch=coords.clone(),
        opts=opts,
        revisers=[reviser],
    )

    # After the prune, the output should have one tour per instance.
    assert seed_b_final.shape == (val_size, N, 2), (
        f"expected (val_size={val_size}, N={N}, 2), got {tuple(seed_b_final.shape)}"
    )
    assert cost_final.shape == (val_size,), (
        f"expected cost shape ({val_size},), got {tuple(cost_final.shape)}"
    )


# ---------------------------------------------------------------------------
# UX-layer helpers: --smoke overrides, CPU/AMP warning, and the per-iter
# progress prints inside `glop_reconnect_with_history`. These guard the
# "stuck at the baseline cascade" issue — without progress output the user
# cannot tell whether the cascade is alive, hung, or simply slow on CPU.
# ---------------------------------------------------------------------------


def test_glop_reconnect_with_history_prints_progress(capsys):
    """Each LCP_TSP call must emit a single progress line so the user
    can see the cascade is alive. Includes stage id, iter within
    stage, total iter count, mean cost, dt, and an ETA. The header
    line ("stage ... start") is also asserted.

    Uses the same `_MockModel` reviser and `argparse.Namespace` opts
    pattern as the rest of this test file; no GPU, checkpoints, or
    AGFN models required.
    """
    from experiments.heatmap_guided.run_benchmark import (
        glop_reconnect_with_history,
    )

    torch.manual_seed(0)
    N, B = 30, 2
    coords = torch.rand(B, N, 2)
    reviser = _MockModel()
    opts = argparse.Namespace(
        revision_lens=[20],
        revision_iters=[3],
        no_aug=True,
        no_prune=True,
        eval_batch_size=B,
        device="cpu",
    )

    def get_cost_func(input, pi):
        return torch.zeros(input.size(0), device=input.device)

    glop_reconnect_with_history(
        get_cost_func=get_cost_func,
        batch=coords.clone(),
        opts=opts,
        revisers=[reviser],
    )
    captured = capsys.readouterr().out
    # One header line ("stage 0/0 start") + one line per iter = 4 lines
    assert captured.count("[cascade]") == 1 + 3
    # Per-iter line contains the expected substrings
    for needle in [
        "stage 0/0",
        "iter 1/3",
        "iter 2/3",
        "iter 3/3",
        "cost=",
        "dt=",
        "eta=",
    ]:
        assert needle in captured, f"missing {needle!r} in {captured!r}"


def test_apply_smoke_overrides():
    """`_apply_smoke_overrides` must collapse the production cascade
    to the fast recipe (L=100, n_iter=2, width=1, --no_aug,
    val/eval<=8) when --smoke is set, and be a strict no-op otherwise.

    The apply-path test is the one a regression would actually break:
    a flag whose name means "fast recipe" must produce a fast recipe.
    """
    from experiments.heatmap_guided.run_benchmark import (
        _apply_smoke_overrides,
    )

    # Apply-path
    opts = argparse.Namespace(
        smoke=True,
        revision_lens=[100, 50, 20],
        revision_iters=[20, 25, 5],
        width=10,
        no_aug=False,
        val_size=128,
        eval_batch_size=128,
    )
    _apply_smoke_overrides(opts)
    assert opts.revision_lens == [100]
    assert opts.revision_iters == [2]
    assert opts.width == 1
    assert opts.no_aug is True
    assert opts.val_size == 8
    assert opts.eval_batch_size == 8

    # No-op when --smoke is off
    opts2 = argparse.Namespace(
        smoke=False,
        revision_lens=[20],
        revision_iters=[1],
        width=5,
        no_aug=False,
        val_size=10,
        eval_batch_size=10,
    )
    _apply_smoke_overrides(opts2)
    assert opts2.revision_lens == [20]
    assert opts2.revision_iters == [1]
    assert opts2.width == 5
    assert opts2.no_aug is False
    assert opts2.val_size == 10
    assert opts2.eval_batch_size == 10

    # No-op also leaves a small val_size alone (the cap is `> 8`).
    opts3 = argparse.Namespace(
        smoke=True,
        revision_lens=[100, 50, 20],
        revision_iters=[20, 25, 5],
        width=10,
        no_aug=False,
        val_size=4,
        eval_batch_size=4,
    )
    _apply_smoke_overrides(opts3)
    assert opts3.val_size == 4
    assert opts3.eval_batch_size == 4


def test_print_cpu_amp_warning(capsys):
    """`_print_cpu_amp_warning` must fire exactly once on CPU+AMP,
    stay silent on CUDA, and stay silent when --no_amp or --smoke is
    set. The user-facing recommendation must include `--no_aug` and
    `--smoke` so a user who is reading the warning knows what to do
    next.
    """
    from experiments.heatmap_guided.run_benchmark import (
        _print_cpu_amp_warning,
    )

    # Fires on CPU + AMP-enabled.
    opts = argparse.Namespace(device="cpu", no_amp=False, smoke=False)
    _print_cpu_amp_warning(opts)
    out = capsys.readouterr().out
    assert "Running on CPU" in out
    assert "--no_aug" in out
    assert "--smoke" in out

    # Silent on CUDA.
    opts_cuda = argparse.Namespace(device="cuda", no_amp=False, smoke=False)
    _print_cpu_amp_warning(opts_cuda)
    assert capsys.readouterr().out == ""

    # Silent when --no_amp is set.
    opts_noamp = argparse.Namespace(device="cpu", no_amp=True, smoke=False)
    _print_cpu_amp_warning(opts_noamp)
    assert capsys.readouterr().out == ""

    # Silent when --smoke is set (the user already opted into fast).
    opts_smoke = argparse.Namespace(device="cpu", no_amp=False, smoke=True)
    _print_cpu_amp_warning(opts_smoke)
    assert capsys.readouterr().out == ""

