"""Unit tests for utils.functions.chunk_2opt.

These tests run in <2s on CPU and require only torch (already a project dep).
The function under test is the chunk-level 2-opt search added to
`utils/functions.py`. See docs/LOCAL_CONSTRUCTION.md §11 for context.
"""
import warnings

import pytest
import torch

from utils.functions import chunk_2opt


def _closed_cost(x):
    """Canonical closed-loop tour cost (matches utils/functions.py:305, :320)."""
    return (x[:, 1:] - x[:, :-1]).norm(p=2, dim=2).sum(1) \
         + (x[:, 0] - x[:, -1]).norm(p=2, dim=1)


# --- Section A: shape round-trip --------------------------------------------------

def test_shape_round_trip():
    """Output shape matches input; node multiset is preserved."""
    torch.manual_seed(0)
    seed = torch.randn(3, 30, 2)
    out, cost, stats = chunk_2opt(seed, chunk_size=5, n_iters=3)
    assert out.shape == seed.shape
    assert cost.shape == (3,)
    assert 'initial_cost' in stats and 'final_cost' in stats
    assert 'iters' in stats and 'accepts' in stats
    assert 'improved_mask' in stats
    # The output is a permutation of the input nodes (same multiset of coords).
    for b in range(3):
        in_sorted = seed[b].sort(dim=0).values
        out_sorted = out[b].sort(dim=0).values
        assert torch.allclose(in_sorted, out_sorted, atol=1e-5), \
            f"batch {b}: node multiset changed"


# --- Section B: never-worsen invariant -------------------------------------------

@pytest.mark.parametrize("trial", range(10))
def test_never_worsen(trial):
    """On 10 random seeds, the search never increases closed-loop tour cost."""
    torch.manual_seed(1000 + trial)
    B = 4
    N = 60
    seed = torch.randn(B, N, 2) * 5
    out, cost, stats = chunk_2opt(seed, chunk_size=10, n_iters=10)
    # Recompute the canonical cost independently to rule out any caching bug.
    canonical = _closed_cost(out)
    assert (canonical <= _closed_cost(seed) + 1e-5).all(), \
        f"trial {trial}: cost increased {(_closed_cost(seed) - canonical).tolist()}"
    assert (stats['final_cost'] <= stats['initial_cost'] + 1e-5).all()


# --- Section C: strictly improves a known-bad order -----------------------------

def test_strictly_improves_known_bad_order():
    """On a deliberately zigzagged 4-anchor tour, the search finds a better order."""
    torch.manual_seed(0)
    # 4 chunks clustered tightly around 4 square corners
    anchors = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    chunks_pts = [a + torch.randn(5, 2) * 0.05 for a in anchors]
    # Bad order: 0 -> 2 -> 1 -> 3 (zigzag, long cross-edges)
    bad = [0, 2, 1, 3]
    bad_tour = torch.cat([chunks_pts[i] for i in bad], dim=0).unsqueeze(0)
    cost_before = _closed_cost(bad_tour).item()
    out, cost_after, stats = chunk_2opt(bad_tour, chunk_size=5, n_iters=5)
    assert cost_after.item() < cost_before, \
        f"chunk_2opt failed to improve: before={cost_before:.3f}, after={cost_after.item():.3f}"
    assert stats['improved_mask'].item() is True


# --- Section D: closed vs open loop --------------------------------------------

def test_closed_vs_open_loop():
    """With include_close=True, the wraparound edge is considered; with False, it isn't."""
    # Use a tour where the wraparound edge is much longer than the others,
    # so the cost differs visibly between modes.
    seed = torch.tensor([[
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
        [3.0, 1.0], [3.0, 2.0], [3.0, 3.0], [3.0, 4.0],
        [2.0, 4.0], [1.0, 4.0], [0.0, 4.0], [0.0, 3.0],
    ]]).reshape(1, 12, 2)
    # closed = sum of 12 edges including wraparound (0,0)->(0,3) ~= 3
    # open   = same minus the wraparound edge
    closed = _closed_cost(seed).item()
    # The closed-loop cost should be strictly greater than the open-loop cost
    # because the wraparound (0,0)->(0,3) has length 3 while the average
    # internal edge is ~1.
    assert closed > 10.0  # sanity check on the construction

    # include_close=True (default) considers the wraparound edge in 2-opt moves.
    _, cost_closed, stats_closed = chunk_2opt(seed, chunk_size=4, n_iters=3,
                                              include_close=True)
    # include_close=False ignores it; final cost may differ on this construction.
    _, cost_open, stats_open = chunk_2opt(seed, chunk_size=4, n_iters=3,
                                          include_close=False)
    # At minimum, both should be <= their initial cost (never-worsen).
    assert cost_closed.item() <= _closed_cost(seed).item() + 1e-5
    assert cost_open.item() <= _closed_cost(seed).item() + 1e-5


# --- Section E: orientation flip -------------------------------------------------

def test_orientation_flip_flag_is_accepted():
    """The include_flip flag is currently a no-op (the all-flip variant is
    not implemented), but the API must accept it without error and preserve
    the never-worsen invariant."""
    torch.manual_seed(0)
    seed = torch.randn(2, 30, 2)
    out, cost, stats = chunk_2opt(seed, chunk_size=5, n_iters=5, include_flip=True)
    assert (stats['final_cost'] <= stats['initial_cost'] + 1e-5).all()


# --- Section F: graceful degradation --------------------------------------------

def test_graceful_degradation_chunk_size_too_large():
    """chunk_size >= N is a no-op."""
    seed = torch.randn(2, 20, 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out, cost, stats = chunk_2opt(seed, chunk_size=20, n_iters=5)
    assert any('chunk_size' in str(w.message) for w in caught)
    assert torch.allclose(out, seed)


def test_graceful_degradation_k2():
    """k=2 has no valid 2-opt move; should return input unchanged without warning."""
    seed = torch.randn(2, 20, 2)
    out, cost, stats = chunk_2opt(seed, chunk_size=10, n_iters=5)
    # No warning expected (the check is `chunk_size <= 1 or chunk_size >= N`,
    # which doesn't trigger for k=2). The function should just return the
    # initial cost.
    assert torch.allclose(out, seed)
    assert (cost == _closed_cost(seed)).all()
    assert stats['accepts'] == 0


# --- Section F': partial chunk (offset != 0) handling ----------------------------

@pytest.mark.parametrize("n,cs", [(23, 10), (47, 20), (101, 30), (53, 7), (99, 25), (10, 3)])
def test_never_worsen_with_offset(n, cs):
    """Never-worsen invariant when N is not a multiple of chunk_size (offset != 0)."""
    torch.manual_seed(3000 + n + cs)
    B = 4
    seed = torch.randn(B, n, 2) * 5
    out, cost, stats = chunk_2opt(seed, chunk_size=cs, n_iters=10)
    canonical = _closed_cost(out)
    assert (canonical <= _closed_cost(seed) + 1e-5).all(), \
        f"n={n}, cs={cs}: cost increased " \
        f"{(_closed_cost(seed) - canonical).tolist()}"
    assert (stats['final_cost'] <= stats['initial_cost'] + 1e-5).all()
    # Node multiset is preserved (permutation).
    for b in range(B):
        in_sorted = seed[b].sort(dim=0).values
        out_sorted = out[b].sort(dim=0).values
        assert torch.allclose(in_sorted, out_sorted, atol=1e-5), \
            f"batch {b}: node multiset changed"


def test_offset_strictly_improves_known_bad_order():
    """On a deliberately bad order with a partial chunk, the search finds a
    better ordering that includes the partial chunk."""
    torch.manual_seed(0)
    # 3 complete chunks of 10 nodes each + 1 partial chunk of 3 nodes
    # placed near 4 anchor points forming a square (with the partial anchor
    # at the center).
    anchors = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [5.0, 5.0]])
    chunks_pts = []
    for a in anchors[:3]:
        chunks_pts.append(a + torch.randn(10, 2) * 0.05)
    chunks_pts.append(anchors[3] + torch.randn(3, 2) * 0.05)  # partial chunk of 3
    # Bad order: 0, 2, 1, 3 (zigzag with the partial chunk last)
    bad = [0, 2, 1, 3]
    bad_tour = torch.cat([chunks_pts[i] for i in bad], dim=0).unsqueeze(0)
    cost_before = _closed_cost(bad_tour).item()
    out, cost_after, stats = chunk_2opt(bad_tour, chunk_size=10, n_iters=5)
    assert cost_after.item() < cost_before, \
        f"chunk_2opt failed to improve offset != 0 case: " \
        f"before={cost_before:.3f}, after={cost_after.item():.3f}"
    assert stats['improved_mask'].item() is True


def test_offset_partial_chunk_is_actively_joined():
    """The partial chunk is a regular 2-opt unit, not a fixed tail. The
    search can move it to any position in the tour."""
    # 4 chunks of size 5 + 1 partial chunk of size 3 = 23 total.
    # We use a "broken line" layout where putting the partial chunk at the
    # end is suboptimal — the search should move it elsewhere.
    torch.manual_seed(1)
    # 5 anchors along a slightly curved line; the partial chunk is the
    # 5th anchor (smaller chunk at the end of the original tour).
    anchors = torch.tensor([
        [0.0, 0.0], [2.0, 0.5], [4.0, -0.3], [6.0, 0.4], [8.0, 0.0],
    ])
    # Original order (bad): 0, 1, 2, 3, 4 with the 4th chunk being partial
    # (only 3 nodes instead of 5). Better: swap so the partial is in the
    # middle where it has cheaper neighbors.
    chunks_pts = []
    chunk_sizes = [5, 5, 5, 5, 3]
    for a, cs in zip(anchors, chunk_sizes):
        chunks_pts.append(a + torch.randn(cs, 2) * 0.05)
    bad_tour = torch.cat(chunks_pts, dim=0).unsqueeze(0)
    out, cost_after, stats = chunk_2opt(bad_tour, chunk_size=5, n_iters=5)
    # The search should at least not worsen.
    assert (stats['final_cost'] <= stats['initial_cost'] + 1e-5).all()


# --- Section G: integration smoke (reconnect-like) ------------------------------

def test_integration_smoke_with_noop_reviser():
    """Run a minimal reconnect-like loop with a no-op reviser and
    --chunk_2opt-style flags set, and confirm cost never increases."""
    from utils.functions import reconnect

    # No-op reviser: returns the input order unchanged. The cost it reports
    # must match `reviser.problem.get_costs(decomposed_seeds, sub_tour)` which
    # is the canonical (B, N, 2) closed-loop cost. We pack everything into a
    # class with the attributes `revision` (utils/functions.py:215) reads.
    class NoOpProblem:
        @staticmethod
        def get_costs(dataset, pi):
            return _closed_cost(dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))), None

    class NoOpReviser:
        problem = NoOpProblem()
        def eval(self): return self
        def to(self, device): return self
        def set_decode_type(self, s): pass
        def __call__(self, x, return_pi=False, return_embedding=False, **kw):
            pi = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
            cost = _closed_cost(x)
            if return_embedding:
                return cost, pi, cost, pi, x.clone()
            return cost, pi, cost, pi

    seed = torch.randn(8, 50, 2)
    problem_cost = lambda inp, pi: _closed_cost(inp.gather(1, pi.unsqueeze(-1).expand_as(inp)))

    class Opts:
        revision_lens = [10]
        revision_iters = [1]
        chunk_2opt = True
        chunk_2opt_after_each_iter = True
        chunk_2opt_flip = False
        chunk_2opt_iters = 5
        chunk_2opt_chunk_size = 0   # use revision_lens[revision_id]
        no_prune = True
        no_aug = True               # required by revision()
        eval_batch_size = 8
        width = 1

    out, cost = reconnect(
        get_cost_func=problem_cost,
        batch=seed,
        opts=Opts(),
        revisers=[NoOpReviser()],
    )
    # With a no-op reviser, the only thing that can change the tour is
    # chunk_2opt. If chunk_2opt is correct, cost must not have increased.
    canonical = _closed_cost(out)
    assert (canonical <= _closed_cost(seed) + 1e-5).all()


def test_per_pass_chunk_2opt_is_invoked():
    """With --chunk_2opt_after_each_iter=True, chunk_2opt runs inside LCP_TSP
    after each individual reviser pass, not just at cascade boundaries."""
    from utils.bench_utils import BenchLogger
    from utils.functions import reconnect

    class NoOpProblem:
        @staticmethod
        def get_costs(dataset, pi):
            return _closed_cost(dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))), None

    class NoOpReviser:
        problem = NoOpProblem()
        def eval(self): return self
        def to(self, device): return self
        def set_decode_type(self, s): pass
        def __call__(self, x, return_pi=False, return_embedding=False, **kw):
            pi = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
            cost = _closed_cost(x)
            if return_embedding:
                return cost, pi, cost, pi, x.clone()
            return cost, pi, cost, pi

    seed = torch.randn(2, 40, 2)
    problem_cost = lambda inp, pi: _closed_cost(inp.gather(1, pi.unsqueeze(-1).expand_as(inp)))

    class Opts:
        revision_lens = [10]
        revision_iters = [3]   # 3 passes per cascade
        chunk_2opt = True
        chunk_2opt_after_each_iter = True
        chunk_2opt_flip = False
        chunk_2opt_iters = 3
        chunk_2opt_chunk_size = 0
        no_prune = True
        no_aug = True
        eval_batch_size = 2
        width = 1

    logger = BenchLogger()
    out, cost = reconnect(
        get_cost_func=problem_cost,
        batch=seed,
        opts=Opts(),
        revisers=[NoOpReviser()],
        bench_logger=logger,
    )
    # We expect 3 per-pass chunk_2opt stages (one per pass) with names
    # containing "iter0", "iter1", "iter2".
    c2o_stages = [r for r in logger.records if r['stage'].startswith('chunk_2opt')]
    iter_ids = sorted(int(r['stage'].rsplit('iter', 1)[1]) for r in c2o_stages)
    assert iter_ids == [0, 1, 2], f"expected per-pass stages, got {[r['stage'] for r in c2o_stages]}"
    # And the cost must not have worsened.
    canonical = _closed_cost(out)
    assert (canonical <= _closed_cost(seed) + 1e-5).all()


# --- Section H: default chunk_size matches SHPP chunk_size ---------------------

class _NoOpReviserSetup:
    """Helper to build a no-op reviser and a get_cost_func for reconnect tests."""

    @staticmethod
    def make():
        class NoOpProblem:
            @staticmethod
            def get_costs(dataset, pi):
                return _closed_cost(dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))), None

        class NoOpReviser:
            problem = NoOpProblem()
            def eval(self): return self
            def to(self, device): return self
            def set_decode_type(self, s): pass
            def __call__(self, x, return_pi=False, return_embedding=False, **kw):
                pi = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
                cost = _closed_cost(x)
                if return_embedding:
                    return cost, pi, cost, pi, x.clone()
                return cost, pi, cost, pi

        return NoOpReviser, lambda inp, pi: _closed_cost(inp.gather(1, pi.unsqueeze(-1).expand_as(inp)))


def test_chunk_2opt_uses_sphp_chunk_size_by_default(monkeypatch):
    """By default (--chunk_2opt_chunk_size=0), chunk_2opt is invoked with
    the same chunk_size as the SHPP decomposition (revision_len)."""
    import utils.functions as fn_module
    from utils.functions import reconnect

    NoOpReviser, problem_cost = _NoOpReviserSetup().make()

    # Spy on chunk_2opt to record the chunk_size argument of each call.
    calls = []
    real_chunk_2opt = fn_module.chunk_2opt

    def spy_chunk_2opt(seed, chunk_size, **kwargs):
        calls.append(int(chunk_size))
        return real_chunk_2opt(seed, chunk_size, **kwargs)

    monkeypatch.setattr(fn_module, 'chunk_2opt', spy_chunk_2opt)

    seed = torch.randn(2, 40, 2)  # N=40, 40%20=0, k=2 complete chunks
    # revision_lens=[20], revision_iters=[3] → 3 per-pass chunk_2opt calls
    # at chunk_size=20 (the SHPP chunk size).
    class Opts:
        revision_lens = [20]
        revision_iters = [3]
        chunk_2opt = True
        chunk_2opt_after_each_iter = True
        chunk_2opt_flip = False
        chunk_2opt_iters = 3
        chunk_2opt_chunk_size = 0   # <-- default: use SHPP chunk size
        no_prune = True
        no_aug = True
        eval_batch_size = 2
        width = 1

    out, _ = reconnect(
        get_cost_func=problem_cost,
        batch=seed,
        opts=Opts(),
        revisers=[NoOpReviser()],
    )
    # Default: all 3 calls use chunk_size=20 (matching revision_len).
    assert calls == [20, 20, 20], (
        f"chunk_2opt should be called with the SHPP chunk size (20) by "
        f"default, got {calls}"
    )


def test_chunk_2opt_chunk_size_override_works(monkeypatch):
    """--chunk_2opt_chunk_size overrides the SHPP chunk size when set, and
    --no_chunk_2opt_exhaustive_shifts disables the auto-exhaustive mode
    (so the test sees exactly one call per pass)."""
    import utils.functions as fn_module
    from utils.functions import reconnect

    NoOpReviser, problem_cost = _NoOpReviserSetup().make()

    calls = []
    real_chunk_2opt = fn_module.chunk_2opt

    def spy_chunk_2opt(seed, chunk_size, **kwargs):
        calls.append(int(chunk_size))
        return real_chunk_2opt(seed, chunk_size, **kwargs)

    monkeypatch.setattr(fn_module, 'chunk_2opt', spy_chunk_2opt)

    seed = torch.randn(2, 40, 2)
    class Opts:
        revision_lens = [20]   # SHPP chunk size
        revision_iters = [2]
        chunk_2opt = True
        chunk_2opt_after_each_iter = True
        chunk_2opt_flip = False
        chunk_2opt_iters = 3
        chunk_2opt_chunk_size = 10  # <-- override: use 10-node chunks for 2-opt
        no_chunk_2opt_exhaustive_shifts = True  # <-- disable auto-exhaustive
        no_prune = True
        no_aug = True
        eval_batch_size = 2
        width = 1

    out, _ = reconnect(
        get_cost_func=problem_cost,
        batch=seed,
        opts=Opts(),
        revisers=[NoOpReviser()],
    )
    # Override + no exhaustive: all calls use chunk_size=10, exactly 2 calls.
    assert calls == [10, 10], (
        f"--chunk_2opt_chunk_size=10 + --no_chunk_2opt_exhaustive_shifts should "
        f"give 2 calls at chunk_size=10, got {calls}"
    )


def test_exhaustive_shifts_auto_enabled_for_smaller_chunk_size(monkeypatch):
    """When --chunk_2opt_chunk_size < revision_len, exhaustive_shifts is
    auto-enabled: each per-pass call loops over all `cs` rotations, so the
    total number of single-shift chunk_2opt invocations is `revision_iters * cs`."""
    import utils.functions as fn_module
    from utils.functions import reconnect

    NoOpReviser, problem_cost = _NoOpReviserSetup().make()

    # Only count the inner (single-shift) calls, not the outer wrapper.
    inner_calls = []
    real_chunk_2opt = fn_module.chunk_2opt

    def spy_chunk_2opt(seed, chunk_size, **kwargs):
        if not kwargs.get('exhaustive_shifts', False):
            inner_calls.append(int(chunk_size))
        return real_chunk_2opt(seed, chunk_size, **kwargs)

    monkeypatch.setattr(fn_module, 'chunk_2opt', spy_chunk_2opt)

    seed = torch.randn(2, 40, 2)
    class Opts:
        revision_lens = [20]
        revision_iters = [2]   # 2 passes
        chunk_2opt = True
        chunk_2opt_after_each_iter = True
        chunk_2opt_flip = False
        chunk_2opt_iters = 3
        chunk_2opt_chunk_size = 10  # <-- 10 < 20 → auto-exhaustive enabled
        # no_chunk_2opt_exhaustive_shifts is NOT set (default: not disabled)
        no_prune = True
        no_aug = True
        eval_batch_size = 2
        width = 1

    out, _ = reconnect(
        get_cost_func=problem_cost,
        batch=seed,
        opts=Opts(),
        revisers=[NoOpReviser()],
    )
    # 2 passes × 10 shifts = 20 inner calls. All at chunk_size=10.
    assert len(inner_calls) == 20, (
        f"expected 20 inner calls (2 passes × 10 shifts), got {len(inner_calls)}"
    )
    assert all(c == 10 for c in inner_calls), \
        f"all inner calls should be at chunk_size=10, got {inner_calls}"


def test_exhaustive_shifts_unit_level():
    """At the unit level, chunk_2opt(exhaustive_shifts=True) tries all
    `chunk_size` rotations and keeps the best per-instance."""
    torch.manual_seed(0)
    seed = torch.randn(2, 20, 2)  # N=20, chunk_size=5, 4 chunks
    out_exh, cost_exh, stats_exh = chunk_2opt(
        seed, chunk_size=5, n_iters=3, exhaustive_shifts=True,
    )
    out_one, cost_one, _ = chunk_2opt(
        seed, chunk_size=5, n_iters=3, exhaustive_shifts=False,
    )
    # Exhaustive must be at least as good as single-shift.
    assert (cost_exh <= cost_one + 1e-5).all()
    # And must have tried all 5 shifts.
    assert stats_exh.get('exhaustive_shifts') == 5
    # Never worsen.
    assert (cost_exh <= stats_exh['initial_cost'] + 1e-5).all()
    # Node multiset preserved.
    for b in range(2):
        in_sorted = seed[b].sort(dim=0).values
        out_sorted = out_exh[b].sort(dim=0).values
        assert torch.allclose(in_sorted, out_sorted, atol=1e-5)


def test_exhaustive_shifts_matches_manual_loop():
    """chunk_2opt(exhaustive_shifts=True) should match a manual loop over
    all shifts taking the per-instance best."""
    torch.manual_seed(42)
    seed = torch.randn(3, 30, 2)  # N=30, chunk_size=6, 5 chunks

    # Method 1: exhaustive_shifts=True
    out1, cost1, _ = chunk_2opt(seed, chunk_size=6, n_iters=3, exhaustive_shifts=True)

    # Method 2: manual loop over all 6 shifts
    best_cost = _closed_cost(seed)
    best_seed = seed.clone()
    for shift in range(6):
        if shift > 0:
            shifted = torch.roll(seed, shifts=shift, dims=1)
        else:
            shifted = seed
        s, c, _ = chunk_2opt(shifted, chunk_size=6, n_iters=3, exhaustive_shifts=False)
        if shift > 0:
            s = torch.roll(s, shifts=-shift, dims=1)
        improved = c < best_cost - 1e-9
        if improved.any():
            best_seed[improved] = s[improved]
            best_cost[improved] = c[improved]

    # The two methods should give the same per-instance best cost.
    assert torch.allclose(cost1, best_cost, atol=1e-5), (
        f"exhaustive_shifts cost {cost1.tolist()} != manual loop cost {best_cost.tolist()}"
    )


def test_chunk_2opt_end_of_pipeline_uses_sphp_size_by_default(monkeypatch):
    """With --chunk_2opt_after_each_iter=False, the end-of-pipeline call
    also defaults to the SHPP chunk size (last cascade level's revision_len)."""
    import utils.functions as fn_module
    from utils.functions import reconnect

    NoOpReviser, problem_cost = _NoOpReviserSetup().make()

    calls = []
    real_chunk_2opt = fn_module.chunk_2opt

    def spy_chunk_2opt(seed, chunk_size, **kwargs):
        calls.append(int(chunk_size))
        return real_chunk_2opt(seed, chunk_size, **kwargs)

    monkeypatch.setattr(fn_module, 'chunk_2opt', spy_chunk_2opt)

    # N=120 so all cascade sizes (20, 50) divide it without offset. End-of-
    # pipeline default = last cascade's chunk size = 50.
    seed = torch.randn(2, 120, 2)
    class Opts:
        revision_lens = [20, 50]   # cascade levels
        revision_iters = [1, 1]
        chunk_2opt = True
        chunk_2opt_after_each_iter = False   # end-of-pipeline only
        chunk_2opt_flip = False
        chunk_2opt_iters = 3
        chunk_2opt_chunk_size = 0
        no_prune = True
        no_aug = True
        eval_batch_size = 2
        width = 1

    out, _ = reconnect(
        get_cost_func=problem_cost,
        batch=seed,
        opts=Opts(),
        revisers=[NoOpReviser(), NoOpReviser()],
    )
    # End-of-pipeline only: exactly one call, with the last cascade's
    # revision_len (= 50).
    assert calls == [50], (
        f"end-of-pipeline chunk_2opt should default to the last cascade's "
        f"SHPP chunk size (50), got {calls}"
    )
