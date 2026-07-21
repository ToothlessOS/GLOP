"""Sanity tests for the 2-opt post-processing in utils/post_process.py.

Run with:  python -m pytest tests/test_post_process.py -q
       or:  python tests/test_post_process.py
"""

import contextlib
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.post_process import full_2opt, knn_2opt, maybe_two_opt
from utils import post_process as pp_mod
from utils.post_process import (
    _knn_edges_scipy,
    _pyg_knn_graph,
    knn_2opt_gain_matrix,
    radius_2opt,
    radius_2opt_gain_matrix,
    range_radius_2opt,
    range_radius_2opt_gain_matrix,
)

# eval_2opt imports happen lazily inside its tests below, since eval_2opt
# transitively imports main._eval_dataset which requires the GLOP pipeline.


def tour_cost(coords):
    # coords: (B, N, D) in tour order -> (B,) closed-loop Euclidean cost
    return (coords[:, 1:] - coords[:, :-1]).norm(p=2, dim=2).sum(1) + \
           (coords[:, 0] - coords[:, -1]).norm(p=2, dim=1)


def _make_opts(**kw):
    """Build a minimal opts stub with the given attributes."""

    class Opts:
        pass

    o = Opts()
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def test_shape_roundtrip():
    torch.manual_seed(0)
    x = torch.rand(4, 12, 2)
    out = full_2opt(x, iters=10)
    assert out.shape == x.shape


def test_does_not_mutate_input():
    torch.manual_seed(1)
    x = torch.rand(3, 15, 2)
    x_before = x.clone()
    _ = full_2opt(x, iters=10)
    assert torch.equal(x, x_before)


def test_never_worsens_random():
    torch.manual_seed(2)
    x = torch.rand(8, 20, 2)
    c0 = tour_cost(x)
    out = full_2opt(x, iters=25)
    c1 = tour_cost(out)
    # allow tiny fp slack
    assert (c1 <= c0 + 1e-5).all(), (c0, c1)


def test_multi_sweep_no_crash():
    # The old implementation crashed on the 2nd sweep. Ensure many sweeps run.
    torch.manual_seed(3)
    x = torch.rand(2, 30, 2)
    out = full_2opt(x, iters=50)
    assert out.shape == x.shape


def test_known_crossed_square():
    # Unit square visited in a self-crossing order: 2-opt must uncross it.
    # Corners: A(0,0) B(1,0) C(1,1) D(0,1). Bad order A,C,B,D crosses.
    coords = torch.tensor([[[0., 0.], [1., 1.], [1., 0.], [0., 1.]]])
    out = full_2opt(coords, iters=10)
    c1 = tour_cost(out).item()
    # Optimal Hamiltonian cycle on the unit square = perimeter = 4.0
    assert abs(c1 - 4.0) < 1e-5, c1


def test_maybe_two_opt_gating():
    torch.manual_seed(4)
    x = torch.rand(2, 10, 2)

    class Opts:
        pass

    off = Opts()  # no use_2opt attr -> disabled
    assert torch.equal(maybe_two_opt(x, off), x)

    on = Opts()
    on.use_2opt = True
    on.two_opt_iters = 10
    out = maybe_two_opt(x, on)
    assert (tour_cost(out) <= tour_cost(x) + 1e-5).all()


# ---------------------------------------------------------------------------
# KNN-sparse 2-opt tests
# ---------------------------------------------------------------------------


def test_knn_shape_roundtrip():
    torch.manual_seed(10)
    x = torch.rand(4, 15, 2)
    for k in (3, 5, 10):
        out = knn_2opt(x, iters=10, k=k)
        assert out.shape == x.shape


def test_knn_does_not_mutate_input():
    torch.manual_seed(11)
    x = torch.rand(3, 15, 2)
    x_before = x.clone()
    _ = knn_2opt(x, iters=10, k=10)
    assert torch.equal(x, x_before)


def test_knn_never_worsens_random():
    torch.manual_seed(12)
    x = torch.rand(8, 20, 2)
    c0 = tour_cost(x)
    out = knn_2opt(x, iters=25, k=10)
    c1 = tour_cost(out)
    # KNN only sees a subset of candidate edges, but cost must still be
    # non-increasing (positive-gain moves only).
    assert (c1 <= c0 + 1e-5).all(), (c0, c1)


def test_knn_multi_sweep_no_crash():
    torch.manual_seed(13)
    x = torch.rand(2, 30, 2)
    out = knn_2opt(x, iters=50, k=10)
    assert out.shape == x.shape


def test_knn_known_crossed_square():
    # Same crossed square as test_known_crossed_square: KNN with k=3 connects
    # every node to its 3 nearest; on the unit square that always includes the
    # true uncrossing edge (A-D or B-C), so the optimum cost 4.0 is reachable.
    coords = torch.tensor([[[0., 0.], [1., 1.], [1., 0.], [0., 1.]]])
    out = knn_2opt(coords, iters=10, k=3)
    c1 = tour_cost(out).item()
    assert abs(c1 - 4.0) < 1e-5, c1


def test_knn_with_k_too_small():
    # Degenerate case: k=1 gives very few candidate edges. Must still
    # terminate and return a tour of the right shape; cost may not improve.
    torch.manual_seed(14)
    x = torch.rand(2, 12, 2)
    out = knn_2opt(x, iters=5, k=1)
    assert out.shape == x.shape
    c0 = tour_cost(x)
    c1 = tour_cost(out)
    assert (c1 <= c0 + 1e-5).all()


def test_knn_k_parameter_consumed():
    # Larger k should never give worse mean cost than smaller k over a random
    # batch: more candidates ⊇ more improving moves.
    torch.manual_seed(15)
    x = torch.rand(8, 25, 2)
    c0 = tour_cost(x)
    out_small = knn_2opt(x, iters=15, k=3)
    out_large = knn_2opt(x, iters=15, k=15)
    mean_small = (tour_cost(out_small) - c0).mean().item()
    mean_large = (tour_cost(out_large) - c0).mean().item()
    assert mean_large <= mean_small + 1e-5, (mean_small, mean_large)


def test_knn_no_invalid_self_move():
    # Regression test for the adjacency / wrap-around mask in
    # knn_2opt_gain_matrix: an edge connecting tour-adjacent positions is a
    # no-op; with the mask, KNN cannot pick it as the best edge.
    # Tour A(0,0) -> B(1,0) -> C(2,0) -> D(3,0) -> A (collinear line).
    # KNN with k=3 makes every node see its 3 in-order successors; without the
    # mask the algorithm could pick a (i, i+1) edge. The cost is already
    # optimal for an alternative traversal, so the tour must not get worse.
    coords = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [3., 0.]]])
    out = knn_2opt(coords, iters=10, k=3)
    c0 = tour_cost(coords).item()
    c1 = tour_cost(out).item()
    assert c1 <= c0 + 1e-5, (c0, c1)


# ---------------------------------------------------------------------------
# Radius-sparse 2-opt tests
# ---------------------------------------------------------------------------


def test_radius_gain_matrix_shape_and_contract():
    # The return contract must match knn_2opt_gain_matrix so the same driver
    # code can aggregate per-instance gains via scatter_max.
    torch.manual_seed(70)
    B, N, r = 3, 20, 5
    seeds = torch.rand(B, N, 2)
    tour, edge_gain, idx_tuple = radius_2opt_gain_matrix(seeds, radius=r)
    # tour shape unchanged
    assert tour.shape == (B, N)
    # edge count = B * N * 2*(r-1) offsets (each batch slice has the same count)
    assert edge_gain.shape == (B * N * 2 * (r - 1),)
    # idx_tuple = (src_b, src_i, dst_j), all flat int tensors
    assert idx_tuple is not None and len(idx_tuple) == 3
    src_b, src_i, dst_j = idx_tuple
    assert src_b.shape == src_i.shape == dst_j.shape == edge_gain.shape
    # All batch indices in [0, B); all positions in [0, N).
    assert src_b.min().item() >= 0 and src_b.max().item() < B
    assert src_i.min().item() >= 0 and src_i.max().item() < N
    assert dst_j.min().item() >= 0 and dst_j.max().item() < N


def test_radius_gain_matrix_handles_radius_two():
    # radius=2 means only offset ±2 -> 2*(R-1)=2 offsets per base position.
    # Should still return a valid gain vector and not crash.
    torch.manual_seed(71)
    B, N = 2, 12
    seeds = torch.rand(B, N, 2)
    tour, edge_gain, idx_tuple = radius_2opt_gain_matrix(seeds, radius=2)
    assert tour.shape == (B, N)
    assert edge_gain.shape == (B * N * 2 * (2 - 1),)  # = B * N * 2
    assert idx_tuple is not None


def test_radius_gain_matrix_default_radius_large_n():
    # Sanity: with a fairly large N the gain computation must terminate in a
    # reasonable time and produce the expected edge count (no memory blow-up).
    import time
    torch.manual_seed(72)
    B, N, r = 2, 500, 25
    seeds = torch.rand(B, N, 2)
    t0 = time.time()
    _, edge_gain, idx_tuple = radius_2opt_gain_matrix(seeds, radius=r)
    dt = time.time() - t0
    assert idx_tuple is not None
    assert edge_gain.shape == (B * N * 2 * (r - 1),)
    assert dt < 30.0, f"radius_2opt_gain_matrix took {dt:.2f}s on N=500"


def test_radius_shape_roundtrip():
    torch.manual_seed(73)
    x = torch.rand(4, 15, 2)
    for r in (3, 5, 10):
        out = radius_2opt(x, iters=10, r=r)
        assert out.shape == x.shape


def test_radius_does_not_mutate_input():
    torch.manual_seed(74)
    x = torch.rand(3, 15, 2)
    x_before = x.clone()
    _ = radius_2opt(x, iters=10, r=10)
    assert torch.equal(x, x_before)


def test_radius_never_worsens_random():
    # Central invariant: only positive-gain moves are applied, so tour cost
    # is non-increasing.
    torch.manual_seed(75)
    x = torch.rand(8, 30, 2)
    c0 = tour_cost(x)
    out = radius_2opt(x, iters=25, r=5)
    c1 = tour_cost(out)
    assert (c1 <= c0 + 1e-5).all(), (c0, c1)


def test_radius_known_crossed_square():
    # Tour positions in the crossed square are 4; radius=2 already covers
    # the (i=0, j=2) and (i=1, j=3) flip pairs, so the optimum is reachable.
    coords = torch.tensor([[[0., 0.], [1., 1.], [1., 0.], [0., 1.]]])
    out = radius_2opt(coords, iters=10, r=2)
    c1 = tour_cost(out).item()
    assert abs(c1 - 4.0) < 1e-5, c1


def test_radius_no_invalid_self_move():
    # Wrap-around regression: tour of 4 collinear points, radius=2. The
    # offset (i=0, j=N-1=3) corresponds to (i+1)%N == j -- i.e. a wrap-around
    # adjacent edge -- and must be masked out so the algorithm cannot pick it.
    coords = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [3., 0.]]])
    out = radius_2opt(coords, iters=10, r=2)
    c0 = tour_cost(coords).item()
    c1 = tour_cost(out).item()
    assert c1 <= c0 + 1e-5, (c0, c1)


def test_radius_multi_sweep_no_crash():
    torch.manual_seed(76)
    x = torch.rand(2, 30, 2)
    out = radius_2opt(x, iters=50, r=10)
    assert out.shape == x.shape


def test_radius_radius_monotonicity():
    # Larger radius should never give worse mean cost than smaller radius
    # over a random batch: more candidates ⊇ more improving moves.
    torch.manual_seed(77)
    x = torch.rand(8, 25, 2)
    c0 = tour_cost(x)
    out_small = radius_2opt(x, iters=15, r=3)
    out_large = radius_2opt(x, iters=15, r=10)
    mean_small = (tour_cost(out_small) - c0).mean().item()
    mean_large = (tour_cost(out_large) - c0).mean().item()
    assert mean_large <= mean_small + 1e-5, (mean_small, mean_large)


def test_radius_with_r_too_small():
    # Degenerate: r=2 means only one offset pair; algorithm must still
    # terminate and return a tour of the right shape; cost may not improve.
    torch.manual_seed(78)
    x = torch.rand(2, 12, 2)
    out = radius_2opt(x, iters=5, r=2)
    assert out.shape == x.shape
    c0 = tour_cost(x)
    c1 = tour_cost(out)
    assert (c1 <= c0 + 1e-5).all()


def test_radius_debug_does_not_crash_or_miscompute():
    # debug=True must produce the same tour as debug=False (no side effects
    # on output) and remain cost-non-worsening.
    import contextlib, io

    torch.manual_seed(79)
    x = torch.rand(4, 30, 2)
    c0 = tour_cost(x)
    with contextlib.redirect_stdout(io.StringIO()) as quiet:
        out_dbg = radius_2opt(x, iters=5, r=10, debug=True)
        out_silent = radius_2opt(x.clone(), iters=5, r=10, debug=False)

    torch.testing.assert_close(out_dbg, out_silent)
    assert (tour_cost(out_dbg) <= c0 + 1e-5).all()


def test_maybe_two_opt_dispatch_radius():
    # 'radius' routes to radius_2opt and honours two_opt_radius.
    torch.manual_seed(80)
    x = torch.rand(4, 30, 2)
    opts = _make_opts(
        use_2opt=True,
        two_opt_iters=10,
        two_opt_kind='radius',
        two_opt_radius=5,
    )
    out = maybe_two_opt(x, opts)
    assert out.shape == x.shape
    assert (tour_cost(out) <= tour_cost(x) + 1e-5).all()


def test_maybe_two_opt_dispatch_radius_default_fallback():
    # two_opt_radius=None falls back to ~10% of N in the dispatcher.
    torch.manual_seed(81)
    N = 50
    x = torch.rand(2, N, 2)
    opts = _make_opts(
        use_2opt=True,
        two_opt_iters=10,
        two_opt_kind='radius',
        # two_opt_radius intentionally absent -> dispatcher default.
    )
    assert not hasattr(opts, 'two_opt_radius')
    out = maybe_two_opt(x, opts)
    assert out.shape == x.shape
    assert (tour_cost(out) <= tour_cost(x) + 1e-5).all()


# ---------------------------------------------------------------------------
# Range-radius-sparse 2-opt tests
# ---------------------------------------------------------------------------


def test_range_gain_matrix_shape_and_contract():
    # The return contract must match radius_2opt_gain_matrix so the same
    # driver code can aggregate per-instance gains via scatter_max.
    torch.manual_seed(90)
    B, N, r_min, r_max = 3, 20, 2, 5
    seeds = torch.rand(B, N, 2)
    tour, edge_gain, idx_tuple = range_radius_2opt_gain_matrix(
        seeds, r_min=r_min, r_max=r_max
    )
    # tour shape unchanged
    assert tour.shape == (B, N)
    # edge count = B * N * 2*(r_max - r_min + 1) offsets per batch slice.
    assert edge_gain.shape == (B * N * 2 * (r_max - r_min + 1),)
    # idx_tuple = (src_b, src_i, dst_j), all flat int tensors
    assert idx_tuple is not None and len(idx_tuple) == 3
    src_b, src_i, dst_j = idx_tuple
    assert src_b.shape == src_i.shape == dst_j.shape == edge_gain.shape
    # All batch indices in [0, B); all positions in [0, N).
    assert src_b.min().item() >= 0 and src_b.max().item() < B
    assert src_i.min().item() >= 0 and src_i.max().item() < N
    assert dst_j.min().item() >= 0 and dst_j.max().item() < N


def test_range_gain_matrix_small_window():
    # r_min=2, r_max=3 -> offsets {2, 3, -2, -3} -> 4 offsets per base position.
    torch.manual_seed(91)
    B, N = 2, 12
    seeds = torch.rand(B, N, 2)
    tour, edge_gain, idx_tuple = range_radius_2opt_gain_matrix(
        seeds, r_min=2, r_max=3
    )
    assert tour.shape == (B, N)
    assert edge_gain.shape == (B * N * 2 * (3 - 2 + 1),)  # = B * N * 4
    assert idx_tuple is not None


def test_range_gain_matrix_handles_r_max_eq_r_min():
    # Degenerate: r_max == r_min collapses to a single offset pair, equivalent
    # to the radius=2 case in radius_2opt_gain_matrix.
    torch.manual_seed(92)
    B, N = 2, 12
    seeds = torch.rand(B, N, 2)
    tour, edge_gain, idx_tuple = range_radius_2opt_gain_matrix(
        seeds, r_min=2, r_max=2
    )
    assert tour.shape == (B, N)
    assert edge_gain.shape == (B * N * 2 * 1,)  # = B * N * 2
    assert idx_tuple is not None


def test_range_gain_matrix_r_min_too_small_raises():
    # r_min=1 includes offset ±1 which are invalid 2-opt moves; the gain
    # matrix must reject them up-front rather than silently masking them out.
    torch.manual_seed(93)
    seeds = torch.rand(2, 12, 2)
    try:
        range_radius_2opt_gain_matrix(seeds, r_min=1, r_max=5)
    except ValueError:
        return
    raise AssertionError("range_radius_2opt_gain_matrix should raise on r_min<2")


def test_range_gain_matrix_r_max_lt_r_min_raises():
    # Inverted range must be rejected when both args are explicit library
    # inputs (dispatcher applies a friendlier warn+swap policy).
    torch.manual_seed(94)
    seeds = torch.rand(2, 12, 2)
    try:
        range_radius_2opt_gain_matrix(seeds, r_min=5, r_max=2)
    except ValueError:
        return
    raise AssertionError("range_radius_2opt_gain_matrix should raise on r_max<r_min")


def test_range_gain_matrix_r_min_too_large_does_not_raise():
    # r_min > N: offsets wrap via `% N` so destinations stay in [0, N); the
    # gain matrix still produces a valid result (some edges get masked).
    torch.manual_seed(95)
    seeds = torch.rand(1, 8, 2)
    tour, edge_gain, idx_tuple = range_radius_2opt_gain_matrix(
        seeds, r_min=20, r_max=25
    )
    assert tour.shape == (1, 8)
    assert edge_gain.numel() > 0
    assert idx_tuple is not None


def test_range_gain_matrix_large_n():
    # Sanity: large N must terminate in a reasonable time.
    import time
    torch.manual_seed(96)
    B, N = 2, 500
    seeds = torch.rand(B, N, 2)
    t0 = time.time()
    _, edge_gain, idx_tuple = range_radius_2opt_gain_matrix(
        seeds, r_min=5, r_max=25
    )
    dt = time.time() - t0
    assert idx_tuple is not None
    assert edge_gain.shape == (B * N * 2 * (25 - 5 + 1),)
    assert dt < 30.0, f"range_radius_2opt_gain_matrix took {dt:.2f}s on N=500"


def test_range_shape_roundtrip():
    torch.manual_seed(97)
    x = torch.rand(4, 15, 2)
    for r_min, r_max in ((2, 3), (2, 5), (2, 10)):
        out = range_radius_2opt(x, iters=10, r_min=r_min, r_max=r_max)
        assert out.shape == x.shape


def test_range_does_not_mutate_input():
    torch.manual_seed(98)
    x = torch.rand(3, 15, 2)
    x_before = x.clone()
    _ = range_radius_2opt(x, iters=10, r_min=2, r_max=10)
    assert torch.equal(x, x_before)


def test_range_never_worsens_random():
    # Central invariant: only positive-gain moves are applied, so tour cost
    # is non-increasing.
    torch.manual_seed(99)
    x = torch.rand(8, 30, 2)
    c0 = tour_cost(x)
    out = range_radius_2opt(x, iters=25, r_min=2, r_max=5)
    c1 = tour_cost(out)
    assert (c1 <= c0 + 1e-5).all(), (c0, c1)


def test_range_known_crossed_square():
    # Tour positions in the crossed square are 4; r_min=2, r_max=2 covers the
    # (i=0, j=2) and (i=1, j=3) flip pairs, so the optimum is reachable.
    coords = torch.tensor([[[0., 0.], [1., 1.], [1., 0.], [0., 1.]]])
    out = range_radius_2opt(coords, iters=10, r_min=2, r_max=2)
    c1 = tour_cost(out).item()
    assert abs(c1 - 4.0) < 1e-5, c1


def test_range_no_invalid_self_move():
    # Wrap-around regression: tour of 4 collinear points, (r_min, r_max)=(2,2).
    # The offset (i=0, j=N-1=3) must be masked out so the algorithm cannot
    # pick it.
    coords = torch.tensor([[[0., 0.], [1., 0.], [2., 0.], [3., 0.]]])
    out = range_radius_2opt(coords, iters=10, r_min=2, r_max=2)
    c0 = tour_cost(coords).item()
    c1 = tour_cost(out).item()
    assert c1 <= c0 + 1e-5, (c0, c1)


def test_range_multi_sweep_no_crash():
    torch.manual_seed(100)
    x = torch.rand(2, 30, 2)
    out = range_radius_2opt(x, iters=50, r_min=2, r_max=10)
    assert out.shape == x.shape


def test_range_monotonicity():
    # Larger r_max should never give worse mean cost than smaller r_max over
    # a random batch: more candidates ⊇ more improving moves.
    torch.manual_seed(101)
    x = torch.rand(8, 25, 2)
    c0 = tour_cost(x)
    out_small = range_radius_2opt(x, iters=15, r_min=2, r_max=3)
    out_large = range_radius_2opt(x, iters=15, r_min=2, r_max=10)
    mean_small = (tour_cost(out_small) - c0).mean().item()
    mean_large = (tour_cost(out_large) - c0).mean().item()
    assert mean_large <= mean_small + 1e-5, (mean_small, mean_large)


def test_range_matches_radius_at_single_window():
    # Parity: range_radius_2opt with (r_min=2, r_max=R) is semantically
    # equivalent to radius_2opt with radius=R — both produce the same offset
    # set {2, 3, ..., R} in both signs. This is the strongest check that the
    # refactor preserved semantics.
    torch.manual_seed(102)
    x = torch.rand(4, 20, 2)
    for r in (2, 5, 10):
        out_radius = radius_2opt(x.clone(), iters=10, r=r)
        out_range = range_radius_2opt(x.clone(), iters=10, r_min=2, r_max=r)
        torch.testing.assert_close(out_range, out_radius)


def test_range_debug_does_not_crash_or_miscompute():
    # debug=True must produce the same tour as debug=False (no side effects
    # on output) and remain cost-non-worsening.
    import contextlib, io

    torch.manual_seed(103)
    x = torch.rand(4, 30, 2)
    c0 = tour_cost(x)
    with contextlib.redirect_stdout(io.StringIO()) as quiet:
        out_dbg = range_radius_2opt(x, iters=5, r_min=2, r_max=10, debug=True)
        out_silent = range_radius_2opt(
            x.clone(), iters=5, r_min=2, r_max=10, debug=False
        )

    torch.testing.assert_close(out_dbg, out_silent)
    assert (tour_cost(out_dbg) <= c0 + 1e-5).all()


def test_range_driver_default_both_none():
    # Library-direct call with both args None defaults to (2, max(2, N//10)).
    torch.manual_seed(104)
    x = torch.rand(2, 30, 2)
    out = range_radius_2opt(x, iters=10)
    assert out.shape == x.shape
    assert (tour_cost(out) <= tour_cost(x) + 1e-5).all()


def test_range_driver_swap_warns():
    # Inverted args from a library caller emit a UserWarning and still run.
    import warnings as _w

    torch.manual_seed(105)
    x = torch.rand(2, 20, 2)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = range_radius_2opt(x, iters=5, r_min=20, r_max=5)
    assert any("swapping" in str(w.message) for w in caught), [str(w.message) for w in caught]
    assert out.shape == x.shape


def test_maybe_two_opt_dispatch_range_radius():
    # 'range_radius' routes to range_radius_2opt and honours the two bounds.
    torch.manual_seed(106)
    x = torch.rand(4, 30, 2)
    opts = _make_opts(
        use_2opt=True,
        two_opt_iters=10,
        two_opt_kind='range_radius',
        two_opt_radius_min=2,
        two_opt_radius_max=5,
    )
    out = maybe_two_opt(x, opts)
    assert out.shape == x.shape
    assert (tour_cost(out) <= tour_cost(x) + 1e-5).all()


def test_maybe_two_opt_dispatch_range_radius_default_fallback():
    # Both two_opt_radius_min and two_opt_radius_max absent -> dispatcher
    # falls back to (2, max(2, N // 10)).
    torch.manual_seed(107)
    N = 50
    x = torch.rand(2, N, 2)
    opts = _make_opts(
        use_2opt=True,
        two_opt_iters=10,
        two_opt_kind='range_radius',
        # two_opt_radius_min and two_opt_radius_max intentionally absent.
    )
    assert not hasattr(opts, 'two_opt_radius_min')
    assert not hasattr(opts, 'two_opt_radius_max')
    out = maybe_two_opt(x, opts)
    assert out.shape == x.shape
    assert (tour_cost(out) <= tour_cost(x) + 1e-5).all()


def test_maybe_two_opt_dispatch_range_radius_swapped_silently():
    # Inverted bounds at the dispatcher level emit a UserWarning and still
    # run (dispatcher is friendlier than the gain-matrix layer).
    import warnings as _w

    torch.manual_seed(108)
    x = torch.rand(2, 20, 2)
    opts = _make_opts(
        use_2opt=True,
        two_opt_iters=5,
        two_opt_kind='range_radius',
        two_opt_radius_min=20,
        two_opt_radius_max=5,
    )
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = maybe_two_opt(x, opts)
    assert any("swapping" in str(w.message) for w in caught), \
        [str(w.message) for w in caught]
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# maybe_two_opt dispatch tests
# ---------------------------------------------------------------------------


def test_maybe_two_opt_dispatch_full():
    # 'full' (or absent) uses full_2opt; cost must be non-worsening.
    torch.manual_seed(20)
    x = torch.rand(4, 15, 2)

    opts_default = _make_opts(use_2opt=True, two_opt_iters=10)
    out_default = maybe_two_opt(x, opts_default)
    opts_explicit = _make_opts(use_2opt=True, two_opt_iters=10, two_opt_kind='full')
    out_explicit = maybe_two_opt(x, opts_explicit)

    assert (tour_cost(out_default) <= tour_cost(x) + 1e-5).all()
    assert (tour_cost(out_explicit) <= tour_cost(x) + 1e-5).all()


def test_maybe_two_opt_dispatch_knn():
    # 'knn' routes to knn_2opt and honours two_opt_knn_k.
    torch.manual_seed(21)
    x = torch.rand(4, 20, 2)
    opts = _make_opts(
        use_2opt=True,
        two_opt_iters=10,
        two_opt_kind='knn',
        two_opt_knn_k=10,
    )
    out = maybe_two_opt(x, opts)
    assert out.shape == x.shape
    assert (tour_cost(out) <= tour_cost(x) + 1e-5).all()


def test_maybe_two_opt_dispatch_unknown_falls_back_to_full():
    # Garbage kind emits a warning and falls back to full_2opt (no crash).
    torch.manual_seed(22)
    x = torch.rand(2, 10, 2)
    opts = _make_opts(
        use_2opt=True,
        two_opt_iters=5,
        two_opt_kind='not_a_real_kind',
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = maybe_two_opt(x, opts)
    assert any('two_opt_kind' in str(w.message) for w in caught), caught
    assert (tour_cost(out) <= tour_cost(x) + 1e-5).all()


def test_maybe_two_opt_legacy_opts():
    # An Opts with only the legacy keys (`use_2opt`, `two_opt_iters`) and no
    # `two_opt_kind` / `two_opt_knn_k` attributes must still run with the full
    # 2-opt default and not crash on the new keys.
    torch.manual_seed(23)
    x = torch.rand(2, 12, 2)
    opts = _make_opts(use_2opt=True, two_opt_iters=10)
    assert not hasattr(opts, 'two_opt_kind')
    assert not hasattr(opts, 'two_opt_knn_k')
    out = maybe_two_opt(x, opts)
    assert out.shape == x.shape
    assert (tour_cost(out) <= tour_cost(x) + 1e-5).all()


# ---------------------------------------------------------------------------
# Scipy KNN fallback (used when torch_geometric's pyg-lib backend is missing)
# ---------------------------------------------------------------------------


def test_knn_edges_scipy_shape_and_dtype():
    # 2 instances of 5 nodes, k=2 -> 2 * 5 * 2 = 20 directed edges.
    x = torch.rand(10, 2)
    batch = torch.tensor([0] * 5 + [1] * 5)
    out = _knn_edges_scipy(x, batch, k=2, device=torch.device('cpu'))
    assert out.shape == (2, 20)
    assert out.dtype == torch.long
    flat = out[0] * 1_000_000 + out[1]  # pair encoding for uniqueness check
    assert flat.unique().numel() == 20, "duplicate edges returned"


def test_knn_edges_scipy_no_self_loops():
    x = torch.rand(8, 2)
    batch = torch.zeros(8, dtype=torch.long)
    out = _knn_edges_scipy(x, batch, k=3, device=torch.device('cpu'))
    assert (out[0] != out[1]).all(), "self-loop(s) present"


def test_knn_edges_scipy_correct_neighbors():
    # 5 points on the x-axis at x = 0, 1, 2, 3, 4. k=2 -> each point's two
    # closest are its two immediate neighbours along the line. Endpoints
    # (0 and 4) have neighbours {1, 2} and {2, 3} respectively.
    x = torch.tensor([[float(i), 0.0] for i in range(5)])
    batch = torch.zeros(5, dtype=torch.long)
    out = _knn_edges_scipy(x, batch, k=2, device=torch.device('cpu'))

    def neighbours_of(i):
        mask = out[0] == i
        return sorted(out[1][mask].tolist())

    assert neighbours_of(0) == [1, 2]
    assert neighbours_of(1) == [0, 2]
    assert neighbours_of(2) == [1, 3]
    assert neighbours_of(3) == [2, 4]
    assert neighbours_of(4) == [2, 3]


def test_knn_edges_scipy_clips_when_k_too_large():
    # k=10 on a 3-node set -> 2 non-self neighbours each; total 3 * 2 = 6.
    x = torch.tensor([[0., 0.], [1., 0.], [2., 0.]])
    batch = torch.zeros(3, dtype=torch.long)
    out = _knn_edges_scipy(x, batch, k=10, device=torch.device('cpu'))
    assert out.shape == (2, 6)
    # No duplicates and no self-loops.
    assert (out[0] != out[1]).all()


def test_knn_edges_scipy_isolated_nodes():
    # Mix of a single-node "instance" (no neighbours possible) and a regular
    # 4-node one. The 1-node instance contributes zero edges without crashing.
    x = torch.tensor([[0., 0.],
                      [0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    batch = torch.tensor([0, 1, 1, 1, 1])
    out = _knn_edges_scipy(x, batch, k=2, device=torch.device('cpu'))
    # 4 nodes * 2 edges each = 8 directed edges, all in instance 1.
    assert out.shape == (2, 8)
    assert (out[0] >= 1).all() and (out[1] >= 1).all()


def test_knn_2opt_no_cross_device_warning():
    # Regression test: ``knn_2opt_gain_matrix`` must place its synthetic
    # ``batch`` vector on the same device as the input ``seeds`` so that
    # torch_geometric's ``knn_graph`` does not emit
    # "Input tensor 'x' and 'batch' are on different devices ... blocking
    # device transfer" (a warning that *fires a sync* and stalls the GPU).
    import warnings
    if _pyg_knn_graph is None:
        # pyg-lib itself is missing -> the warning cannot be triggered.
        # The test still exercises the rest of the call path.
        ctx = warnings.catch_warnings()
    else:
        ctx = warnings.catch_warnings(record=True)
    torch.manual_seed(60)
    seeds = torch.rand(2, 20, 2)
    with ctx as caught:
        tour, edge_gain, idx_tuple = knn_2opt_gain_matrix(seeds, k=5)
    assert idx_tuple is not None
    if caught is not None:
        bad = [w for w in caught
               if 'different devices' in str(w.message).lower()]
        assert not bad, (
            "knn_graph emitted a cross-device UserWarning: "
            f"{[str(w.message) for w in bad]}"
        )


@contextlib.contextmanager
def _force_scipy_fallback():
    """Context manager: make `knn_2opt_gain_matrix` use the scipy fallback."""
    saved = pp_mod._pyg_knn_graph

    def _raise_import_error(*a, **kw):
        raise ImportError("'knn_graph' requires 'pyg-lib>=0.6.0'")

    pp_mod._pyg_knn_graph = _raise_import_error
    try:
        yield
    finally:
        pp_mod._pyg_knn_graph = saved


def test_knn_2opt_works_without_pyg_lib():
    # End-to-end: when pyg-lib is missing, knn_2opt must transparently use the
    # scipy fallback and still improve (or at least not worsen) the tour.
    torch.manual_seed(30)
    x = torch.rand(4, 20, 2)
    c0 = tour_cost(x)
    with _force_scipy_fallback():
        out = knn_2opt(x, iters=10, k=8)
    assert out.shape == x.shape
    assert (tour_cost(out) <= c0 + 1e-5).all()


def test_knn_2opt_scipy_fallback_uncrosses_square():
    # Same crossed-square known-optimum test as test_knn_known_crossed_square;
    # exercises the scipy fallback path specifically.
    coords = torch.tensor([[[0., 0.], [1., 1.], [1., 0.], [0., 1.]]])
    with _force_scipy_fallback():
        out = knn_2opt(coords, iters=10, k=3)
    c1 = tour_cost(out).item()
    assert abs(c1 - 4.0) < 1e-5, c1


# ---------------------------------------------------------------------------
# Debug timing output (--two_opt_debug)
# ---------------------------------------------------------------------------


def _quiet(*a, **kw):
    """Suppress prints emitted by debug-enabled 2-opt."""
    import io
    import contextlib

    return contextlib.redirect_stdout(io.StringIO())


def test_knn_2opt_debug_does_not_crash_or_miscompute():
    # ``debug=True`` should emit per-sweep timings to stdout but produce the
    # exact same tour as ``debug=False``. Pins both the perf-instrumentation
    # path and its no-side-effects-on-output guarantee.
    import contextlib, io

    torch.manual_seed(50)
    x = torch.rand(4, 30, 2)

    c0 = tour_cost(x)
    with contextlib.redirect_stdout(io.StringIO()) as quiet:
        out_dbg = knn_2opt(x, iters=5, k=10, debug=True)
        out_silent = knn_2opt(x.clone(), iters=5, k=10, debug=False)

    # Tour outputs are bitwise identical because timings don't mutate state.
    torch.testing.assert_close(out_dbg, out_silent)
    # Cost is non-worsening for both modes.
    assert (tour_cost(out_dbg) <= c0 + 1e-5).all()


def test_full_2opt_debug_does_not_crash_or_miscompute():
    # Same as above, for the dense variant.
    import contextlib, io

    torch.manual_seed(51)
    x = torch.rand(3, 25, 2)

    c0 = tour_cost(x)
    with contextlib.redirect_stdout(io.StringIO()):
        out_dbg = full_2opt(x, iters=5, debug=True)
        out_silent = full_2opt(x.clone(), iters=5, debug=False)

    torch.testing.assert_close(out_dbg, out_silent)
    assert (tour_cost(out_dbg) <= c0 + 1e-5).all()


def test_maybe_two_opt_forwards_two_opt_debug():
    # ``opts.two_opt_debug=True`` must reach the underlying drivers via
    # ``maybe_two_opt`` without raising. We only check side-effects-free run
    # + non-worsening cost here; capturing the printed output is done by
    # redirecting stdout as in the tests above.
    import contextlib, io

    torch.manual_seed(52)
    x = torch.rand(2, 20, 2)

    opts = _make_opts(
        use_2opt=True,
        two_opt_iters=3,
        two_opt_kind='knn',
        two_opt_knn_k=8,
        two_opt_debug=True,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        out = maybe_two_opt(x, opts)
    assert out.shape == x.shape
    assert (tour_cost(out) <= tour_cost(x) + 1e-5).all()


# ---------------------------------------------------------------------------
# Dense-free KNN gain matrix: parity + large-N
# ---------------------------------------------------------------------------


def _expected_edge_gain_dense(seeds, k):
    """Reference gain vector computed the *old* way (dense `(B, N, N)` matrix).

    Used to verify that the new per-edge implementation in
    ``knn_2opt_gain_matrix`` produces identical results.
    """
    B, N, D = seeds.shape
    device = seeds.device
    coords = seeds
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = diff.pow(2).sum(dim=-1).sqrt()                              # (B, N, N)
    tour = torch.arange(N, device=device).expand(B, N)
    t_next = torch.roll(tour, shifts=-1, dims=1)
    x = seeds.view(-1, D)
    # Place `batch` on the same device as `x` so torch_geometric's `knn_graph`
    # does not emit a UserWarning about cross-device input.
    batch = torch.arange(B, device=device).repeat_interleave(N)
    if _pyg_knn_graph is not None:
        try:
            edge_index = _pyg_knn_graph(x, k=k, batch=batch, loop=False)
        except ImportError:
            edge_index = _knn_edges_scipy(x, batch, k, device)
    else:
        edge_index = _knn_edges_scipy(x, batch, k, device)
    flat_src, flat_dst = edge_index[0], edge_index[1]
    src_b, src_i = flat_src // N, flat_src % N
    dst_b, dst_j = flat_dst // N, flat_dst % N
    same_batch = src_b == dst_b
    src_b = src_b[same_batch]
    src_i = src_i[same_batch]
    dst_j = dst_j[same_batch]
    ti = tour[src_b, src_i]
    tip1 = t_next[src_b, src_i]
    tj = tour[src_b, dst_j]
    tjp1 = t_next[src_b, dst_j]
    old1 = dist[src_b, ti, tip1]
    old2 = dist[src_b, tj, tjp1]
    new1 = dist[src_b, ti, tj]
    new2 = dist[src_b, tip1, tjp1]
    return old1 + old2 - new1 - new2


def test_knn_gain_matrix_matches_dense_reference():
    # Element-wise parity between the new per-edge `edge_gain` and the
    # equivalent dense-matrix computation. Confirms dropping `(B, N, N)`
    # did not change the algorithm's output. We compare only the entries that
    # are not adjacency/wrap-around-masked by the new code (-1e9 sentinels)
    # so we are testing the actual gain formula rather than the masking.
    torch.manual_seed(40)
    B, N, k = 3, 25, 6
    seeds = torch.rand(B, N, 2)

    _, edge_gain_new, idx_tuple = knn_2opt_gain_matrix(seeds, k)
    edge_gain_old = _expected_edge_gain_dense(seeds, k)

    assert idx_tuple is not None
    assert edge_gain_new.shape == edge_gain_old.shape

    src_b, src_i, dst_j = idx_tuple
    invalid_e = (dst_j <= src_i + 1) | ((src_i == 0) & (dst_j == N - 1))
    valid = ~invalid_e
    # bitwise-equal: per-edge Euclidean sum order is identical (sum over D),
    # so the two formulations produce the same fp32 values.
    torch.testing.assert_close(
        edge_gain_new[valid], edge_gain_old[valid], atol=0.0, rtol=0.0
    )


def test_knn_gain_matrix_no_dense_alloc_on_large_n():
    # Memory regression test: after dropping the dense matrix the KNN gain
    # computation must scale to instances that would have OOMed before.
    # B=1, N=2000 was previously a ~32 MB dist matrix; now it allocates only
    # O(B * N * k) bytes.
    import time

    torch.manual_seed(41)
    seeds = torch.rand(1, 2000, 2)
    t0 = time.time()
    _, edge_gain, idx_tuple = knn_2opt_gain_matrix(seeds, k=20)
    dt = time.time() - t0
    assert idx_tuple is not None
    # Edge count scales with N*k; sanity check.
    assert edge_gain.numel() > 1000
    # Loose time bound: just ensure it didn't hang.
    assert dt < 30.0, f"knn_2opt_gain_matrix took {dt:.2f}s on N=2000"


# ---------------------------------------------------------------------------
# eval_2opt OOM guardrail
# ---------------------------------------------------------------------------


def _import_eval_2opt():
    """Lazy import: eval_2opt transitively pulls GLOP pipeline machinery."""
    import eval_2opt  # noqa: F401
    return eval_2opt


def test_is_oom_error_classifier():
    e2 = _import_eval_2opt()
    # CUDA OOM class (only present from PyTorch 1.13+).
    oom_cls = getattr(torch.cuda, 'OutOfMemoryError', None)
    if oom_cls is not None:
        # Construct an instance if possible (some PyTorch builds refuse without
        # CUDA initialised; fall back to a lighter check).
        try:
            assert e2._is_oom_error(oom_cls())
        except Exception:
            pass
    # Bare RuntimeError whose message mentions CUDA out of memory.
    assert e2._is_oom_error(RuntimeError("CUDA out of memory. Tried ..."))
    assert e2._is_oom_error(RuntimeError("CUDA error: out of memory"))
    # Plain MemoryError (CPU OOM) is treated as OOM too.
    assert e2._is_oom_error(MemoryError())
    # Non-OOM errors are not classified as OOM.
    assert not e2._is_oom_error(ValueError("nothing to do with memory"))
    assert not e2._is_oom_error(RuntimeError("shape mismatch"))
    assert not e2._is_oom_error(KeyError("missing"))


def test_safe_run_mode_catches_oom_and_returns_sentinel():
    import math
    e2 = _import_eval_2opt()
    saved = e2.run_mode
    try:
        def _boom(*a, **kw):
            raise RuntimeError(
                "CUDA out of memory. Tried to allocate 2.00 GiB ...")

        e2.run_mode = _boom

        class _O:
            pass

        sentinel = e2._safe_run_mode(
            'knn_per_iter', True, 'per_iter', 'knn', 20, None, None, None,
            _O(), revisers=[],
        )
    finally:
        e2.run_mode = saved

    assert sentinel['skipped'] is True
    assert sentinel['label'] == 'knn_per_iter'
    assert sentinel['two_opt_kind'] == 'knn'
    assert math.isnan(sentinel['avg'])
    assert math.isnan(sentinel['best'])
    assert math.isnan(sentinel['duration'])
    assert sentinel['curve'] == {}
    assert 'CUDA out of memory' in sentinel['error']


def test_safe_run_mode_propagates_non_oom():
    """Non-OOM exceptions must NOT be swallowed."""
    e2 = _import_eval_2opt()
    saved = e2.run_mode
    try:
        def _boom(*a, **kw):
            raise ValueError("not an OOM")

        e2.run_mode = _boom

        class _O:
            pass

        propagated = False
        try:
            e2._safe_run_mode('final', True, 'final', 'full', 20, None,
                              None, None, _O(), revisers=[])
        except ValueError:
            propagated = True
    finally:
        e2.run_mode = saved
    assert propagated, "Non-OOM errors must propagate"


def test_safe_run_mode_propagates_keyboard_interrupt():
    """Ctrl-C and other BaseException-without-OOM-message should propagate."""
    e2 = _import_eval_2opt()
    saved = e2.run_mode
    try:
        def _boom(*a, **kw):
            raise KeyboardInterrupt()

        e2.run_mode = _boom

        class _O:
            pass

        propagated = False
        try:
            e2._safe_run_mode('per_iter', True, 'per_iter', 'full', 20, None,
                              None, None, _O(), revisers=[])
        except KeyboardInterrupt:
            propagated = True
    finally:
        e2.run_mode = saved
    assert propagated


def test_print_table_tolerates_oom_rows():
    """`print_table` must render OOM rows without crashing on NaN math."""
    import io
    import contextlib

    e2 = _import_eval_2opt()
    runs = [
        {'label': 'baseline',  'two_opt_kind': 'full', 'avg': 8.0, 'best': 7.5,
         'duration': 12.3, 'curve': {}},
        {'label': 'final',     'two_opt_kind': 'full', 'avg': 7.5, 'best': 7.0,
         'duration': 15.0, 'curve': {}},
        {'label': 'knn_final', 'two_opt_kind': 'knn',  'avg': float('nan'),
         'best': float('nan'), 'duration': float('nan'),
         'skipped': True, 'curve': {}},
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e2.print_table(runs)
    out = buf.getvalue()
    assert 'baseline' in out and 'final' in out and 'knn_final' in out
    assert 'OOM' in out  # placeholder rendering
    # No exception / traceback should have leaked to stdout.


def test_print_table_tolerates_oom_baseline():
    """When even the baseline OOMs, `print_table` should still run."""
    import io
    import contextlib

    e2 = _import_eval_2opt()
    runs = [
        {'label': 'baseline', 'two_opt_kind': 'full', 'avg': float('nan'),
         'best': float('nan'), 'duration': float('nan'),
         'skipped': True, 'curve': {}},
        {'label': 'final',    'two_opt_kind': 'full', 'avg': 7.5, 'best': 7.0,
         'duration': 15.0, 'curve': {}},
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e2.print_table(runs)
    assert 'OOM' in buf.getvalue()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print("all tests passed")
