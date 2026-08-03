"""Post-processing functions for improving GLOP solutions.

This module implements a batched, fully-vectorized (GPU/CPU) **2-opt** local
search for Euclidean TSP tours, used as an optional refinement step on top of the
GLOP pipeline.

Tour representation
    A tour is a coordinate tensor ``seeds`` of shape ``(B, N, D)`` whose row
    order along axis 1 *is* the visiting order (the same representation the GLOP
    pipeline carries in ``utils/functions.py``). Distances are Euclidean and are
    recomputed from the coordinates on demand; no precomputed distance matrix is
    required.

Algorithm
    A 2-opt move removes two tour edges ``(t_i, t_{i+1})`` and ``(t_j, t_{j+1})``
    and reconnects them as ``(t_i, t_j)`` and ``(t_{i+1}, t_{j+1})``, reversing
    the segment in between. The gain of a move is::

        gain = c(t_i, t_{i+1}) + c(t_j, t_{j+1}) - c(t_i, t_j) - c(t_{i+1}, t_{j+1})

    ``full_2opt_gain_matrix`` evaluates this gain for every ``(i, j)`` pair at
    once as an ``(B, N, N)`` tensor, masking invalid pairs (adjacent, reversed,
    or the wrap-around edge). ``full_2opt`` then repeatedly applies the single
    best-gain move per instance (best-improvement) until no improving move
    remains or ``iters`` sweeps have run. Because only positive-gain moves are
    ever applied, the tour cost is non-increasing.

Complexity / limitations
    * ``full_2opt_gain_matrix`` materializes an ``O(B * N^2)`` gain matrix
      and so targets small-to-moderate ``N`` (a few hundred).
    * ``knn_2opt_gain_matrix`` is the K-sparse counterpart: both the candidate
      set and the per-edge distance computation scale as ``O(B * N * k)``,
      with ``k`` typically 10-30. Recommended for ``N >~ 500``.
    * ``radius_2opt_gain_matrix`` is the *tour-position* sparse counterpart:
      the candidate set is purely a function of position offsets
      ``s in {±2, ..., ±r} mod N``, so the candidate set and per-edge
      computation both scale as ``O(B * N * r)`` with ``r`` typically
      ``~ N/10``. No KNN graph is built — useful when adjacent-in-tour
      candidates are most promising and Euclidean KNN would over-include
      distant edges.
    * ``range_radius_2opt_gain_matrix`` generalises ``radius_2opt_gain_matrix``
      from a fixed window ``[2, r]`` to ``[r_min, r_max]``. Per-call
      complexity is ``O(B * N * (r_max - r_min + 1))``. Useful when the
      best offset range is neither anchored at 2 nor at a fixed upper bound.
    * ``sampling_radius_2opt_gain_matrix`` is a *shifting-window* variant of
      ``range_radius_2opt_gain_matrix``: the offset window
      ``[base + shift, base + shift + r]`` is re-sampled each call, so a
      sequence of sweeps covers a broader effective offset range than a
      single static window, at per-call cost ``O(B * N * (r + 1))`` equal to
      ``range_radius_2opt_gain_matrix`` with the same ``r`` and ``r_min =
      base``.

Entry points
    * ``full_2opt(seeds, iters=10)`` — improve a batch of tours with a dense
      candidate set; returns coords. Targets small-to-moderate ``N``.
    * ``knn_2opt(seeds, iters=10, k=20)`` — k-nearest-neighbour-sparse variant.
      Each sweep considers only the KNN edges of the current coordinates, so
      both the candidate set *and* the per-edge distance computation scale as
      ``O(B * N * k)`` instead of ``O(B * N^2)``. Suitable for larger ``N``
      where the dense gain matrix is too costly. Edge gains are aggregated
      per-instance via ``scatter_max``.
    * ``radius_2opt(seeds, iters=10, r)`` — tour-position-sparse variant.
      Each sweep considers only edges whose *tour-position* offset is in
      ``{±2, ..., ±r} mod N`` (no Euclidean KNN), so the candidate set and
      per-edge distance computation both scale as ``O(B * N * r)``. ``r``
      must be supplied; the dispatcher defaults to ``~ N/10`` when unset.
      Returns coords. Edge gains are aggregated per-instance via
      ``scatter_max``.
    * ``range_radius_2opt(seeds, iters=10, r_min, r_max)`` — generalisation of
      ``radius_2opt`` whose offset window is ``[r_min, r_max]`` instead of
      ``[2, r]``. The candidate set and per-edge distance computation scale
      as ``O(B * N * (r_max - r_min + 1))``. ``r_min`` must be ``>= 2`` (offsets
      0 and ±1 are invalid 2-opt moves); the dispatcher defaults to
      ``(2, max(2, N // 10))`` when either bound is unset and warns + swaps
      if the bounds are inverted. Edge gains are aggregated per-instance via
      ``scatter_max``.
    * ``sampling_radius_2opt(seeds, iters=10, base, r)`` — shifting-window
      tour-position-sparse variant. Each sweep re-samples an offset shift
      and evaluates offsets in ``[base + shift, base + shift + r]`` (and
      their negatives) instead of a single fixed window. Per-call
      candidate cost is ``O(B * N * (r + 1))``. ``base`` must be ``>= 2``
      and ``base + r < N``; the dispatcher defaults to
      ``(2, max(2, N // 10))`` when either arg is unset and warns + swaps
      if the bounds are inverted. Edge gains are aggregated per-instance
      via ``scatter_max``.
    * ``hop_radius_2opt(seeds, iters=10, base, h)`` — hop-stride
      tour-position-sparse variant. Each sweep evaluates offsets
      ``{base, base+h, base+2h, ..., base+(k-1)*h}`` (and their negatives),
      where ``k = (N - base - 1) // 2 // h``. Successive sweeps use the
      same offset set (no random shift), so the per-call cost
      ``O(B * N * k)`` buys sparse coverage of a broader offset range
      than a single dense fixed window. ``base`` must be ``>= 2`` and
      ``h`` must be ``>= 1``; the dispatcher defaults to
      ``(2, max(2, N // 10))`` when either arg is unset. Edge gains are
      aggregated per-instance via ``scatter_max``.
    * ``decomp_2opt(seeds, iters=10, revision_len, r)`` — decomposition-aware
      variant that targets the edges *the GLOP revisor never re-optimises*:
      the seams between revisor windows. The candidate NODE set is every
      tour position within ``r`` hops of a decomposition boundary (centred
      on each boundary NODE position); the candidate EDGE set is the full
      ``locs × locs`` pairwise cross-product — every 2-opt move whose both
      removed edges lie near a seam, including cross-seam reconnections.
      Restricted to ``--two_opt_mode=per_iter`` because seams are only
      well-defined in the reassembled tour frame produced by ``LCP_TSP``.
      The dispatcher warns and skips in ``final`` mode. ``r`` defaults to
      ``max(2, revision_len // 10)`` when unset. Edge gains are aggregated
      per-instance via ``scatter_max``.
    * ``maybe_two_opt(seeds, opts, revision_len=None)`` — opts-gated wrapper
      used by the pipeline. Dispatches on ``opts.two_opt_kind`` (``"full"``,
      ``"knn"``, ``"radius"``, ``"range_radius"``, ``"sampling_radius"``,
      ``"hop_radius"``, or ``"decomp"``); unknown values fall back to
      ``"full"`` with a ``UserWarning``. ``revision_len`` is forwarded by
      the per-iter call site so the ``"decomp"`` kind knows which revisor
      layer it belongs to.

Implementation notes
    * The KNN gain matrix computes the four endpoint distances per edge
      directly from coords via a single ``(E, 4, D)`` advanced-index lookup —
      *no* dense ``(B, N, N)`` distance matrix is materialised.
    * The KNN graph itself is requested via
      ``torch_geometric.nn.knn_graph`` when its ``pyg-lib`` backend is
      available. When ``pyg-lib`` is missing or too old to satisfy the
      ``>=0.6.0`` version requirement (e.g. on PyTorch 1.13) the call
      transparently falls back to ``_knn_edges_scipy``, which builds
      per-instance ``scipy.spatial.cKDTree`` graphs on CPU. The two paths
      return identical ``(2, E)`` directed-edge tensors, so the rest of the
      gain computation is unaffected.

The pipeline wiring and CLI flags live in ``utils/functions.py`` (``reconnect``
and ``LCP_TSP``) and ``main.py`` respectively. CLI flags:

    * ``--use_2opt`` — master switch.
    * ``--two_opt_kind {full, knn, radius, range_radius, sampling_radius,
      hop_radius, decomp, decomp_sampling}`` — which algorithm to dispatch.
    * ``--two_opt_mode {final, per_iter}`` — run once after the whole pipeline
      (``"final"``) or after every revisor iteration (``"per_iter"``).
      ``"decomp"`` is only valid with ``"per_iter"``; the dispatcher warns
      and skips in ``"final"`` mode.
    * ``--two_opt_iters`` — sweeps per invocation (default 10).
    * ``--two_opt_knn_k`` — ``k`` for ``"knn"`` (default 20, ignored otherwise).
    * ``--two_opt_radius`` — ``r`` for ``"radius"`` (default ``None``, meaning
      10% of ``--problem_size`` with a floor of 2; ignored otherwise).
    * ``--two_opt_radius_min`` / ``--two_opt_radius_max`` — inclusive bounds
      of the offset window for ``"range_radius"`` (defaults ``2`` and
      ``None``; ``None`` for ``r_max`` means 10% of ``--problem_size``).
    * ``--two_opt_sampling_base`` / ``--two_opt_sampling_r`` — starting offset
      and window size for ``"sampling_radius"`` (defaults ``2`` and
      ``None``; ``None`` for ``r`` means 10% of ``--problem_size``).
    * ``--two_opt_hop_base`` / ``--two_opt_hop_h`` — starting offset and
      hop stride for ``"hop_radius"`` (defaults ``2`` and ``None``;
      ``None`` for ``h`` means 10% of ``--problem_size`` floored at 2;
      ignored unless ``two_opt_kind == "hop_radius"``).
    * ``--two_opt_decomp_radius`` — ``r`` (seam-neighbourhood half-width
      along the tour) for ``"decomp"`` (default ``None``, meaning
      ``max(2, revision_len // 10)``; ignored otherwise).
    * ``--two_opt_decomp_sampling_base`` / ``--two_opt_decomp_sampling_seam_radius``
      / ``--two_opt_decomp_sampling_candidate_r`` — starting offset,
      seam-neighbourhood half-width, and far-offset window size for
      ``"decomp_sampling"`` (defaults ``2``, ``None``, ``None``; ``None``
      for the latter two means ``max(2, revision_len // 10)`` and
      ``max(2, N // 10)`` respectively). ``"decomp_sampling"`` is only
      valid with ``--two_opt_mode=per_iter``; the dispatcher warns and
      skips in ``"final"`` mode.
    * ``--two_opt_debug`` — print per-sweep phase timings to stdout for
      performance investigation.

Tests
    ``tests/test_post_process.py`` covers both algorithms and the dispatch:
    30 tests including KNN shape / non-mutation / non-worsening / known-optimum
    crossed-square / multi-sweep stability / adjacency-mask regression /
    k-monotonicity / dense-free parity, all four ``maybe_two_opt`` dispatch
    branches, six ``_knn_edges_scipy`` correctness tests, three debug-mode
    tests that pin the ``--two_opt_debug`` instrumentation path, and a
    bitwise-parity test that confirms the dense-free KNN gain matrix matches
    the dense reference.

See the README section "2-opt post-processing".
"""

import time
import warnings

import torch
from torch_scatter import scatter_max

# torch_geometric's `knn_graph` requires `pyg-lib` (>=0.6.0), which in turn
# needs a recent PyTorch. For older PyTorch (e.g. 1.13) we fall back to a
# pure-scipy implementation that produces the same `(2, E)` edge format.
try:
    from torch_geometric.nn import knn_graph as _pyg_knn_graph
except ImportError:
    _pyg_knn_graph = None


def _cuda_sync_if_available():
    """Best-effort CUDA sync so timer deltas include device work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _print_debug_timings(
    algo, sweep_idx, sweep_times, totals, sweep_t0, n_accepted, n_edges=0
):
    """Emit a single-line per-sweep breakdown when ``--two_opt_debug`` is set.

    Format (whitespace-aligned, easy to grep)::
        [knn_2opt] sweep 3/10  total=120.4ms  edges=80000 accepted=8 |
            knn=42.1 filter=1.3 gather=2.0 distance=3.5 mask=0.4
            scatter_max=1.0 apply_loop=58.2 reorder=11.5
    """
    _cuda_sync_if_available()
    total_ms = (time.perf_counter() - sweep_t0) * 1000
    # Accumulate per-phase totals across sweeps.
    for k, v in sweep_times.items():
        totals[k] = totals.get(k, 0.0) + v
    parts = " ".join(f"{k}={v:.1f}" for k, v in sweep_times.items())
    edges_str = f" edges={n_edges}" if n_edges else ""
    acc_str = f" accepted={n_accepted}" if n_accepted else ""
    print(
        f"[{algo}] sweep {sweep_idx:<3} total={total_ms:7.1f}ms"
        f"{edges_str}{acc_str} | {parts}",
        flush=True,
    )


# Phase names printed by the debug helpers above. The driver functions
# iterate over this list when emitting totals, so keep it in sync with the
# keys populated in the gain-matrix / driver code.
_DEBUG_PHASE_KEYS = (
    "knn",
    "radius",
    "filter",
    "gather",
    "distance",
    "mask",
    "scatter_max",
    "apply_loop",
    "reorder",
)


def _knn_edges_scipy(x, batch_vec, k, device):
    """Per-instance KNN edges via scipy.cKDTree; no pyg-lib dependency.

    Mirrors ``torch_geometric.nn.knn_graph(x, k=k, batch=batch, loop=False)``:
    for every node in each batch instance, emit directed edges to its ``k``
    nearest neighbours (excluding self). Indices are in the global flat-index
    space that matches ``x.shape[0]`` so the existing post-processing code
    can use them unmodified.

    Args:
        x: ``(BN, D)`` coordinates in the global flat layout (any device).
        batch_vec: ``(BN,)`` integer batch / instance index per node.
        k: desired number of nearest neighbours per node.
        device: target device for the returned tensor.

    Returns:
        ``(2, E)`` ``int64`` tensor of ``(src, dst)`` pairs on ``device``.
        If no valid edges can be produced, returns an empty ``(2, 0)`` tensor.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)
    batch_np = batch_vec.detach().cpu().numpy()

    src_chunks, dst_chunks = [], []
    offset = 0
    for b in np.unique(batch_np):
        mask = batch_np == b
        coords_b = x_np[mask]
        N_b = coords_b.shape[0]
        if N_b <= 1:
            offset += N_b
            continue
        # Clip k to at most N_b - 1 to avoid self in the candidate list.
        actual_k = min(int(k), N_b - 1)
        tree = cKDTree(coords_b)
        # Query k+1 neighbours (self being closest), drop the self column.
        _, indices = tree.query(coords_b, k=actual_k + 1)
        src_chunks.append(np.repeat(np.arange(N_b), actual_k) + offset)
        dst_chunks.append(indices[:, 1:].reshape(-1) + offset)
        offset += N_b

    if not src_chunks:
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    src = np.concatenate(src_chunks)
    dst = np.concatenate(dst_chunks)
    edge_index = np.stack([src, dst])  # (2, E)
    return torch.from_numpy(edge_index).to(device).long()


def full_2opt_gain_matrix(seeds):
    # Evaluate the gain of all possible 2opt flips on a TSP tour
    # Euclidean TSP only!!!
    # Materialises an O(B * N^2) gain matrix; targets small-to-moderate N.
    # For larger instances use knn_2opt_gain_matrix (sparse candidates).
    device = seeds.device
    B, N, D = seeds.shape

    # Fixed coordinates and input tour
    coords = seeds  # (B, N, D)
    tour = torch.arange(N, device=device).expand(B, N).clone()  # (B, N), index

    # Compute the distance matrix between all pairs of nodes
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = diff.pow(2).sum(dim=-1).sqrt()  # (B, N, N)

    # Tour and successor in tour order
    t = tour  # (B, N), index
    t_next = torch.roll(t, shifts=-1, dims=1)  # (B, N), index, successor in closed tour
    batch_idx = torch.arange(B, device=device)[:, None]  # (B, 1)

    # Compute the 2-opt gain
    # c(i, i+1) + c(j, j+1) - c(i, j) - c(i+1, j+1)
    # Old edge costs
    # The advanced indexing here is equivalent to the loop (Note B is boardcasted to same shape):
    # for b in range(B):
    #   for i in range(N):
    #     old_edge[b, i] = dist[b, t[b, i], t_next[b, i]]
    old_edge = dist[batch_idx, t, t_next]

    # Expand to (B, N, N) to align i and j axes
    old_i_mat = old_edge.unsqueeze(2).expand(B, N, N)  # cost of edge at position i
    old_j_mat = old_edge.unsqueeze(1).expand(B, N, N)  # cost of edge at position j

    # Prepare node indices for new edges
    # ti, ti+1 have shape (B, N, 1); tj, tj+1 have shape (B, 1, N)
    ti = t.unsqueeze(2)  # (B, N, 1)
    tip1 = t_next.unsqueeze(2)  # (B, N, 1)
    tj = t.unsqueeze(1)  # (B, 1, N)
    tjp1 = t_next.unsqueeze(1)  # (B, 1, N)

    # Broadcast batch indices to (B, N, N)
    batch_idx_3d = batch_idx.unsqueeze(2).expand(B, N, N)  # (B, N, N)

    # New edges (t_i, t_j) and (t_{i+1}, t_{j+1}) Costs
    # The advanced indexing here is equivalent to the loop:
    # for b in range(B):
    #   for i in range(N):
    #       for j in range(N):
    #           new1[b, i, j] = dist[b, ti[b, i, j]=t[b, i], tj[b, i, j]=t[b, j]]
    new1 = dist[batch_idx_3d, ti.expand(B, N, N), tj.expand(B, N, N)]  # (B, N, N)
    new2 = dist[batch_idx_3d, tip1.expand(B, N, N), tjp1.expand(B, N, N)]  # (B, N, N)

    # Compute gain
    gain = old_i_mat + old_j_mat - new1 - new2  # (B, N, N)

    # Build mask function
    # Two opt swap with self or neighbour make no changes to results
    # Also, we consider the symmetry of Euclidean TSP
    i_idx = torch.arange(N, device=device)
    j_idx = torch.arange(N, device=device)
    I, J = torch.meshgrid(i_idx, j_idx, indexing="ij")

    # Invalid when:
    #  - j <= i+1  (adjacent or reversed)
    #  - (i == 0 and j == N-1) (wrap-around same edge)
    invalid = (J <= I + 1) | ((I == 0) & (J == N - 1))
    mask = ~invalid  # (N, N)

    # Optionally set invalid gains to a large negative value
    gain_masked = gain.masked_fill(~mask, -1e9)

    return dist, tour, gain_masked, mask


def full_2opt(seeds, iters=10, debug=False, record=None):
    # Run full 2opt.
    # seeds: (B, N, D) coordinates in tour order (the tour is implicit in row order).
    # Returns: improved (B, N, D) coordinates, still in tour order.
    # ``debug`` enables per-sweep phase-timing output to stdout, used by the
    # ``--two_opt_debug`` CLI flag for perf investigation.
    # ``record`` is an optional mutable list; when provided, |i_star - j_star|
    # is appended once per accepted move (best_gain > 0) on every accepted
    # sweep. Used by ``eval_2opt.py`` to characterise swap locality.
    B, N, D = seeds.shape

    # Work on a copy so we never mutate the caller's tensor.
    seeds = seeds.clone()
    times = {} if debug else None

    sweep_idx = 0
    while sweep_idx < iters:
        sweep_idx += 1
        sweep_times = {} if debug else None
        _cuda_sync_if_available()
        sweep_t0 = time.perf_counter()

        # Phase: gain matrix construction (O(B * N^2) memory).
        gain_t0 = time.perf_counter()
        dist, tour, gain, mask = full_2opt_gain_matrix(seeds)
        _cuda_sync_if_available()
        if debug:
            sweep_times["gain_matrix"] = (time.perf_counter() - gain_t0) * 1000

        # gain => (B, N, N); pick the single best flip per instance.
        best_gain, flat_idx = gain.view(B, -1).max(dim=1)  # => (B, N^2) => (B,)
        if (best_gain <= 0).all():
            if debug:
                _print_debug_timings(
                    "full_2opt", sweep_idx, sweep_times, times, sweep_t0, 0
                )
            break

        # Convert flat_idx back to (i, j)
        i_star = (flat_idx // N).tolist()
        j_star = (flat_idx % N).tolist()

        # Record the cyclic short-arc distance min(|i-j|, N-|i-j|) for each
        # accepted move. The TSP tour is closed, so the smaller of the
        # forward-hop and the wrap-around-hop is the natural locality measure.
        if record is not None:
            record.extend(
                min(abs(i_star[b] - j_star[b]), N - abs(i_star[b] - j_star[b]))
                for b in range(B)
                if best_gain[b] > 0
            )

        # Phase: per-batch Python loop applying segment reversal.
        apply_t0 = time.perf_counter()
        new_tour = tour.clone()
        n_accepted = 0
        for b in range(B):
            if best_gain[b] <= 0:
                continue
            i = i_star[b]
            j = j_star[b]
            t_b = tour[b]
            new_tour[b] = torch.cat(
                [t_b[: i + 1], torch.flip(t_b[i + 1 : j + 1], dims=[0]), t_b[j + 1 :]],
                dim=0,
            )
            n_accepted += 1
        _cuda_sync_if_available()
        if debug:
            sweep_times["apply_loop"] = (time.perf_counter() - apply_t0) * 1000

        # Phase: final seeds reorder via gather(1, ...).
        reorder_t0 = time.perf_counter()
        seeds = seeds.gather(1, new_tour.unsqueeze(-1).expand(B, N, D))
        _cuda_sync_if_available()
        if debug:
            sweep_times["reorder"] = (time.perf_counter() - reorder_t0) * 1000

        if debug:
            _print_debug_timings(
                "full_2opt", sweep_idx, sweep_times, times, sweep_t0, n_accepted
            )

    return seeds


def knn_2opt_gain_matrix(seeds, k, times=None):
    # Evaluate the gain of all possible 2opt flips on a TSP tour on a
    # k-nearest-neighbour graph. The dense `(B, N, N)` distance matrix is
    # intentionally avoided: only ~B*N*k candidate edges participate, so the
    # four distances per edge are computed directly from coords via a small
    # `(E, 4, D)` lookup. Memory is O(B*N*k) rather than O(B*N^2).
    # Euclidean TSP only!!!
    #
    # ``times`` is an optional mutable dict populated (when not None) with
    # per-phase millisecond timings: ``knn``, ``filter``, ``gather``,
    # ``distance``, ``mask``. Useful for `--two_opt_debug` profiling.
    device = seeds.device
    B, N, D = seeds.shape

    # Fixed coordinates and input tour
    coords = seeds  # (B, N, D)
    tour = torch.arange(N, device=device).expand(B, N).clone()  # (B, N), index

    # Tour and successor in tour order
    t = tour  # (B, N), index
    t_next = torch.roll(t, shifts=-1, dims=1)  # (B, N), index, successor in closed tour

    x = seeds.view(-1, D)  # (B*N, D), align with pyg knn_graph input
    # Place `batch` on the same device as `x` so torch_geometric's `knn_graph`
    # does not have to perform a blocking device transfer (which fires a
    # UserWarning and stalls the GPU pipeline).
    batch = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)

    # Phase: kNN graph construction.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    # kNN graph — use torch_geometric's `knn_graph` when its `pyg-lib`
    # backend is available (faster, GPU-friendly), otherwise fall back to the
    # pure-scipy implementation. Both produce `(2, E)` directed edges with
    # self-loops excluded, matching `(src, dst)` semantics.
    if _pyg_knn_graph is not None:
        try:
            edge_index = _pyg_knn_graph(x, k=k, batch=batch, loop=False)
        except ImportError:
            # pyg-lib missing at call time (PyTorch 1.13 etc.) -> fall back.
            edge_index = _knn_edges_scipy(x, batch, k, device)
    else:
        edge_index = _knn_edges_scipy(x, batch, k, device)
    _cuda_sync_if_available()
    if times is not None:
        times["knn"] = times.get("knn", 0.0) + (time.perf_counter() - t0) * 1000

    # Phase: convert to batched format + filter to within-tour edges.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    flat_src = edge_index[0]
    flat_dst = edge_index[1]
    src_b = flat_src // N
    src_i = flat_src % N
    dst_b = flat_dst // N
    dst_i = flat_dst % N

    same_batch = src_b == dst_b  # Boolean mask, (E,)
    src_b = src_b[same_batch]
    src_i = src_i[same_batch]
    dst_j = dst_i[same_batch]  # rename dst_i -> dst_j after filtering
    E = src_b.numel()
    _cuda_sync_if_available()
    if times is not None:
        times["filter"] = times.get("filter", 0.0) + (time.perf_counter() - t0) * 1000

    if E == 0:
        gain = torch.empty(B, 0, device=device)
        return tour, gain, None

    # Compute 2-opt gains for these (b, i, j).
    # Old edges: (t_i, t_{i+1}) and (t_j, t_{j+1})
    ti = t[src_b, src_i]  # advanced indexing per edge
    tip1 = t_next[src_b, src_i]
    tj = t[src_b, dst_j]
    tjp1 = t_next[src_b, dst_j]

    # Phase: gather (E, 4, D) endpoint coords.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    node_idx = torch.stack([ti, tip1, tj, tjp1], dim=1)  # (E, 4)
    pts = coords[src_b[:, None], node_idx]  # (E, 4, D)
    _cuda_sync_if_available()
    if times is not None:
        times["gather"] = times.get("gather", 0.0) + (time.perf_counter() - t0) * 1000

    # Phase: per-edge distance computation (4 lengths).
    _cuda_sync_if_available()
    t0 = time.perf_counter()

    def _len(a, b):
        return (pts[:, a] - pts[:, b]).pow(2).sum(dim=-1).sqrt()

    old1 = _len(0, 1)  # |t_i -- t_{i+1}|
    old2 = _len(2, 3)  # |t_j -- t_{j+1}|
    new1 = _len(0, 2)  # |t_i -- t_j|
    new2 = _len(1, 3)  # |t_{i+1} -- t_{j+1}|
    edge_gain = old1 + old2 - new1 - new2  # (E,)
    _cuda_sync_if_available()
    if times is not None:
        times["distance"] = (
            times.get("distance", 0.0) + (time.perf_counter() - t0) * 1000
        )

    # Phase: adjacency / wrap-around mask.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    invalid_e = (dst_j <= src_i + 1) | ((src_i == 0) & (dst_j == N - 1))  # (E,)
    edge_gain = edge_gain.masked_fill(invalid_e, -1e9)
    _cuda_sync_if_available()
    if times is not None:
        times["mask"] = times.get("mask", 0.0) + (time.perf_counter() - t0) * 1000

    # Return per-edge gains plus the corresponding (b, i, j) for the driver.
    return tour, edge_gain, (src_b, src_i, dst_j)


def radius_2opt_gain_matrix(seeds, radius, times=None):
    # Evaluate the gain of 2-opt flips on a TSP tour for candidate edges
    # whose *tour-position* offset is in ``[2, radius]`` (and the symmetric
    # negative offsets). The candidate set is fully deterministic — no KNN
    # graph over the current coords — and the dense ``(B, N, N)`` distance
    # matrix is intentionally avoided: only the four endpoint coordinates
    # per edge participate. Memory is ``O(B * N * radius)``.
    #
    # Mirrors the return contract of ``knn_2opt_gain_matrix``:
    # ``(tour, edge_gain, (src_b, src_i, dst_j))`` so the same driver code
    # can aggregate per-instance gains via ``scatter_max``.
    #
    # ``times`` is an optional mutable dict populated (when not None) with
    # per-phase millisecond timings: ``radius``, ``gather``, ``distance``,
    # ``mask``. Useful for ``--two_opt_debug`` profiling.
    device = seeds.device
    B, N, D = seeds.shape

    # Fixed coordinates and input tour
    coords = seeds  # (B, N, D)
    tour = torch.arange(N, device=device).expand(B, N).clone()  # (B, N), index

    # Tour and successor in tour order
    t = tour  # (B, N), index
    t_next = torch.roll(t, shifts=-1, dims=1)  # (B, N), index, successor in closed tour

    # Phase: tour-position candidate construction.
    # For each base position i in [0, N), candidate destinations are
    # j = (i + s) mod N for s in {±2, ±3, ..., ±radius}. Offset 0 (self) and
    # ±1 (adjacent edges) are excluded — those would be invalid 2-opt moves
    # that the mask phase below would zero out anyway, so excluding them
    # up-front saves work and keeps the gain vector small.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    offsets = torch.arange(2, radius + 1, device=device)  # (R-1,)
    offsets = torch.cat([offsets, -offsets], dim=0)  # (2*(R-1),)
    I = torch.arange(N, device=device)  # (N,)
    J = (I.unsqueeze(1) + offsets) % N  # (N, 2*(R-1))
    I_tiled = I.unsqueeze(1).expand_as(J)  # (N, 2*(R-1))

    # The (src_i, dst_j) edges are the same for every batch instance, so
    # tile them B times to give one (src_i, dst_j) block per batch element.
    # Order: batch-major, then row-major within a block (matches knn_2opt).
    src_i = I_tiled.reshape(-1).repeat(B)  # (E,)
    dst_j = J.reshape(-1).repeat(B)  # (E,)
    E_per_b = N * 2 * (radius - 1)
    src_b = torch.arange(B, device=device).repeat_interleave(E_per_b)
    _cuda_sync_if_available()
    if times is not None:
        times["radius"] = times.get("radius", 0.0) + (time.perf_counter() - t0) * 1000

    # Compute 2-opt gains for these (b, i, j).
    # Old edges: (t_i, t_{i+1}) and (t_j, t_{j+1})
    ti = t[src_b, src_i]  # advanced indexing per edge
    tip1 = t_next[src_b, src_i]
    tj = t[src_b, dst_j]
    tjp1 = t_next[src_b, dst_j]

    # Phase: gather (E, 4, D) endpoint coords.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    node_idx = torch.stack([ti, tip1, tj, tjp1], dim=1)  # (E, 4)
    pts = coords[src_b[:, None], node_idx]  # (E, 4, D)
    _cuda_sync_if_available()
    if times is not None:
        times["gather"] = times.get("gather", 0.0) + (time.perf_counter() - t0) * 1000

    # Phase: per-edge distance computation (4 lengths).
    _cuda_sync_if_available()
    t0 = time.perf_counter()

    def _len(a, b):
        return (pts[:, a] - pts[:, b]).pow(2).sum(dim=-1).sqrt()

    old1 = _len(0, 1)  # |t_i -- t_{i+1}|
    old2 = _len(2, 3)  # |t_j -- t_{j+1}|
    new1 = _len(0, 2)  # |t_i -- t_j|
    new2 = _len(1, 3)  # |t_{i+1} -- t_{j+1}|
    edge_gain = old1 + old2 - new1 - new2  # (E,)
    _cuda_sync_if_available()
    if times is not None:
        times["distance"] = (
            times.get("distance", 0.0) + (time.perf_counter() - t0) * 1000
        )

    # Phase: adjacency / wrap-around mask.
    # Offsets ±1 were excluded at construction time, so most edges pass, but
    # the wrap-around pair (i=0, j=N-1) still surfaces once per instance via
    # the +s=N-1 offset on i=0 and must be masked.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    invalid_e = (dst_j <= src_i + 1) | ((src_i == 0) & (dst_j == N - 1))  # (E,)
    edge_gain = edge_gain.masked_fill(invalid_e, -1e9)
    _cuda_sync_if_available()
    if times is not None:
        times["mask"] = times.get("mask", 0.0) + (time.perf_counter() - t0) * 1000

    # Return per-edge gains plus the corresponding (b, i, j) for the driver.
    return tour, edge_gain, (src_b, src_i, dst_j)


def radius_2opt(seeds, iters=10, r=None, debug=False, record=None):
    # Best-improvement 2-opt with a *tour-position* sparse candidate set.
    # Mirrors ``knn_2opt`` line-for-line, swapping the KNN-graph candidate
    # builder for the offsets-only builder in ``radius_2opt_gain_matrix``.
    #
    # ``r`` is the half-width of the candidate window: for each tour
    # position ``i`` the candidates are ``j = (i + s) mod N`` for
    # ``s in {±2, ..., ±r}``. ``r`` must be provided (no implicit default
    # — ``maybe_two_opt`` applies a 10%-of-N fallback at the dispatcher).
    device = seeds.device
    B, N, D = seeds.shape

    seeds = seeds.clone()

    # ``times`` accumulates per-phase ms across the whole run; reset each sweep
    # by summing existing keys. Used for `--two_opt_debug` profiling.
    times = {} if debug else None

    sweep_idx = 0
    while sweep_idx < iters:
        sweep_idx += 1
        gain_times = {} if debug else None
        _cuda_sync_if_available()
        sweep_t0 = time.perf_counter()

        tour, edge_gain, idx_tuple = radius_2opt_gain_matrix(
            seeds=seeds, radius=r, times=gain_times
        )
        if idx_tuple is None or edge_gain.numel() == 0:
            if debug:
                _print_debug_timings(
                    "radius_2opt", sweep_idx, gain_times, times, sweep_t0, 0
                )
            break

        src_b, src_i, dst_j = idx_tuple
        n_edges = edge_gain.numel()

        # Phase: scatter_max + best_i / best_j extraction.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        best_gain, argmax = scatter_max(src=edge_gain, index=src_b)
        best_i = src_i[argmax]
        best_j = dst_j[argmax]
        _cuda_sync_if_available()
        if debug:
            gain_times["scatter_max"] = (time.perf_counter() - t0) * 1000

        # Early exit with no 2-opt gains
        if (best_gain <= 0).all():
            if debug:
                _print_debug_timings(
                    "radius_2opt",
                    sweep_idx,
                    gain_times,
                    times,
                    sweep_t0,
                    0,
                    n_edges=n_edges,
                )
            break

        # Record the cyclic short-arc distance min(|i-j|, N-|i-j|) for each
        # accepted move. Vectorised: best_i/best_j are already (B,) tensors
        # on-device. The TSP tour is closed, so we take the smaller of the
        # forward-hop and the wrap-around-hop.
        if record is not None:
            accepted = best_gain > 0
            if accepted.any():
                d = (best_i - best_j).abs()
                d = torch.minimum(d, N - d)
                record.extend(d[accepted].tolist())

        # Phase: per-batch Python loop applying the segment reversal.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        new_tour = tour.clone()
        n_accepted = 0
        for b in range(B):
            if best_gain[b] <= 0:
                continue
            i = best_i[b].item()
            j = best_j[b].item()
            t_b = tour[b]
            new_tour[b] = torch.cat(
                [t_b[: i + 1], torch.flip(t_b[i + 1 : j + 1], dims=[0]), t_b[j + 1 :]],
                dim=0,
            )
            n_accepted += 1
        _cuda_sync_if_available()
        if debug:
            gain_times["apply_loop"] = (time.perf_counter() - t0) * 1000

        # Phase: final seeds reorder via gather(1, ...).
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        seeds = seeds.gather(1, new_tour.unsqueeze(-1).expand(B, N, D))
        _cuda_sync_if_available()
        if debug:
            gain_times["reorder"] = (time.perf_counter() - t0) * 1000

        if debug:
            _print_debug_timings(
                "radius_2opt",
                sweep_idx,
                gain_times,
                times,
                sweep_t0,
                n_accepted,
                n_edges=n_edges,
            )

    return seeds


def range_radius_2opt_gain_matrix(seeds, r_min, r_max, times=None):
    # Evaluate the gain of 2-opt flips on a TSP tour for candidate edges
    # whose *tour-position* offset is in ``[r_min, r_max]`` (and the symmetric
    # negative offsets). Generalises ``radius_2opt_gain_matrix`` from a fixed
    # window ``[2, radius]`` to an arbitrary ``[r_min, r_max]``. The candidate
    # set is fully deterministic — no KNN graph over the current coords — and
    # the dense ``(B, N, N)`` distance matrix is intentionally avoided: only
    # the four endpoint coordinates per edge participate. Memory is
    # ``O(B * N * (r_max - r_min + 1))``.
    #
    # Mirrors the return contract of ``radius_2opt_gain_matrix`` and
    # ``knn_2opt_gain_matrix``: ``(tour, edge_gain, (src_b, src_i, dst_j))``
    # so the same driver code can aggregate per-instance gains via
    # ``scatter_max``.
    #
    # ``times`` is an optional mutable dict populated (when not None) with
    # per-phase millisecond timings: ``radius``, ``gather``, ``distance``,
    # ``mask``. Useful for ``--two_opt_debug`` profiling.
    if r_min < 2:
        raise ValueError(
            f"range_radius_2opt_gain_matrix: r_min={r_min} must be >= 2 "
            "(offsets 0 and ±1 are invalid 2-opt moves)."
        )
    if r_max < r_min:
        raise ValueError(
            f"range_radius_2opt_gain_matrix: r_max={r_max} < r_min={r_min}."
        )

    device = seeds.device
    B, N, D = seeds.shape

    # Fixed coordinates and input tour
    coords = seeds  # (B, N, D)
    tour = torch.arange(N, device=device).expand(B, N).clone()  # (B, N), index

    # Tour and successor in tour order
    t = tour  # (B, N), index
    t_next = torch.roll(t, shifts=-1, dims=1)  # (B, N), index, successor in closed tour

    # Phase: tour-position candidate construction.
    # For each base position i in [0, N), candidate destinations are
    # j = (i + s) mod N for s in {±r_min, ±(r_min+1), ..., ±r_max}. Offsets
    # outside [2, radius] would be invalid 2-opt moves that the mask phase
    # below would zero out anyway, so excluding them up-front saves work and
    # keeps the gain vector small.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    offsets = torch.arange(
        r_min, r_max + 1, device=device
    )  # (R,) where R = r_max - r_min + 1
    offsets = torch.cat([offsets, -offsets], dim=0)  # (2*R,)
    I = torch.arange(N, device=device)  # (N,)
    J = (I.unsqueeze(1) + offsets) % N  # (N, 2*R)
    I_tiled = I.unsqueeze(1).expand_as(J)  # (N, 2*R)

    # The (src_i, dst_j) edges are the same for every batch instance, so
    # tile them B times to give one (src_i, dst_j) block per batch element.
    # Order: batch-major, then row-major within a block (matches knn_2opt).
    src_i = I_tiled.reshape(-1).repeat(B)  # (E,)
    dst_j = J.reshape(-1).repeat(B)  # (E,)
    E_per_b = N * 2 * (r_max - r_min + 1)
    src_b = torch.arange(B, device=device).repeat_interleave(E_per_b)
    _cuda_sync_if_available()
    if times is not None:
        times["radius"] = times.get("radius", 0.0) + (time.perf_counter() - t0) * 1000

    # Compute 2-opt gains for these (b, i, j).
    # Old edges: (t_i, t_{i+1}) and (t_j, t_{j+1})
    ti = t[src_b, src_i]  # advanced indexing per edge
    tip1 = t_next[src_b, src_i]
    tj = t[src_b, dst_j]
    tjp1 = t_next[src_b, dst_j]

    # Phase: gather (E, 4, D) endpoint coords.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    node_idx = torch.stack([ti, tip1, tj, tjp1], dim=1)  # (E, 4)
    pts = coords[src_b[:, None], node_idx]  # (E, 4, D)
    _cuda_sync_if_available()
    if times is not None:
        times["gather"] = times.get("gather", 0.0) + (time.perf_counter() - t0) * 1000

    # Phase: per-edge distance computation (4 lengths).
    _cuda_sync_if_available()
    t0 = time.perf_counter()

    def _len(a, b):
        return (pts[:, a] - pts[:, b]).pow(2).sum(dim=-1).sqrt()

    old1 = _len(0, 1)  # |t_i -- t_{i+1}|
    old2 = _len(2, 3)  # |t_j -- t_{j+1}|
    new1 = _len(0, 2)  # |t_i -- t_j|
    new2 = _len(1, 3)  # |t_{i+1} -- t_{j+1}|
    edge_gain = old1 + old2 - new1 - new2  # (E,)
    _cuda_sync_if_available()
    if times is not None:
        times["distance"] = (
            times.get("distance", 0.0) + (time.perf_counter() - t0) * 1000
        )

    # Phase: adjacency / wrap-around mask.
    # The window [r_min, r_max] always excludes offset ±1 (r_min >= 2 by
    # construction), but the wrap-around pair (i=0, j=N-1) still surfaces
    # once per instance via the +s=N-1 offset on i=0 and must be masked.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    invalid_e = (dst_j <= src_i + 1) | ((src_i == 0) & (dst_j == N - 1))  # (E,)
    edge_gain = edge_gain.masked_fill(invalid_e, -1e9)
    _cuda_sync_if_available()
    if times is not None:
        times["mask"] = times.get("mask", 0.0) + (time.perf_counter() - t0) * 1000

    # Return per-edge gains plus the corresponding (b, i, j) for the driver.
    return tour, edge_gain, (src_b, src_i, dst_j)


def range_radius_2opt(
    seeds, iters=10, r_min=None, r_max=None, debug=False, record=None
):
    # Best-improvement 2-opt with a *tour-position* sparse candidate set
    # whose offset window is ``[r_min, r_max]``. Mirrors ``radius_2opt``
    # line-for-line, swapping the fixed-radius candidate builder for the
    # range-based builder in ``range_radius_2opt_gain_matrix``.
    #
    # ``r_min`` and ``r_max`` are the inclusive bounds of the offset window:
    # for each tour position ``i`` the candidates are ``j = (i + s) mod N``
    # for ``s in {±r_min, ..., ±r_max}``. ``r_min`` must be >= 2 (offsets
    # 0 and ±1 are invalid 2-opt moves and are masked out anyway). Library
    # callers may pass both args; the dispatcher (``maybe_two_opt``) fills
    # sensible defaults ``(2, max(2, N // 10))`` when called from the CLI.
    device = seeds.device
    B, N, D = seeds.shape

    # Library-direct default fallback. The dispatcher applies its own
    # policy first; this catches library callers (e.g. notebooks) that
    # pass neither arg.
    if r_min is None and r_max is None:
        r_min, r_max = 2, max(2, N // 10)
    elif r_min is None:
        r_min = 2
    elif r_max is None:
        r_max = max(r_min, max(2, N // 10))
    if r_min > r_max:
        warnings.warn(
            f"range_radius_2opt: r_min={r_min} > r_max={r_max}; swapping",
            stacklevel=2,
        )
        r_min, r_max = r_max, r_min

    seeds = seeds.clone()

    # ``times`` accumulates per-phase ms across the whole run; reset each sweep
    # by summing existing keys. Used for `--two_opt_debug` profiling.
    times = {} if debug else None

    sweep_idx = 0
    while sweep_idx < iters:
        sweep_idx += 1
        gain_times = {} if debug else None
        _cuda_sync_if_available()
        sweep_t0 = time.perf_counter()

        tour, edge_gain, idx_tuple = range_radius_2opt_gain_matrix(
            seeds=seeds, r_min=r_min, r_max=r_max, times=gain_times
        )
        if idx_tuple is None or edge_gain.numel() == 0:
            if debug:
                _print_debug_timings(
                    "range_radius_2opt", sweep_idx, gain_times, times, sweep_t0, 0
                )
            break

        src_b, src_i, dst_j = idx_tuple
        n_edges = edge_gain.numel()

        # Phase: scatter_max + best_i / best_j extraction.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        best_gain, argmax = scatter_max(src=edge_gain, index=src_b)
        best_i = src_i[argmax]
        best_j = dst_j[argmax]
        _cuda_sync_if_available()
        if debug:
            gain_times["scatter_max"] = (time.perf_counter() - t0) * 1000

        # Early exit with no 2-opt gains
        if (best_gain <= 0).all():
            if debug:
                _print_debug_timings(
                    "range_radius_2opt",
                    sweep_idx,
                    gain_times,
                    times,
                    sweep_t0,
                    0,
                    n_edges=n_edges,
                )
            break

        # Record the cyclic short-arc distance min(|i-j|, N-|i-j|) for each
        # accepted move. Vectorised: best_i/best_j are already (B,) tensors
        # on-device. The TSP tour is closed, so we take the smaller of the
        # forward-hop and the wrap-around-hop.
        if record is not None:
            accepted = best_gain > 0
            if accepted.any():
                d = (best_i - best_j).abs()
                d = torch.minimum(d, N - d)
                record.extend(d[accepted].tolist())

        # Phase: per-batch Python loop applying the segment reversal.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        new_tour = tour.clone()
        n_accepted = 0
        for b in range(B):
            if best_gain[b] <= 0:
                continue
            i = best_i[b].item()
            j = best_j[b].item()
            t_b = tour[b]
            new_tour[b] = torch.cat(
                [t_b[: i + 1], torch.flip(t_b[i + 1 : j + 1], dims=[0]), t_b[j + 1 :]],
                dim=0,
            )
            n_accepted += 1
        _cuda_sync_if_available()
        if debug:
            gain_times["apply_loop"] = (time.perf_counter() - t0) * 1000

        # Phase: final seeds reorder via gather(1, ...).
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        seeds = seeds.gather(1, new_tour.unsqueeze(-1).expand(B, N, D))
        _cuda_sync_if_available()
        if debug:
            gain_times["reorder"] = (time.perf_counter() - t0) * 1000

        if debug:
            _print_debug_timings(
                "range_radius_2opt",
                sweep_idx,
                gain_times,
                times,
                sweep_t0,
                n_accepted,
                n_edges=n_edges,
            )

    return seeds


def sampling_radius_2opt_gain_matrix(seeds, base, r, times=None):
    # Evaluate the gain of 2-opt flips on a TSP tour for candidate edges
    # whose *tour-position* offset lies in a shifting window
    # ``[base + shift, base + shift + r]`` (and the symmetric negative
    # offsets). The shift ``shift`` is sampled once per call from
    # ``[0, N - base - r)`` so the window slides across the offset range
    # across successive sweeps, broadening effective coverage at lower
    # per-call cost than a single wide static window. The candidate set
    # is otherwise identical to ``range_radius_2opt_gain_matrix`` — no
    # KNN graph over the current coords, no dense ``(B, N, N)`` distance
    # matrix — and only the four endpoint coordinates per edge
    # participate. Memory is ``O(B * N * (r + 1))``.
    #
    # Mirrors the return contract of ``radius_2opt_gain_matrix``,
    # ``range_radius_2opt_gain_matrix``, and ``knn_2opt_gain_matrix``:
    # ``(tour, edge_gain, (src_b, src_i, dst_j))`` so the same driver
    # code can aggregate per-instance gains via ``scatter_max``.
    #
    # ``times`` is an optional mutable dict populated (when not None) with
    # per-phase millisecond timings: ``radius``, ``gather``, ``distance``,
    # ``mask``. Useful for ``--two_opt_debug`` profiling.
    if base < 2:
        raise ValueError(
            f"sampling_radius_2opt_gain_matrix: base={base} must be >= 2 "
            "(offsets 0 and ±1 are invalid 2-opt moves)."
        )
    if r < 2:
        raise ValueError(
            f"sampling_radius_2opt_gain_matrix: r={r} must be >= 2 "
            "(offsets 0 and ±1 are invalid 2-opt moves)."
        )

    device = seeds.device
    B, N, D = seeds.shape

    if base + r >= N:
        raise ValueError(
            f"sampling_radius_2opt_gain_matrix: base + r = {base + r} "
            f"must be < N = {N} so the sampled shifting window has room."
        )

    # Fixed coordinates and input tour
    coords = seeds  # (B, N, D)
    tour = torch.arange(N, device=device).expand(B, N).clone()  # (B, N), index

    # Tour and successor in tour order
    t = tour  # (B, N), index
    t_next = torch.roll(t, shifts=-1, dims=1)  # (B, N), index, successor in closed tour

    _cuda_sync_if_available()
    t0 = time.perf_counter()
    # Sample a single shared shift per call: candidates for all batches
    # in this call look at offsets [base + shift, base + shift + r] (and
    # their negatives). Across successive calls / sweeps the driver
    # re-samples, broadening effective coverage of the offset range.
    # We clamp the upper bound to >= 1 so ``torch.randint`` always gets a
    # valid range, and fall back to shift=0 when the window already
    # saturates the available room (base + r == N - 1).
    shift_max = max(0, N - base - r)
    if shift_max > 0:
        shift = int(torch.randint(0, shift_max, size=()).item())
    else:
        shift = 0

    # Phase: tour-position candidate construction with shifting window.
    # Offsets are ``base + shift, base + shift + 1, ..., base + shift + r``
    # (and their negatives). Offsets 0 and ±1 are valid indices in the
    # range but are invalid 2-opt moves, so the mask phase below zeros
    # them out.
    offsets = torch.arange(
        base + shift, base + shift + r + 1, device=device
    )  # (R,) where R = r + 1
    offsets = torch.cat([offsets, -offsets], dim=0)  # (2*R,)
    I = torch.arange(N, device=device)  # (N,)
    J = (I.unsqueeze(1) + offsets) % N  # (N, 2*R)
    I_tiled = I.unsqueeze(1).expand_as(J)  # (N, 2*R)

    # The (src_i, dst_j) edges are the same for every batch instance, so
    # tile them B times to give one (src_i, dst_j) block per batch element.
    # Order: batch-major, then row-major within a block (matches knn_2opt).
    src_i = I_tiled.reshape(-1).repeat(B)  # (E,)
    dst_j = J.reshape(-1).repeat(B)  # (E,)
    E_per_b = N * 2 * (r + 1)
    src_b = torch.arange(B, device=device).repeat_interleave(E_per_b)
    _cuda_sync_if_available()
    if times is not None:
        times["radius"] = times.get("radius", 0.0) + (time.perf_counter() - t0) * 1000

    # Compute 2-opt gains for these (b, i, j).
    # Old edges: (t_i, t_{i+1}) and (t_j, t_{j+1})
    ti = t[src_b, src_i]  # advanced indexing per edge
    tip1 = t_next[src_b, src_i]
    tj = t[src_b, dst_j]
    tjp1 = t_next[src_b, dst_j]

    # Phase: gather (E, 4, D) endpoint coords.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    node_idx = torch.stack([ti, tip1, tj, tjp1], dim=1)  # (E, 4)
    pts = coords[src_b[:, None], node_idx]  # (E, 4, D)
    _cuda_sync_if_available()
    if times is not None:
        times["gather"] = times.get("gather", 0.0) + (time.perf_counter() - t0) * 1000

    # Phase: per-edge distance computation (4 lengths).
    _cuda_sync_if_available()
    t0 = time.perf_counter()

    def _len(a, b):
        return (pts[:, a] - pts[:, b]).pow(2).sum(dim=-1).sqrt()

    old1 = _len(0, 1)  # |t_i -- t_{i+1}|
    old2 = _len(2, 3)  # |t_j -- t_{j+1}|
    new1 = _len(0, 2)  # |t_i -- t_j|
    new2 = _len(1, 3)  # |t_{i+1} -- t_{j+1}|
    edge_gain = old1 + old2 - new1 - new2  # (E,)
    _cuda_sync_if_available()
    if times is not None:
        times["distance"] = (
            times.get("distance", 0.0) + (time.perf_counter() - t0) * 1000
        )

    # Phase: adjacency / wrap-around mask.
    # The sampled window always excludes offset ±1 (base >= 2 by
    # construction), but the wrap-around pair (i=0, j=N-1) still
    # surfaces once per instance via the +s=N-1 offset on i=0 and must
    # be masked.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    invalid_e = (dst_j <= src_i + 1) | ((src_i == 0) & (dst_j == N - 1))  # (E,)
    edge_gain = edge_gain.masked_fill(invalid_e, -1e9)
    _cuda_sync_if_available()
    if times is not None:
        times["mask"] = times.get("mask", 0.0) + (time.perf_counter() - t0) * 1000

    # Return per-edge gains plus the corresponding (b, i, j) for the driver.
    return tour, edge_gain, (src_b, src_i, dst_j)


def sampling_radius_2opt(seeds, iters=10, base=None, r=None, debug=False, record=None):
    # Best-improvement 2-opt with a *tour-position* sparse candidate set
    # whose offset window ``[base + shift, base + shift + r]`` shifts
    # randomly each sweep (the shift is re-sampled by
    # ``sampling_radius_2opt_gain_matrix``). Mirrors ``range_radius_2opt``
    # line-for-line, swapping the fixed-window candidate builder for the
    # shifting-window builder.
    #
    # ``base`` is the inclusive lower bound of the offset window; must
    # be >= 2 (offsets 0 and ±1 are invalid 2-opt moves and are masked
    # out anyway). ``r`` is the window size beyond ``base``: the upper
    # bound is ``base + r`` (inclusive). Library callers may pass both
    # args; the dispatcher (``maybe_two_opt``) fills sensible defaults
    # ``(2, max(2, N // 10))`` when called from the CLI.
    device = seeds.device
    B, N, D = seeds.shape

    # Library-direct default fallback. The dispatcher applies its own
    # policy first; this catches library callers (e.g. notebooks) that
    # pass neither arg.
    if base is None and r is None:
        base, r = 2, max(2, N // 10)
    elif base is None:
        base = 2
    elif r is None:
        r = max(base, max(2, N // 10))
    if base > r:
        warnings.warn(
            f"sampling_radius_2opt: base={base} > r={r}; swapping",
            stacklevel=2,
        )
        base, r = r, base

    seeds = seeds.clone()

    # ``times`` accumulates per-phase ms across the whole run; reset each sweep
    # by summing existing keys. Used for `--two_opt_debug` profiling.
    times = {} if debug else None

    sweep_idx = 0
    while sweep_idx < iters:
        sweep_idx += 1
        gain_times = {} if debug else None
        _cuda_sync_if_available()
        sweep_t0 = time.perf_counter()

        tour, edge_gain, idx_tuple = sampling_radius_2opt_gain_matrix(
            seeds=seeds, base=base, r=r, times=gain_times
        )
        if idx_tuple is None or edge_gain.numel() == 0:
            if debug:
                _print_debug_timings(
                    "sampling_radius_2opt",
                    sweep_idx,
                    gain_times,
                    times,
                    sweep_t0,
                    0,
                )
            break

        src_b, src_i, dst_j = idx_tuple
        n_edges = edge_gain.numel()

        # Phase: scatter_max + best_i / best_j extraction.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        best_gain, argmax = scatter_max(src=edge_gain, index=src_b)
        best_i = src_i[argmax]
        best_j = dst_j[argmax]
        _cuda_sync_if_available()
        if debug:
            gain_times["scatter_max"] = (time.perf_counter() - t0) * 1000

        # Early exit with no 2-opt gains
        if (best_gain <= 0).all():
            if debug:
                _print_debug_timings(
                    "sampling_radius_2opt",
                    sweep_idx,
                    gain_times,
                    times,
                    sweep_t0,
                    0,
                    n_edges=n_edges,
                )
            break

        # Record the cyclic short-arc distance min(|i-j|, N-|i-j|) for each
        # accepted move. Vectorised: best_i/best_j are already (B,) tensors
        # on-device. The TSP tour is closed, so we take the smaller of the
        # forward-hop and the wrap-around-hop.
        if record is not None:
            accepted = best_gain > 0
            if accepted.any():
                d = (best_i - best_j).abs()
                d = torch.minimum(d, N - d)
                record.extend(d[accepted].tolist())

        # Phase: per-batch Python loop applying the segment reversal.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        new_tour = tour.clone()
        n_accepted = 0
        for b in range(B):
            if best_gain[b] <= 0:
                continue
            i = best_i[b].item()
            j = best_j[b].item()
            t_b = tour[b]
            new_tour[b] = torch.cat(
                [t_b[: i + 1], torch.flip(t_b[i + 1 : j + 1], dims=[0]), t_b[j + 1 :]],
                dim=0,
            )
            n_accepted += 1
        _cuda_sync_if_available()
        if debug:
            gain_times["apply_loop"] = (time.perf_counter() - t0) * 1000

        # Phase: final seeds reorder via gather(1, ...).
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        seeds = seeds.gather(1, new_tour.unsqueeze(-1).expand(B, N, D))
        _cuda_sync_if_available()
        if debug:
            gain_times["reorder"] = (time.perf_counter() - t0) * 1000

        if debug:
            _print_debug_timings(
                "sampling_radius_2opt",
                sweep_idx,
                gain_times,
                times,
                sweep_t0,
                n_accepted,
                n_edges=n_edges,
            )

    return seeds


def hop_radius_2opt_gain_matrix(seeds, base, h, times=None):
    # Evaluate the gain of 2-opt flips on a TSP tour for candidate edges
    # whose *tour-position* offset is *hop-strided*: every ``h`` hops
    # starting at ``base`` — i.e. offsets
    # ``{base, base+h, base+2h, ..., base+(k-1)*h}`` (and the symmetric
    # negative offsets), where ``k = (N - base - 1) // 2 // h``. This
    # is the hop-stride analogue of ``sampling_radius_2opt_gain_matrix``:
    # instead of a contiguous window that shifts per call, the candidate
    # set is a regular strided sample over the same ``[base, N//2]``
    # budget — so the per-call cost is ``O(B * N * k)`` with ``k``
    # playing the role of ``r + 1`` from sampling_radius. Successive
    # sweeps use the same offset set (no random shift), so the benefit
    # comes from sparse coverage of a broader offset range at lower
    # per-call cost than a single dense fixed window.
    #
    # Mirrors the return contract of every other sparse variant:
    # ``(tour, edge_gain, (src_b, src_i, dst_j))`` so the same driver
    # code can aggregate per-instance gains via ``scatter_max``.
    #
    # ``times`` is an optional mutable dict populated (when not None)
    # with per-phase millisecond timings: ``radius``, ``gather``,
    # ``distance``, ``mask``. Useful for ``--two_opt_debug`` profiling.
    if base < 2:
        raise ValueError(
            f"hop_radius_2opt_gain_matrix: base={base} must be >= 2 "
            "(offsets 0 and ±1 are invalid 2-opt moves)."
        )
    if h < 1:
        raise ValueError(
            f"hop_radius_2opt_gain_matrix: h={h} must be >= 1 "
            "(hop stride must be positive so the offset set is non-empty)."
        )

    device = seeds.device
    B, N, D = seeds.shape

    # Fixed coordinates and input tour
    coords = seeds  # (B, N, D)
    tour = torch.arange(N, device=device).expand(B, N).clone()  # (B, N), index

    # Tour and successor in tour order
    t = tour  # (B, N), index
    t_next = torch.roll(t, shifts=-1, dims=1)  # (B, N), index, successor in closed tour

    # Phase: hop-stride candidate construction.
    # Per base position i in [0, N), candidate destinations are
    # j = (i + s) mod N for s in {base, base+h, base+2h, ..., base+(k-1)*h}
    # (and their negatives). Offsets are regularly strided over the
    # same [base, N//2] budget that range_radius covers densely, so
    # k plays the role of (r + 1) from sampling_radius.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    k = (N - base - 1) // (2 * h)  # 0-index number of positive offsets
    # sample index that are within a certain hop from each other
    idx = base + h * torch.arange(k, device=device)
    offsets = torch.cat([idx, -idx], dim=0)  # (2*k,)
    I = torch.arange(N, device=device)  # (N,)
    J = (I.unsqueeze(1) + offsets) % N  # (N, 2*k)
    I_tiled = I.unsqueeze(1).expand_as(J)  # (N, 2*k)

    # Empty-candidate guard (mirror the decomp_2opt "L < 2" early-exit):
    # when (base, h, N) admits no hop, return empty so the driver
    # short-circuits.
    if k == 0:
        if times is not None:
            times["radius"] = (
                times.get("radius", 0.0) + (time.perf_counter() - t0) * 1000
            )
        return (
            tour,
            torch.empty(0, device=device),
            (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            ),
        )

    # The (src_i, dst_j) edges are the same for every batch instance, so
    # tile them B times to give one (src_i, dst_j) block per batch element.
    # Order: batch-major, then row-major within a block (matches knn_2opt).
    src_i = I_tiled.reshape(-1).repeat(B)  # (E,)
    dst_j = J.reshape(-1).repeat(B)  # (E,)
    E_per_b = N * 2 * k
    src_b = torch.arange(B, device=device).repeat_interleave(E_per_b)
    _cuda_sync_if_available()
    if times is not None:
        times["radius"] = times.get("radius", 0.0) + (time.perf_counter() - t0) * 1000

    # Compute 2-opt gains for these (b, i, j).
    # Old edges: (t_i, t_{i+1}) and (t_j, t_{j+1})
    ti = t[src_b, src_i]  # advanced indexing per edge
    tip1 = t_next[src_b, src_i]
    tj = t[src_b, dst_j]
    tjp1 = t_next[src_b, dst_j]

    # Phase: gather (E, 4, D) endpoint coords.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    node_idx = torch.stack([ti, tip1, tj, tjp1], dim=1)  # (E, 4)
    pts = coords[src_b[:, None], node_idx]  # (E, 4, D)
    _cuda_sync_if_available()
    if times is not None:
        times["gather"] = times.get("gather", 0.0) + (time.perf_counter() - t0) * 1000

    # Phase: per-edge distance computation (4 lengths).
    _cuda_sync_if_available()
    t0 = time.perf_counter()

    def _len(a, b):
        return (pts[:, a] - pts[:, b]).pow(2).sum(dim=-1).sqrt()

    old1 = _len(0, 1)  # |t_i -- t_{i+1}|
    old2 = _len(2, 3)  # |t_j -- t_{j+1}|
    new1 = _len(0, 2)  # |t_i -- t_j|
    new2 = _len(1, 3)  # |t_{i+1} -- t_{j+1}|
    edge_gain = old1 + old2 - new1 - new2  # (E,)
    _cuda_sync_if_available()
    if times is not None:
        times["distance"] = (
            times.get("distance", 0.0) + (time.perf_counter() - t0) * 1000
        )

    # Phase: adjacency / wrap-around mask.
    # The hop set ``base, base+h, ...`` always excludes offset ±1
    # (``base >= 2`` by construction), but the wrap-around pair
    # (i=0, j=N-1) still surfaces once per instance via the +s=N-1
    # offset on i=0 and must be masked.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    invalid_e = (dst_j <= src_i + 1) | ((src_i == 0) & (dst_j == N - 1))  # (E,)
    edge_gain = edge_gain.masked_fill(invalid_e, -1e9)
    _cuda_sync_if_available()
    if times is not None:
        times["mask"] = times.get("mask", 0.0) + (time.perf_counter() - t0) * 1000

    # Return per-edge gains plus the corresponding (b, i, j) for the driver.
    return tour, edge_gain, (src_b, src_i, dst_j)


def hop_radius_2opt(seeds, iters=10, base=None, h=None, debug=False, record=None):
    # Best-improvement 2-opt with a *tour-position* sparse candidate set
    # whose offsets are *hop-strided* (every ``h`` hops, starting at
    # ``base``): the candidate set is
    # ``{base, base+h, base+2h, ..., base+(k-1)*h}`` and the symmetric
    # negative offsets, where ``k = (N - base - 1) // 2 // h``. Mirrors
    # ``sampling_radius_2opt`` line-for-line, swapping the shifting-window
    # candidate builder for the hop-stride builder.
    #
    # ``base`` is the smallest hop offset; must be >= 2 (offsets 0 and
    # ±1 are invalid 2-opt moves and are masked out anyway). ``h`` is
    # the hop stride — the gap between consecutive offsets. Library
    # callers may pass both args; the dispatcher (``maybe_two_opt``)
    # fills sensible defaults ``(2, max(2, N // 10))`` when called from
    # the CLI.
    device = seeds.device
    B, N, D = seeds.shape

    # Library-direct default fallback. The dispatcher applies its own
    # policy first; this catches library callers (e.g. notebooks) that
    # pass neither arg. There is no swap-and-warn when ``base > h``:
    # ``h`` is a stride, not a width, so the pairing is meaningful
    # rather than contradictory.
    if base is None and h is None:
        base, h = 2, max(2, N // 10)
    elif base is None:
        base = 2
    elif h is None:
        h = max(base, max(2, N // 10))

    seeds = seeds.clone()

    # ``times`` accumulates per-phase ms across the whole run; reset each sweep
    # by summing existing keys. Used for `--two_opt_debug` profiling.
    times = {} if debug else None

    sweep_idx = 0
    while sweep_idx < iters:
        sweep_idx += 1
        gain_times = {} if debug else None
        _cuda_sync_if_available()
        sweep_t0 = time.perf_counter()

        tour, edge_gain, idx_tuple = hop_radius_2opt_gain_matrix(
            seeds=seeds, base=base, h=h, times=gain_times
        )
        if idx_tuple is None or edge_gain.numel() == 0:
            if debug:
                _print_debug_timings(
                    "hop_radius_2opt",
                    sweep_idx,
                    gain_times,
                    times,
                    sweep_t0,
                    0,
                )
            break

        src_b, src_i, dst_j = idx_tuple
        n_edges = edge_gain.numel()

        # Phase: scatter_max + best_i / best_j extraction.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        best_gain, argmax = scatter_max(src=edge_gain, index=src_b)
        best_i = src_i[argmax]
        best_j = dst_j[argmax]
        _cuda_sync_if_available()
        if debug:
            gain_times["scatter_max"] = (time.perf_counter() - t0) * 1000

        # Early exit with no 2-opt gains
        if (best_gain <= 0).all():
            if debug:
                _print_debug_timings(
                    "hop_radius_2opt",
                    sweep_idx,
                    gain_times,
                    times,
                    sweep_t0,
                    0,
                    n_edges=n_edges,
                )
            break

        # Record the cyclic short-arc distance min(|i-j|, N-|i-j|) for each
        # accepted move. Vectorised: best_i/best_j are already (B,) tensors
        # on-device. The TSP tour is closed, so we take the smaller of the
        # forward-hop and the wrap-around-hop.
        if record is not None:
            accepted = best_gain > 0
            if accepted.any():
                d = (best_i - best_j).abs()
                d = torch.minimum(d, N - d)
                record.extend(d[accepted].tolist())

        # Phase: per-batch Python loop applying the segment reversal.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        new_tour = tour.clone()
        n_accepted = 0
        for b in range(B):
            if best_gain[b] <= 0:
                continue
            i = best_i[b].item()
            j = best_j[b].item()
            t_b = tour[b]
            new_tour[b] = torch.cat(
                [t_b[: i + 1], torch.flip(t_b[i + 1 : j + 1], dims=[0]), t_b[j + 1 :]],
                dim=0,
            )
            n_accepted += 1
        _cuda_sync_if_available()
        if debug:
            gain_times["apply_loop"] = (time.perf_counter() - t0) * 1000

        # Phase: final seeds reorder via gather(1, ...).
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        seeds = seeds.gather(1, new_tour.unsqueeze(-1).expand(B, N, D))
        _cuda_sync_if_available()
        if debug:
            gain_times["reorder"] = (time.perf_counter() - t0) * 1000

        if debug:
            _print_debug_timings(
                "hop_radius_2opt",
                sweep_idx,
                gain_times,
                times,
                sweep_t0,
                n_accepted,
                n_edges=n_edges,
            )

    return seeds


def knn_2opt(seeds, iters=10, k=20, debug=False, record=None):
    device = seeds.device
    B, N, D = seeds.shape

    seeds = seeds.clone()

    # ``times`` accumulates per-phase ms across the whole run; reset each sweep
    # by summing existing keys. Used for `--two_opt_debug` profiling.
    times = {} if debug else None

    sweep_idx = 0
    while sweep_idx < iters:
        sweep_idx += 1
        gain_times = {} if debug else None
        _cuda_sync_if_available()
        sweep_t0 = time.perf_counter()

        tour, edge_gain, idx_tuple = knn_2opt_gain_matrix(
            seeds=seeds, k=k, times=gain_times
        )
        if idx_tuple is None or edge_gain.numel() == 0:
            if debug:
                _print_debug_timings(
                    "knn_2opt", sweep_idx, gain_times, times, sweep_t0, 0
                )
            break

        src_b, src_i, dst_j = idx_tuple
        n_edges = edge_gain.numel()

        # Phase: scatter_max + best_i / best_j extraction.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        best_gain, argmax = scatter_max(src=edge_gain, index=src_b)
        best_i = src_i[argmax]
        best_j = dst_j[argmax]
        _cuda_sync_if_available()
        if debug:
            gain_times["scatter_max"] = (time.perf_counter() - t0) * 1000

        # Early exit with no 2-opt gains
        if (best_gain <= 0).all():
            if debug:
                _print_debug_timings(
                    "knn_2opt",
                    sweep_idx,
                    gain_times,
                    times,
                    sweep_t0,
                    0,
                    n_edges=n_edges,
                )
            break

        # Record the cyclic short-arc distance min(|i-j|, N-|i-j|) for each
        # accepted move. Vectorised: best_i/best_j are already (B,) tensors
        # on-device. The TSP tour is closed, so we take the smaller of the
        # forward-hop and the wrap-around-hop.
        if record is not None:
            accepted = best_gain > 0
            if accepted.any():
                d = (best_i - best_j).abs()
                d = torch.minimum(d, N - d)
                record.extend(d[accepted].tolist())

        # Phase: per-batch Python loop applying the segment reversal.
        # This is the suspected hotspot when many sweeps accept moves; the
        # .item() syncs + per-instance torch.cat are PyTorch-on-Python overhead.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        new_tour = tour.clone()
        n_accepted = 0
        for b in range(B):
            if best_gain[b] <= 0:
                continue
            i = best_i[b].item()
            j = best_j[b].item()
            t_b = tour[b]
            new_tour[b] = torch.cat(
                [t_b[: i + 1], torch.flip(t_b[i + 1 : j + 1], dims=[0]), t_b[j + 1 :]],
                dim=0,
            )
            n_accepted += 1
        _cuda_sync_if_available()
        if debug:
            gain_times["apply_loop"] = (time.perf_counter() - t0) * 1000

        # Phase: final seeds reorder via gather(1, ...).
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        seeds = seeds.gather(1, new_tour.unsqueeze(-1).expand(B, N, D))
        _cuda_sync_if_available()
        if debug:
            gain_times["reorder"] = (time.perf_counter() - t0) * 1000

        if debug:
            _print_debug_timings(
                "knn_2opt",
                sweep_idx,
                gain_times,
                times,
                sweep_t0,
                n_accepted,
                n_edges=n_edges,
            )

    return seeds


def decomp_2opt_gain_matrix(seeds, revision_len, radius, times=None):
    # Evaluate 2-opt gains for the *seam-focused* candidate set used by
    # ``decomp_2opt``. The candidate NODE set is every tour position within
    # ``radius`` hops (along the tour) of a decomposition boundary; the
    # candidate EDGE set is the full pairwise cross-product of those nodes
    # (``I = J = locs``). A 2-opt move is only considered when *both*
    # removed edges lie near a decomposition seam — the edges that the GLOP
    # revisor never re-optimizes because they sit across window boundaries.
    # Reconnections between different seams are reachable, so the search can
    # fix long-range seam crossings that the revisor's per-window
    # optimization cannot.
    #
    # Decomposition boundary NODE positions in the *reassembled* tour frame
    # that 2-opt consumes are at multiples of ``revision_len`` (plus one
    # trailing segment start when ``N % revision_len != 0``). See
    # ``utils/functions.py:decomposition`` and ``LCP_TSP``.
    #
    # Returns ``(tour, edge_gain, (src_b, src_i, dst_j))`` — the same
    # contract used by ``range_radius_2opt_gain_matrix`` so the driver can
    # share the per-instance ``scatter_max`` aggregation and per-batch
    # reversal loop. ``radius`` here is the half-width of the seam
    # neighbourhood along the tour (the number of hops on either side of
    # each boundary node).
    if revision_len is None:
        raise ValueError(
            "decomp_2opt_gain_matrix: revision_len must be provided "
            "(seams are defined per revisor layer)."
        )
    if radius < 2:
        raise ValueError(
            f"decomp_2opt_gain_matrix: radius={radius} must be >= 2 "
            "(offsets 0 and ±1 are invalid 2-opt moves)."
        )

    device = seeds.device
    B, N, D = seeds.shape

    coords = seeds  # (B, N, D)
    tour = torch.arange(N, device=device).expand(B, N).clone()  # (B, N), index
    t = tour
    t_next = torch.roll(t, shifts=-1, dims=1)  # successor in closed tour

    # --- Phase: candidate construction -----------------------------------
    # Boundary NODE positions: {0, RL, 2*RL, ..., (num_chunks-1)*RL}, plus
    # the trailing segment start ``N - offset`` when ``offset != 0``.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    offset = N % revision_len
    starts = torch.arange(0, N - offset + 1, revision_len, device=device)
    # r-radius NEIGHBOURHOOD ALONG THE TOUR around every boundary: every
    # position within ``radius`` hops of any boundary node, centred on the
    # boundary NODE position ``b``. Overlaps are deduplicated for cost —
    # the reachable move set is identical.
    delta = torch.arange(-radius, radius + 1, device=device)  # (2r+1,)
    locs = ((starts[:, None] + delta[None, :]) % N).reshape(-1)
    locs = torch.unique(locs)
    L = locs.numel()
    if L < 2:
        # Not enough candidates for any valid 2-opt move — return empty
        # gain so the driver short-circuits.
        if times is not None:
            times["radius"] = (
                times.get("radius", 0.0) + (time.perf_counter() - t0) * 1000
            )
        return (
            tour,
            torch.empty(0, device=device),
            (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            ),
        )

    # Candidate EDGES = all unordered pairs in `locs` (I = J = locs).
    # Mask the i >= j half so each pair is scored exactly once.
    ii, jj = torch.meshgrid(locs, locs, indexing="ij")  # (L, L) each
    flat_i = ii.reshape(-1)
    flat_j = jj.reshape(-1)
    keep = flat_j > flat_i
    src_i = flat_i[keep].repeat(B)  # (E,)
    dst_j = flat_j[keep].repeat(B)  # (E,)
    E_per_b = int(keep.sum().item())
    src_b = torch.arange(B, device=device).repeat_interleave(E_per_b)
    _cuda_sync_if_available()
    if times is not None:
        times["radius"] = times.get("radius", 0.0) + (time.perf_counter() - t0) * 1000

    # --- Phase: gather (E, 4, D) endpoint coords --------------------------
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    ti = t[src_b, src_i]
    tip1 = t_next[src_b, src_i]
    tj = t[src_b, dst_j]
    tjp1 = t_next[src_b, dst_j]
    node_idx = torch.stack([ti, tip1, tj, tjp1], dim=1)  # (E, 4)
    pts = coords[src_b[:, None], node_idx]  # (E, 4, D)
    _cuda_sync_if_available()
    if times is not None:
        times["gather"] = times.get("gather", 0.0) + (time.perf_counter() - t0) * 1000

    # --- Phase: per-edge gain computation --------------------------------
    _cuda_sync_if_available()
    t0 = time.perf_counter()

    def _len(a, b):
        return (pts[:, a] - pts[:, b]).pow(2).sum(dim=-1).sqrt()

    old1 = _len(0, 1)  # |t_i -- t_{i+1}|
    old2 = _len(2, 3)  # |t_j -- t_{j+1}|
    new1 = _len(0, 2)  # |t_i -- t_j|
    new2 = _len(1, 3)  # |t_{i+1} -- t_{j+1}|
    edge_gain = old1 + old2 - new1 - new2  # (E,)
    _cuda_sync_if_available()
    if times is not None:
        times["distance"] = (
            times.get("distance", 0.0) + (time.perf_counter() - t0) * 1000
        )

    # --- Phase: adjacency / wrap-around mask -----------------------------
    # Drop moves where j == i+1 (adjacent edges) and the wrap-around pair
    # (i=0, j=N-1). ``dst_j > src_i`` is already guaranteed by construction.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    invalid_e = ((dst_j <= src_i + 1) | ((src_i == 0) & (dst_j == N - 1))) | (
        dst_j <= src_i
    )
    edge_gain = edge_gain.masked_fill(invalid_e, -1e9)
    _cuda_sync_if_available()
    if times is not None:
        times["mask"] = times.get("mask", 0.0) + (time.perf_counter() - t0) * 1000

    return tour, edge_gain, (src_b, src_i, dst_j)


def decomp_2opt(seeds, iters=10, revision_len=None, r=None, debug=False, record=None):
    # Best-improvement 2-opt with the *seam-focused* candidate set built
    # in ``decomp_2opt_gain_matrix``. Mirrors ``range_radius_2opt`` (and
    # ``radius_2opt``) line-for-line; the only algorithmic difference is
    # how the candidate (b, i, j) edges are constructed.
    #
    # ``revision_len`` is the *current* revisor layer's window length and
    # defines where the seams land in the reassembled tour frame. It is
    # mandatory — seams have no meaning outside a revisor layer.
    #
    # ``r`` (the seam-neighbourhood half-width along the tour) defaults
    # to ``max(2, revision_len // 10)`` when unset, mirroring the CLI
    # default convention used by ``maybe_two_opt`` for the other sparse
    # variants.
    device = seeds.device
    B, N, D = seeds.shape

    if revision_len is None:
        raise ValueError(
            "decomp_2opt: revision_len must be provided "
            "(seams are defined per revisor layer)."
        )

    # Optional: DO NOT run for smaller revision_len
    # if revision_len == 20:
    #    return seeds

    # Optional: Cur r in half for revison_len==50
    # if revision_len == 50:
    #    r //= 2

    if r is None:
        r = max(2, revision_len // 10)

    seeds = seeds.clone()

    times = {} if debug else None

    sweep_idx = 0
    while sweep_idx < iters:
        sweep_idx += 1
        gain_times = {} if debug else None
        _cuda_sync_if_available()
        sweep_t0 = time.perf_counter()

        tour, edge_gain, idx_tuple = decomp_2opt_gain_matrix(
            seeds=seeds, revision_len=revision_len, radius=r, times=gain_times
        )
        if idx_tuple is None or edge_gain.numel() == 0:
            if debug:
                _print_debug_timings(
                    "decomp_2opt", sweep_idx, gain_times, times, sweep_t0, 0
                )
            break

        src_b, src_i, dst_j = idx_tuple
        n_edges = edge_gain.numel()

        # Phase: scatter_max + best_i / best_j extraction.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        best_gain, argmax = scatter_max(src=edge_gain, index=src_b)
        best_i = src_i[argmax]
        best_j = dst_j[argmax]
        _cuda_sync_if_available()
        if debug:
            gain_times["scatter_max"] = (time.perf_counter() - t0) * 1000

        # Early exit with no 2-opt gains.
        if (best_gain <= 0).all():
            if debug:
                _print_debug_timings(
                    "decomp_2opt",
                    sweep_idx,
                    gain_times,
                    times,
                    sweep_t0,
                    0,
                    n_edges=n_edges,
                )
            break

        # Record the cyclic short-arc distance min(|i-j|, N-|i-j|) for
        # each accepted move (same convention as the other variants).
        if record is not None:
            accepted = best_gain > 0
            if accepted.any():
                d = (best_i - best_j).abs()
                d = torch.minimum(d, N - d)
                record.extend(d[accepted].tolist())

        # Phase: per-batch Python loop applying the segment reversal.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        new_tour = tour.clone()
        n_accepted = 0
        for b in range(B):
            if best_gain[b] <= 0:
                continue
            i = best_i[b].item()
            j = best_j[b].item()
            t_b = tour[b]
            new_tour[b] = torch.cat(
                [t_b[: i + 1], torch.flip(t_b[i + 1 : j + 1], dims=[0]), t_b[j + 1 :]],
                dim=0,
            )
            n_accepted += 1
        _cuda_sync_if_available()
        if debug:
            gain_times["apply_loop"] = (time.perf_counter() - t0) * 1000

        # Phase: final seeds reorder via gather(1, ...).
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        seeds = seeds.gather(1, new_tour.unsqueeze(-1).expand(B, N, D))
        _cuda_sync_if_available()
        if debug:
            gain_times["reorder"] = (time.perf_counter() - t0) * 1000

        if debug:
            _print_debug_timings(
                "decomp_2opt",
                sweep_idx,
                gain_times,
                times,
                sweep_t0,
                n_accepted,
                n_edges=n_edges,
            )

    return seeds


def decomp_sampling_2opt_gain_matrix(
    seeds, base, revision_len, r_seam, r_candidate, times=None
):
    # Evaluate 2-opt gains for the *seam-initiated, far-distance* candidate
    # set used by ``decomp_sampling_2opt``. This is the composition of
    # ``decomp_2opt_gain_matrix`` (seam-neighbourhood source nodes) and
    # ``sampling_radius_2opt_gain_matrix`` (far-distance destinations via
    # signed, shifted offsets).
    #
    # Source nodes ``I`` are the unique seam-neighbourhood positions
    # (every tour position within ``r_seam`` hops of any decomposition
    # boundary, deduplicated). Each source ``i`` jumps to destinations
    # ``(i + s) % N`` for ``s`` in ``[base+shift, base+shift+r_candidate+1)``
    # (inclusive upper bound, matching ``sampling_radius_2opt_gain_matrix``)
    # and the symmetric negatives, where ``shift`` is re-sampled uniformly
    # per call. The signed-offset set covers the long-arc half of the tour
    # (``|s| >= N//2`` would be overlapping on the short arc), so the
    # ``shift_max = max(0, N//2 - base - r_candidate)`` clamp keeps the
    # window non-degenerate.
    #
    # Targets seam-induced long-range crossings that the GLOP revisor
    # cannot fix (the revisor optimizes within a window of size
    # ``revision_len``; anything outside that window, especially across
    # seams, is unreachable). The combination restricts swap sources to
    # the seam neighbourhood — where revisor-induced crossings are most
    # likely — while still reaching far enough destinations to fix
    # long-range seam crossings.
    #
    # Returns ``(tour, edge_gain, (src_b, src_i, dst_j))`` — the same
    # contract used by every other sparse variant so the driver can
    # share the per-instance ``scatter_max`` aggregation and per-batch
    # reversal loop. ``times`` (when not ``None``) is populated with the
    # ``radius`` (Phase 1 + Phase 2), ``gather``, ``distance``, and
    # ``mask`` keys.
    if revision_len is None:
        raise ValueError(
            "decomp_sampling_2opt_gain_matrix: revision_len must be provided "
            "(seams are defined per revisor layer)."
        )
    if base < 2:
        raise ValueError(
            f"decomp_sampling_2opt_gain_matrix: base={base} must be >= 2 "
            "(offsets 0 and ±1 are invalid 2-opt moves)."
        )
    if r_seam < 2:
        raise ValueError(
            f"decomp_sampling_2opt_gain_matrix: r_seam={r_seam} must be >= 2 "
            "(offsets 0 and ±1 are invalid 2-opt moves)."
        )
    if r_candidate < 1:
        raise ValueError(
            f"decomp_sampling_2opt_gain_matrix: r_candidate={r_candidate} "
            "must be >= 1 (otherwise the offset set is empty)."
        )

    device = seeds.device
    B, N, D = seeds.shape

    coords = seeds  # (B, N, D)
    tour = torch.arange(N, device=device).expand(B, N).clone()  # (B, N), index
    t = tour
    t_next = torch.roll(t, shifts=-1, dims=1)  # successor in closed tour

    # --- Phase 1: seam-neighbourhood source nodes --------------------------
    # Boundary NODE positions: {0, RL, 2*RL, ..., (num_chunks-1)*RL}, plus
    # the trailing segment start ``N - offset`` when ``offset != 0``.
    # Same construction as ``decomp_2opt_gain_matrix``.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    offset = N % revision_len
    starts = torch.arange(0, N - offset + 1, revision_len, device=device)
    delta = torch.arange(-r_seam, r_seam + 1, device=device)  # (2r_seam+1,)
    locs = ((starts[:, None] + delta[None, :]) % N).reshape(-1)
    locs = torch.unique(locs)
    L = locs.numel()

    # --- Phase 2: far-distance destinations (sampled per call) ------------
    # Re-sampled sliding window of signed offsets so successive sweeps
    # cover a broader effective offset range. Clamp to ``N//2 - base - r_candidate``
    # so the window stays in the long-arc half (the ±symmetry would
    # otherwise double-count short arcs).
    shift_max = max(
        0, N // 2 - base - r_candidate
    )  # OG: shift_max = max(0, N // 2 - base - r_candidate)
    if shift_max > 0:
        shift = int(torch.randint(0, shift_max, size=()).item())
    else:
        shift = 0
    offsets = torch.arange(
        base + shift, base + shift + r_candidate + 1, device=device
    )  # (r_candidate+1,)
    offsets = torch.cat([offsets, -offsets], dim=0)  # (2*(r_candidate+1),)

    # --- Empty-candidate guard -------------------------------------------
    # If the seam set is too small OR the offset set is empty, return
    # empty gain so the driver short-circuits.
    if L < 2 or offsets.numel() == 0:
        if times is not None:
            times["radius"] = (
                times.get("radius", 0.0) + (time.perf_counter() - t0) * 1000
            )
        empty = torch.empty(0, dtype=torch.long, device=device)
        return (
            tour,
            torch.empty(0, device=device),
            (empty, empty, empty),
        )

    # --- Candidate EDGES: every seam node × every far offset --------------
    # Note: ``I`` is the *seam* set, NOT ``arange(N)`` — that was a bug
    # in the original stub that effectively reduced this strategy to a
    # global sampling-radius search over all N nodes. With seam-restricted
    # sources, the candidate set is the directed cross-product
    # ``{i} × {(i + s) % N : s in offsets}``.
    I = locs  # (L,)
    J = (I.unsqueeze(1) + offsets) % N  # (L, 2*(r_candidate+1))
    I_tiled = I.unsqueeze(1).expand_as(J)  # (L, 2*(r_candidate+1))

    src_i = I_tiled.reshape(-1).repeat(B)  # (E,)
    dst_j = J.reshape(-1).repeat(B)  # (E,)
    E_per_b = L * offsets.numel()
    src_b = torch.arange(B, device=device).repeat_interleave(E_per_b)

    _cuda_sync_if_available()
    if times is not None:
        times["radius"] = times.get("radius", 0.0) + (time.perf_counter() - t0) * 1000

    # --- Phase: gather (E, 4, D) endpoint coords --------------------------
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    ti = t[src_b, src_i]
    tip1 = t_next[src_b, src_i]
    tj = t[src_b, dst_j]
    tjp1 = t_next[src_b, dst_j]
    node_idx = torch.stack([ti, tip1, tj, tjp1], dim=1)  # (E, 4)
    pts = coords[src_b[:, None], node_idx]  # (E, 4, D)
    _cuda_sync_if_available()
    if times is not None:
        times["gather"] = times.get("gather", 0.0) + (time.perf_counter() - t0) * 1000

    # --- Phase: per-edge gain computation --------------------------------
    _cuda_sync_if_available()
    t0 = time.perf_counter()

    def _len(a, b):
        return (pts[:, a] - pts[:, b]).pow(2).sum(dim=-1).sqrt()

    old1 = _len(0, 1)  # |t_i -- t_{i+1}|
    old2 = _len(2, 3)  # |t_j -- t_{j+1}|
    new1 = _len(0, 2)  # |t_i -- t_j|
    new2 = _len(1, 3)  # |t_{i+1} -- t_{j+1}|
    edge_gain = old1 + old2 - new1 - new2  # (E,)
    _cuda_sync_if_available()
    if times is not None:
        times["distance"] = (
            times.get("distance", 0.0) + (time.perf_counter() - t0) * 1000
        )

    # --- Phase: adjacency / wrap-around mask -----------------------------
    # The signed offsets mean ``dst_j`` may be ``<= src_i`` even when
    # ``dst_j != src_i + 1 mod N``; this mask is the same as the one in
    # ``sampling_radius_2opt_gain_matrix`` (no ``dst_j <= src_i`` defensive
    # redundancy, since the ±offsets make that test redundant with the
    # first disjunct only when ``dst_j > src_i``).
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    invalid_e = (dst_j <= src_i + 1) | ((src_i == 0) & (dst_j == N - 1))
    edge_gain = edge_gain.masked_fill(invalid_e, -1e9)
    _cuda_sync_if_available()
    if times is not None:
        times["mask"] = times.get("mask", 0.0) + (time.perf_counter() - t0) * 1000

    return tour, edge_gain, (src_b, src_i, dst_j)


def decomp_sampling_2opt(
    seeds,
    iters=10,
    base=None,
    revision_len=None,
    r_seam=None,
    r_candidate=None,
    debug=False,
    record=None,
):
    # Best-improvement 2-opt with the *seam-initiated, far-distance*
    # candidate set built in ``decomp_sampling_2opt_gain_matrix``. Mirrors
    # ``decomp_2opt`` (and ``sampling_radius_2opt``) line-for-line; the
    # only algorithmic difference is the candidate-set construction.
    #
    # ``revision_len`` is the *current* revisor layer's window length and
    # defines where the seams land in the reassembled tour frame. It is
    # mandatory — seams have no meaning outside a revisor layer.
    #
    # Defaults (applied here AND in the dispatcher; the dispatcher wins
    # so CLI args are honoured):
    #   base        = 2
    #   r_seam      = max(2, revision_len // 10)
    #   r_candidate = max(2, N // 10)
    device = seeds.device
    B, N, D = seeds.shape

    if revision_len is None:
        raise ValueError(
            "decomp_sampling_2opt: revision_len must be provided "
            "(seams are defined per revisor layer)."
        )

    if base is None:
        base = 2
    if r_seam is None:
        r_seam = max(2, revision_len // 10)
    if r_candidate is None:
        r_candidate = max(2, N // 10)

    seeds = seeds.clone()

    times = {} if debug else None

    sweep_idx = 0
    while sweep_idx < iters:
        sweep_idx += 1
        gain_times = {} if debug else None
        _cuda_sync_if_available()
        sweep_t0 = time.perf_counter()

        tour, edge_gain, idx_tuple = decomp_sampling_2opt_gain_matrix(
            seeds=seeds,
            base=base,
            revision_len=revision_len,
            r_seam=r_seam,
            r_candidate=r_candidate,
            times=gain_times,
        )
        if idx_tuple is None or edge_gain.numel() == 0:
            if debug:
                _print_debug_timings(
                    "decomp_sampling_2opt",
                    sweep_idx,
                    gain_times,
                    times,
                    sweep_t0,
                    0,
                )
            break

        src_b, src_i, dst_j = idx_tuple
        n_edges = edge_gain.numel()

        # Phase: scatter_max + best_i / best_j extraction.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        best_gain, argmax = scatter_max(src=edge_gain, index=src_b)
        best_i = src_i[argmax]
        best_j = dst_j[argmax]
        _cuda_sync_if_available()
        if debug:
            gain_times["scatter_max"] = (time.perf_counter() - t0) * 1000

        # Early exit with no 2-opt gains.
        if (best_gain <= 0).all():
            if debug:
                _print_debug_timings(
                    "decomp_sampling_2opt",
                    sweep_idx,
                    gain_times,
                    times,
                    sweep_t0,
                    0,
                    n_edges=n_edges,
                )
            break

        # Record the cyclic short-arc distance min(|i-j|, N-|i-j|) for
        # each accepted move (same convention as the other variants).
        if record is not None:
            accepted = best_gain > 0
            if accepted.any():
                d = (best_i - best_j).abs()
                d = torch.minimum(d, N - d)
                record.extend(d[accepted].tolist())

        # Phase: per-batch Python loop applying the segment reversal.
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        new_tour = tour.clone()
        n_accepted = 0
        for b in range(B):
            if best_gain[b] <= 0:
                continue
            i = best_i[b].item()
            j = best_j[b].item()
            t_b = tour[b]
            new_tour[b] = torch.cat(
                [t_b[: i + 1], torch.flip(t_b[i + 1 : j + 1], dims=[0]), t_b[j + 1 :]],
                dim=0,
            )
            n_accepted += 1
        _cuda_sync_if_available()
        if debug:
            gain_times["apply_loop"] = (time.perf_counter() - t0) * 1000

        # Phase: final seeds reorder via gather(1, ...).
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        seeds = seeds.gather(1, new_tour.unsqueeze(-1).expand(B, N, D))
        _cuda_sync_if_available()
        if debug:
            gain_times["reorder"] = (time.perf_counter() - t0) * 1000

        if debug:
            _print_debug_timings(
                "decomp_sampling_2opt",
                sweep_idx,
                gain_times,
                times,
                sweep_t0,
                n_accepted,
                n_edges=n_edges,
            )

    return seeds


def maybe_two_opt(seeds, opts, revision_len=None):
    """Optionally run 2-opt on a (B, N, D) coordinate tour, gated by opts.

    Reads flags via ``getattr`` with safe defaults so entry-point scripts that
    do not define the 2-opt options keep working (with 2-opt disabled).

    Dispatch keys (all read via ``getattr`` with defaults):

    * ``opts.use_2opt`` (bool, default ``False``) — master switch.
    * ``opts.two_opt_kind`` (str, default ``"full"``) — ``"full"``, ``"knn"``,
      ``"radius"``, ``"range_radius"``, ``"sampling_radius"``, or
      ``"decomp"``; unknown values fall back to ``"full"`` with a warning.
    * ``opts.two_opt_iters`` (int, default ``10``) — sweeps per invocation.
    * ``opts.two_opt_knn_k`` (int, default ``20``) — k for KNN-sparse 2-opt
      (ignored unless ``two_opt_kind == "knn"``).
    * ``opts.two_opt_radius`` (int or ``None``, default ``None``) — r for
      radius-sparse 2-opt. When ``None`` (the CLI default), the dispatcher
      falls back to ``max(2, N // 10)`` so the radius scales with the
      instance size (ignored unless ``two_opt_kind == "radius"``).
    * ``opts.two_opt_radius_min`` (int or ``None``, default ``None``) and
      ``opts.two_opt_radius_max`` (int or ``None``, default ``None``) — bounds
      of the offset window for range-radius 2-opt. Defaults are ``2`` and
      ``max(2, N // 10)`` respectively. If the user supplies ``r_min > r_max``
      the dispatcher silently swaps and emits a ``UserWarning`` (ignored
      unless ``two_opt_kind == "range_radius"``).
    * ``opts.two_opt_sampling_base`` (int or ``None``, default ``None``) and
      ``opts.two_opt_sampling_r`` (int or ``None``, default ``None``) — the
      base offset and window size for sampling-radius 2-opt. Defaults are
      ``2`` and ``max(2, N // 10)`` respectively. The sampled window is
      ``[base + shift, base + shift + r]`` where ``shift`` is re-sampled
      each sweep, so successive sweeps cover a broader effective offset
      range than a single static window. If the user supplies
      ``base > r`` the dispatcher silently swaps and emits a
      ``UserWarning`` (ignored unless ``two_opt_kind == "sampling_radius"``).
    * ``opts.two_opt_decomp_radius`` (int or ``None``, default ``None``) —
      half-width of the seam neighbourhood along the tour for the
      decomposition-aware 2-opt. When ``None`` the dispatcher falls back to
      ``max(2, revision_len // 10)`` (only consulted when ``two_opt_kind ==
      "decomp"``).
    * ``opts.two_opt_decomp_sampling_base`` (int or ``None``, default
      ``None``) — starting offset for the far-distance window in the
      decomposition+sampling composition. When ``None`` the dispatcher
      falls back to ``2`` (only consulted when ``two_opt_kind ==
      "decomp_sampling"``).
    * ``opts.two_opt_decomp_sampling_seam_radius`` (int or ``None``, default
      ``None``) — half-width of the seam neighbourhood along the tour for
      the decomposition+sampling composition. When ``None`` the dispatcher
      falls back to ``max(2, revision_len // 10)`` (only consulted when
      ``two_opt_kind == "decomp_sampling"``).
    * ``opts.two_opt_decomp_sampling_candidate_r`` (int or ``None``, default
      ``None``) — far-distance window size for the decomposition+sampling
      composition. The actual destination set is
      ``[base + shift, base + shift + r_candidate + 1)`` and its negation
      (matching ``sampling_radius_2opt_gain_matrix``). When ``None`` the
      dispatcher falls back to ``max(2, N // 10)`` (only consulted when
      ``two_opt_kind == "decomp_sampling"``).
    * ``opts.two_opt_debug`` (bool, default ``False``) — print per-sweep phase
      timings to stdout for performance investigation.
    * ``opts.record_two_opt_swaps`` (bool, default ``False``) — when True,
      the chosen 2-opt routine records ``|i_star - j_star|`` for every
      accepted move into ``opts.two_opt_swap_sink`` (a list). Used by
      ``eval_2opt.py`` to plot a swap-distance histogram.

    The ``revision_len`` kwarg is forwarded by the per-iter call site
    (``utils/functions.py:LCP_TSP``) so the ``"decomp"`` kind knows which
    revisor layer the call belongs to. It is unused by all other kinds and
    defaults to ``None`` for backward compatibility with the final-mode call
    site and library callers.
    """
    if not getattr(opts, "use_2opt", False):
        return seeds

    iters = int(getattr(opts, "two_opt_iters", 10))
    kind = getattr(opts, "two_opt_kind", "full")
    debug = bool(getattr(opts, "two_opt_debug", False))
    record = (
        getattr(opts, "two_opt_swap_sink", None)
        if getattr(opts, "record_two_opt_swaps", False)
        else None
    )

    if kind == "knn":
        k = int(getattr(opts, "two_opt_knn_k", 20))
        return knn_2opt(seeds, iters=iters, k=k, debug=debug, record=record)

    if kind == "radius":
        r = getattr(opts, "two_opt_radius", None)
        if r is None:
            # Default: 10% of the instance size (floored at 2 so the offset
            # range is non-empty).
            N = seeds.shape[1]
            r = max(2, N // 10)
        return radius_2opt(seeds, iters=iters, r=int(r), debug=debug, record=record)

    if kind == "range_radius":
        r_min = getattr(opts, "two_opt_radius_min", None)
        r_max = getattr(opts, "two_opt_radius_max", None)
        N = seeds.shape[1]
        # Default policy: r_min=2 (offsets 0/±1 are invalid 2-opt moves),
        # r_max ~ 10% of N. Either may be overridden by CLI flags.
        if r_min is None:
            r_min = 2
        if r_max is None:
            r_max = max(2, N // 10)
        if r_min > r_max:
            warnings.warn(
                f"two_opt: r_min={r_min} > r_max={r_max}; swapping",
                stacklevel=2,
            )
            r_min, r_max = r_max, r_min
        return range_radius_2opt(
            seeds,
            iters=iters,
            r_min=int(r_min),
            r_max=int(r_max),
            debug=debug,
            record=record,
        )

    if kind == "sampling_radius":
        # Shifting-window tour-position sparse variant: each call samples
        # a new offset shift, so successive sweeps cover a broader
        # effective offset range than a single static window.
        base = getattr(opts, "two_opt_sampling_base", None)
        r = getattr(opts, "two_opt_sampling_r", None)
        N = seeds.shape[1]
        if base is None:
            base = 2
        if r is None:
            r = max(2, N // 10)
        if base > r:
            warnings.warn(
                f"two_opt: sampling base={base} > r={r}; swapping",
                stacklevel=2,
            )
            base, r = r, base
        return sampling_radius_2opt(
            seeds,
            iters=iters,
            base=int(base),
            r=int(r),
            debug=debug,
            record=record,
        )

    if kind == "hop_radius":
        # Hop-stride tour-position sparse variant: each sweep picks
        # candidate offsets ``base, base+h, base+2h, ..., base+(k-1)*h``
        # (and the symmetric negatives), where
        # ``k = (N - base - 1) // 2 // h``. Regular stride, no shifting
        # window — successive sweeps use the same offset set, so the
        # benefit comes from sparse coverage of a broader offset range
        # at per-call cost ``O(B * N * k)``. No swap-and-warn when
        # ``base > h`` because ``h`` is a stride, not a width.
        base = getattr(opts, "two_opt_hop_base", None)
        h = getattr(opts, "two_opt_hop_h", None)
        N = seeds.shape[1]
        if base is None:
            base = 2
        if h is None:
            h = max(2, N // 10)
        return hop_radius_2opt(
            seeds,
            iters=iters,
            base=int(base),
            h=int(h),
            debug=debug,
            record=record,
        )

    if kind == "decomp":
        # Seams are only well-defined in the reassembled tour frame
        # produced by LCP_TSP, so the decomposition-aware variant is
        # restricted to per-iter mode. In final mode we emit a warning
        # and skip rather than silently producing a meaningless search.
        if getattr(opts, "two_opt_mode", "final") != "per_iter":
            warnings.warn(
                "two_opt_kind='decomp' requires --two_opt_mode=per_iter "
                "(seams are only well-defined per revisor iter); skipping.",
                stacklevel=2,
            )
            return seeds
        rl = revision_len
        if rl is None:
            # Library caller did not thread revision_len; fall back to the
            # last layer's window length when available. Emit a warning so
            # the silent fallback does not surprise users.
            rl_list = getattr(opts, "revision_lens", None)
            if rl_list:
                rl = int(rl_list[-1])
                warnings.warn(
                    f"two_opt_kind='decomp': revision_len not provided; "
                    f"using opts.revision_lens[-1]={rl} as a heuristic.",
                    stacklevel=2,
                )
        if rl is None:
            warnings.warn(
                "two_opt_kind='decomp': no revision_len available; skipping.",
                stacklevel=2,
            )
            return seeds
        r = getattr(opts, "two_opt_decomp_radius", None)
        return decomp_2opt(
            seeds,
            iters=iters,
            revision_len=int(rl),
            r=int(r) if r is not None else None,
            debug=debug,
            record=record,
        )

    if kind == "decomp_sampling":
        # Seam-initiated × far-distance composition. Like ``decomp``, this
        # variant needs a per-revisor-iter ``revision_len`` to know where
        # the seams land, so it is restricted to ``per_iter`` mode. In
        # ``final`` mode we emit a warning and skip rather than silently
        # producing a meaningless search.
        if getattr(opts, "two_opt_mode", "final") != "per_iter":
            warnings.warn(
                "two_opt_kind='decomp_sampling' requires "
                "--two_opt_mode=per_iter (seams are only well-defined per "
                "revisor iter); skipping.",
                stacklevel=2,
            )
            return seeds
        rl = revision_len
        if rl is None:
            # Library caller did not thread revision_len; fall back to the
            # last layer's window length when available. Emit a warning so
            # the silent fallback does not surprise users.
            rl_list = getattr(opts, "revision_lens", None)
            if rl_list:
                rl = int(rl_list[-1])
                warnings.warn(
                    f"two_opt_kind='decomp_sampling': revision_len not "
                    f"provided; using opts.revision_lens[-1]={rl} as a "
                    f"heuristic.",
                    stacklevel=2,
                )
        if rl is None:
            warnings.warn(
                "two_opt_kind='decomp_sampling': no revision_len "
                "available; skipping.",
                stacklevel=2,
            )
            return seeds
        base = getattr(opts, "two_opt_decomp_sampling_base", None)
        r_seam = getattr(opts, "two_opt_decomp_sampling_seam_radius", None)
        r_cand = getattr(opts, "two_opt_decomp_sampling_candidate_r", None)
        return decomp_sampling_2opt(
            seeds,
            iters=iters,
            base=int(base) if base is not None else None,
            revision_len=int(rl),
            r_seam=int(r_seam) if r_seam is not None else None,
            r_candidate=int(r_cand) if r_cand is not None else None,
            debug=debug,
            record=record,
        )

    if kind != "full":
        warnings.warn(
            f"unknown two_opt_kind={kind!r}; falling back to 'full'",
            stacklevel=2,
        )
    return full_2opt(seeds, iters=iters, debug=debug, record=record)
