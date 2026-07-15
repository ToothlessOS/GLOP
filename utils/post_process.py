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

Entry points
    * ``full_2opt(seeds, iters=10)`` — improve a batch of tours with a dense
      candidate set; returns coords. Targets small-to-moderate ``N``.
    * ``knn_2opt(seeds, iters=10, k=20)`` — k-nearest-neighbour-sparse variant.
      Each sweep considers only the KNN edges of the current coordinates, so
      both the candidate set *and* the per-edge distance computation scale as
      ``O(B * N * k)`` instead of ``O(B * N^2)``. Suitable for larger ``N``
      where the dense gain matrix is too costly. Edge gains are aggregated
      per-instance via ``scatter_max``.
    * ``maybe_two_opt(seeds, opts)`` — opts-gated wrapper used by the pipeline.
      Dispatches on ``opts.two_opt_kind`` (``"full"`` or ``"knn"``); unknown
      values fall back to ``"full"`` with a ``UserWarning``.

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
    * ``--two_opt_kind {full, knn}`` — which algorithm to dispatch.
    * ``--two_opt_mode {final, per_iter}`` — run once after the whole pipeline
      (``"final"``) or after every revisor iteration (``"per_iter"``).
    * ``--two_opt_iters`` — sweeps per invocation (default 10).
    * ``--two_opt_knn_k`` — ``k`` for ``"knn"`` (default 20, ignored otherwise).
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


def _print_debug_timings(algo, sweep_idx, sweep_times, totals,
                         sweep_t0, n_accepted, n_edges=0):
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
    parts = ' '.join(f'{k}={v:.1f}' for k, v in sweep_times.items())
    edges_str = f' edges={n_edges}' if n_edges else ''
    acc_str = f' accepted={n_accepted}' if n_accepted else ''
    print(f'[{algo}] sweep {sweep_idx:<3} total={total_ms:7.1f}ms'
          f'{edges_str}{acc_str} | {parts}', flush=True)


# Phase names printed by the debug helpers above. The driver functions
# iterate over this list when emitting totals, so keep it in sync with the
# keys populated in the gain-matrix / driver code.
_DEBUG_PHASE_KEYS = (
    'knn', 'filter', 'gather', 'distance', 'mask',
    'scatter_max', 'apply_loop', 'reorder',
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


def full_2opt(seeds, iters=10, debug=False):
    # Run full 2opt.
    # seeds: (B, N, D) coordinates in tour order (the tour is implicit in row order).
    # Returns: improved (B, N, D) coordinates, still in tour order.
    # ``debug`` enables per-sweep phase-timing output to stdout, used by the
    # ``--two_opt_debug`` CLI flag for perf investigation.
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
            sweep_times['gain_matrix'] = (time.perf_counter() - gain_t0) * 1000

        # gain => (B, N, N); pick the single best flip per instance.
        best_gain, flat_idx = gain.view(B, -1).max(dim=1)  # => (B, N^2) => (B,)
        if (best_gain <= 0).all():
            if debug:
                _print_debug_timings('full_2opt', sweep_idx, sweep_times, times,
                                     sweep_t0, 0)
            break

        # Convert flat_idx back to (i, j)
        i_star = (flat_idx // N).tolist()
        j_star = (flat_idx % N).tolist()

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
            sweep_times['apply_loop'] = (time.perf_counter() - apply_t0) * 1000

        # Phase: final seeds reorder via gather(1, ...).
        reorder_t0 = time.perf_counter()
        seeds = seeds.gather(1, new_tour.unsqueeze(-1).expand(B, N, D))
        _cuda_sync_if_available()
        if debug:
            sweep_times['reorder'] = (time.perf_counter() - reorder_t0) * 1000

        if debug:
            _print_debug_timings('full_2opt', sweep_idx, sweep_times, times,
                                 sweep_t0, n_accepted)

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
        times['knn'] = times.get('knn', 0.0) + (time.perf_counter() - t0) * 1000

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
        times['filter'] = times.get('filter', 0.0) + (time.perf_counter() - t0) * 1000

    if E == 0:
        gain = torch.empty(B, 0, device=device)
        return tour, gain, None

    # Compute 2-opt gains for these (b, i, j).
    # Old edges: (t_i, t_{i+1}) and (t_j, t_{j+1})
    ti = t[src_b, src_i]    # advanced indexing per edge
    tip1 = t_next[src_b, src_i]
    tj = t[src_b, dst_j]
    tjp1 = t_next[src_b, dst_j]

    # Phase: gather (E, 4, D) endpoint coords.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    node_idx = torch.stack([ti, tip1, tj, tjp1], dim=1)              # (E, 4)
    pts = coords[src_b[:, None], node_idx]                            # (E, 4, D)
    _cuda_sync_if_available()
    if times is not None:
        times['gather'] = times.get('gather', 0.0) + (time.perf_counter() - t0) * 1000

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
        times['distance'] = times.get('distance', 0.0) + (time.perf_counter() - t0) * 1000

    # Phase: adjacency / wrap-around mask.
    _cuda_sync_if_available()
    t0 = time.perf_counter()
    invalid_e = (dst_j <= src_i + 1) | ((src_i == 0) & (dst_j == N - 1))  # (E,)
    edge_gain = edge_gain.masked_fill(invalid_e, -1e9)
    _cuda_sync_if_available()
    if times is not None:
        times['mask'] = times.get('mask', 0.0) + (time.perf_counter() - t0) * 1000

    # Return per-edge gains plus the corresponding (b, i, j) for the driver.
    return tour, edge_gain, (src_b, src_i, dst_j)


def knn_2opt(seeds, iters=10, k=20, debug=False):
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

        tour, edge_gain, idx_tuple = knn_2opt_gain_matrix(seeds=seeds, k=k,
                                                          times=gain_times)
        if idx_tuple is None or edge_gain.numel() == 0:
            if debug:
                _print_debug_timings('knn_2opt', sweep_idx, gain_times, times,
                                     sweep_t0, 0)
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
            gain_times['scatter_max'] = (time.perf_counter() - t0) * 1000

        # Early exit with no 2-opt gains
        if (best_gain <= 0).all():
            if debug:
                _print_debug_timings('knn_2opt', sweep_idx, gain_times, times,
                                     sweep_t0, 0, n_edges=n_edges)
            break

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
            gain_times['apply_loop'] = (time.perf_counter() - t0) * 1000

        # Phase: final seeds reorder via gather(1, ...).
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        seeds = seeds.gather(1, new_tour.unsqueeze(-1).expand(B, N, D))
        _cuda_sync_if_available()
        if debug:
            gain_times['reorder'] = (time.perf_counter() - t0) * 1000

        if debug:
            _print_debug_timings('knn_2opt', sweep_idx, gain_times, times,
                                 sweep_t0, n_accepted, n_edges=n_edges)

    return seeds


def maybe_two_opt(seeds, opts):
    """Optionally run 2-opt on a (B, N, D) coordinate tour, gated by opts.

    Reads flags via ``getattr`` with safe defaults so entry-point scripts that
    do not define the 2-opt options keep working (with 2-opt disabled).

    Dispatch keys (all read via ``getattr`` with defaults):

    * ``opts.use_2opt`` (bool, default ``False``) — master switch.
    * ``opts.two_opt_kind`` (str, default ``"full"``) — ``"full"`` or ``"knn"``;
      unknown values fall back to ``"full"`` with a warning.
    * ``opts.two_opt_iters`` (int, default ``10``) — sweeps per invocation.
    * ``opts.two_opt_knn_k`` (int, default ``20``) — k for KNN-sparse 2-opt
      (ignored when ``two_opt_kind == "full"``).
    * ``opts.two_opt_debug`` (bool, default ``False``) — print per-sweep phase
      timings to stdout for performance investigation.
    """
    if not getattr(opts, "use_2opt", False):
        return seeds

    iters = int(getattr(opts, "two_opt_iters", 10))
    kind = getattr(opts, "two_opt_kind", "full")
    debug = bool(getattr(opts, "two_opt_debug", False))

    if kind == "knn":
        k = int(getattr(opts, "two_opt_knn_k", 20))
        return knn_2opt(seeds, iters=iters, k=k, debug=debug)

    if kind != "full":
        warnings.warn(
            f"unknown two_opt_kind={kind!r}; falling back to 'full'",
            stacklevel=2,
        )
    return full_2opt(seeds, iters=iters, debug=debug)
