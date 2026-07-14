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
    ``full_2opt_gain_matrix`` materializes ``O(B * N^2)`` memory for the gain
    matrix, so this implementation targets small-to-moderate ``N`` (e.g. a few
    hundred). For very large instances, candidate-edge selection / sparsity would
    be required (see the TODO in ``full_2opt_gain_matrix``).

Entry points
    * ``full_2opt(seeds, iters=10)`` — improve a batch of tours, returns coords.
    * ``maybe_two_opt(seeds, opts)`` — opts-gated wrapper used by the pipeline.

The pipeline wiring and CLI flags live in ``utils/functions.py`` (``reconnect``
and ``LCP_TSP``) and ``main.py`` respectively; see the README section
"2-opt post-processing".
"""

import torch


def full_2opt_gain_matrix(seeds):
    # Evaluate the gain of all possible 2opt flips on a TSP tour
    # Euclidean TSP only!!!
    # TODO: This works for smaller instances; Sparsity / selection of candidate edges req'd for larger instances
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


def full_2opt(seeds, iters=10):
    # Run full 2opt.
    # seeds: (B, N, D) coordinates in tour order (the tour is implicit in row order).
    # Returns: improved (B, N, D) coordinates, still in tour order.
    B, N, D = seeds.shape

    # Work on a copy so we never mutate the caller's tensor.
    seeds = seeds.clone()

    for _ in range(iters):
        # Get gain metrics for the current tour (gain_matrix assumes an
        # identity tour over the *current* row order of seeds).
        dist, tour, gain, mask = full_2opt_gain_matrix(seeds)

        # gain => (B, N, N); pick the single best flip per instance.
        best_gain, flat_idx = gain.view(B, -1).max(dim=1)  # => (B, N^2) => (B,)
        if (best_gain <= 0).all():
            break

        # Convert flat_idx back to (i, j)
        i_star = (flat_idx // N).tolist()
        j_star = (flat_idx % N).tolist()

        # Build the new permutation (tour is arange(N) per row) by reversing the
        # segment (i+1 .. j] for every improving instance.
        new_tour = tour.clone()
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

        # Reorder the coordinates by the new permutation so seeds stays a
        # coordinate tensor in tour order for the next sweep / the return value.
        seeds = seeds.gather(1, new_tour.unsqueeze(-1).expand(B, N, D))

    return seeds


def maybe_two_opt(seeds, opts):
    """Optionally run 2-opt on a (B, N, D) coordinate tour, gated by opts.

    Reads flags via getattr with safe defaults so entry-point scripts that do
    not define the 2-opt options keep working (with 2-opt disabled).
    """
    if not getattr(opts, "use_2opt", False):
        return seeds
    return full_2opt(seeds, iters=getattr(opts, "two_opt_iters", 10))
