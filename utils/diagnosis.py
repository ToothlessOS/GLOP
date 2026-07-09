# Check if heuristics for optimal TSP tour are valid. Indicating whether decompose-on-edge is req'd

import os

import numpy as np
import torch
from scipy.spatial import ConvexHull

import matplotlib

matplotlib.use("Agg")  # headless backend; safe even when no DISPLAY is present
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def check_no_self_intersection(seeds):
    """
    Takes a batch of TSP tours and checks if any of the tours have self-intersections.
    Args:
        seeds: Tensor of shape (B, N, 2) representing B TSP tours
    Returns:
        has_intersection: Tensor of shape (B,) indicating if each tour has self-intersections
        num_intersections: Tensor of shape (B,) indicating the number of self-intersections in each tour
        fraction_intersections: Tensor of shape (B,) indicating the fraction of valid edge pairs that intersect in each tour
        num_valid_pairs: Tensor of shape (B,) indicating the number of valid edge pairs considered for intersection checking in each tour
        inter; the raw intersection results for each pair of edges in each tour, shape (B, N, N)
    Note:
        Peak memory is O(B * N^2) due to broadcasting (B, N, 1, 2) - (B, 1, N, 2)
        into (B, N, N, 2) intermediates. For very large N, prefer the
        precomputed-pair-index variant.
    """
    B, N, D = seeds.shape  # Batchsize, Number of nodes, Coords

    # Trivial cases: tour has at most two adjacent edges, no possible intersection.
    if N < 3:
        z_bool = torch.zeros(B, dtype=torch.bool, device=seeds.device)
        z_f = torch.zeros(B, dtype=torch.float32, device=seeds.device)
        return z_bool, z_f, z_f, z_f, z_f

    # Build edges
    p = seeds  # p[B, k, 2]
    p_next = torch.roll(
        p, shifts=-1, dims=1
    )  # p[B, k+1, 2], this is the next node in the tour

    # Build all edges pairs (i, j). i indexes the row (i -> i+1 edge),
    # j indexes the column (j -> j+1 edge).
    idx = torch.arange(N, device=seeds.device)
    i = idx.view(1, N, 1).expand(B, N, N)  # [B, N, N]
    j = idx.view(1, 1, N).expand(B, N, N)  # [B, N, N]

    # Mask out same and adjacent edges (including wrap-around)
    adj = (i == j) | ((i + 1) % N == j) | (i == (j + 1) % N)
    upper = j > i  # Consider only upper triangle to avoid double counting
    valid = ~adj & upper  # [B, N, N]

    # Gather endpoints for all pairs
    # Shape: (B, N, 2) => (B, N, N, 2)
    P = p.unsqueeze(2)  # [B, N, 1, 2]
    Pn = p_next.unsqueeze(2)  # [B, N, 1, 2]
    Q = p.unsqueeze(1)  # [B, 1, N, 2]
    Qn = p_next.unsqueeze(1)  # [B, 1, N, 2]

    # Orientation helper
    def orient(A, B, C):
        BA = B - A
        CA = C - A
        return BA[..., 0] * CA[..., 1] - BA[..., 1] * CA[..., 0]  # Element-wise

    # Check for intersection
    o1 = orient(P, Pn, Q)  # (B, N, N)
    o2 = orient(P, Pn, Qn)  # (B, N, N)
    o3 = orient(Q, Qn, P)  # (B, N, N)
    o4 = orient(Q, Qn, Pn)  # (B, N, N)

    inter = (o1 * o2 < 0) & (o3 * o4 < 0) & valid  # (B, N, N), with valid mask applied

    # Metrics
    # `torch.any` does not accept a tuple of dims; flatten the (N, N) panel first.
    has_intersection = inter.flatten(1).any(dim=1)  # (B,)
    num_intersections = inter.sum(dim=(1, 2))  # (B,)
    num_valid_pairs = valid.sum(dim=(1, 2))  # (B,)
    fraction_intersections = num_intersections.float() / num_valid_pairs.float()  # (B,)

    return (
        has_intersection,
        num_intersections,
        fraction_intersections,
        num_valid_pairs,
        inter,  # Get raw results for analysis
    )


def fix_intersections_via_2opt(tours, max_iter=20):
    """Direct crossing-fix 2-opt: for every detected (i, j) crossing,
    reverse ``tours[..., i+1 : j+1]`` (the classical "swap the two edge
    sequences" move). Iterates until no crossings remain or ``max_iter``
    is reached. Mutates ``tours`` in place and also returns it, alongside
    a stats dict.

    Args:
        tours: (B, N, 2) tensor of node coordinates in tour order. The
            tour is treated as a closed loop. Modified in place.
        max_iter: hard cap on outer iterations to guarantee termination.

    Returns:
        tours: the same tensor with crossings removed (or reduced as far
            as ``max_iter`` allows).
        info: dict with keys
            ``iters_used`` (int),
            ``initial_intersections`` (int),
            ``final_intersections`` (int),
            ``initial_cost`` (B,) tensor of closed-loop costs before,
            ``final_cost`` (B,) tensor of closed-loop costs after.

    Note:
        The ``inter`` mask from :func:`check_no_self_intersection` is
        upper-triangular with adjacent edges zeroed out, so for any
        detected pair ``(i, j)`` we have ``i + 1 < j`` and the wrap-around
        edge pair is not detected (this is a known limitation of the
        detector, not of this fixer).

        Each per-tour fix re-detects intersections before picking the
        next pair. Without this, the ``inter`` mask from before the
        previous flip goes stale — flipping ``tours[b, i+1 : j+1]``
        shifts every index inside that range, so acting on the old
        ``inter`` mask afterward acts on the wrong segment and can
        introduce new crossings (this was the source of the
        ``B >= 2`` regression on TSP-200 where the function was
        observed to leave tours with more crossings than it started).
    """

    print("[POST] Fixing self-intersections via 2-opt...")

    initial_tours = tours.detach().clone()
    B, N, _ = tours.shape

    def _closed_loop_cost(t):
        return (t[:, 1:] - t[:, :-1]).norm(p=2, dim=2).sum(1) + (
            t[:, 0] - t[:, -1]
        ).norm(p=2, dim=1)

    if N < 3:
        return tours, {
            "iters_used": 0,
            "initial_intersections": 0,
            "final_intersections": 0,
            "initial_cost": _closed_loop_cost(tours),
            "final_cost": _closed_loop_cost(tours),
        }

    initial_cost = _closed_loop_cost(tours)
    iters_used = 0
    final_intersections_count = 0

    def _fix_one(b):
        """Re-detect crossings in tour ``b`` and apply a single 2-opt flip
        to the first non-adjacent detected pair. Returns True if a flip
        was applied. Stops cleanly when the tour has no fixable crossings.

        Re-detection is required between consecutive fixes because the
        previous flip reversed positions (i+1, ..., j) — any remaining
        crossings whose indices fall inside that range now refer to
        different nodes. Acting on the stale ``inter`` mask from before
        the flip introduces new crossings; that was the source of the
        "introduces more edge intersections" bug on ``B >= 2`` tours.
        """
        _, _, _, _, inter_b = check_no_self_intersection(tours[b : b + 1])
        if not inter_b.any():
            return False

        pairs = torch.nonzero(inter_b[0], as_tuple=False)
        if pairs.numel() == 0:
            return False

        i_all = pairs[:, 0]
        j_all = pairs[:, 1]
        # Defensive: skip any pair that is not strictly non-adjacent.
        # The detector's upper-triangular + adj-mask already guarantees
        # this, but a future refactor could relax the assumption.
        mask = j_all > i_all + 1
        i_all = i_all[mask]
        j_all = j_all[mask]
        if i_all.numel() == 0:
            return False

        i = int(i_all[0].item())
        j = int(j_all[0].item())
        tours[b, i + 1 : j + 1] = torch.flip(tours[b, i + 1 : j + 1], dims=[0])
        return True

    for it in range(max_iter):
        iters_used = it + 1
        has_inter, n_inter, _, _, _ = check_no_self_intersection(tours)

        if not has_inter.any():
            final_intersections_count = 0
            break

        any_change = False
        for b in range(B):
            if not has_inter[b]:
                continue

            # Converge this tour: re-detect after every fix. Each
            # successful flip strictly reduces this tour's crossing
            # count, so the inner loop terminates well before the cap
            # in the common case.
            inner_cap = max(2 * int(n_inter[b].item()), 100)
            for _ in range(inner_cap):
                if not _fix_one(b):
                    break
                any_change = True

        # Re-detect to update n_inter for the next iteration; also captures
        # the final intersection count when the loop exits via any_change=False.
        _, n_inter_after, _, _, _ = check_no_self_intersection(tours)
        final_intersections_count = int(n_inter_after.sum().item())

        if not any_change:
            break

    info = {
        "iters_used": iters_used,
        "initial_intersections": int(
            check_no_self_intersection(initial_tours)[1].sum().item()
        ),
        "final_intersections": final_intersections_count,
        "initial_cost": initial_cost,
        "final_cost": _closed_loop_cost(tours),
    }
    return tours, info


def check_convex_hull(seeds):
    B, N, D = seeds.shape  # Batchsize, Number of nodes, Coords
    if N < 3:
        return torch.ones(B, dtype=torch.float32, device=seeds.device)
    # Iterate over batch B
    consistency = torch.zeros(B, dtype=torch.float32, device=seeds.device)
    for i in range(B):
        points = seeds[i].cpu().numpy()  # Convert to numpy for ConvexHull

        # scipy.spatial.ConvexHull raises QhullError on degenerate input
        # (e.g. all-collinear or coincident points). Treat that as trivially
        # satisfying the convex-hull property (the hull is degenerate).
        try:
            hull_seq = torch.tensor(
                ConvexHull(points).vertices, dtype=torch.long, device=seeds.device
            )  # Indices of convex hull vertices in CCW order
        except Exception:
            consistency[i] = 1.0
            continue
        H = hull_seq.shape[0]  # Number of convex hull vertices

        # seeds[i, k] is the k-th point in the tour, so the tour position of
        # vertex k is just k. The hull vertices sorted by tour position is
        # therefore just the sorted hull indices.
        tour_positions = hull_seq  # (H,)
        _, order = torch.sort(tour_positions)
        tour_seq = hull_seq[order]  # hull vertices in tour order

        # Function to test cyclic equality allowing rotation and reversal
        def cyclic_equal(a, b):
            # a, b: (H,)
            if H == 0:
                return True
            # To allow rotation, we "double" a and look for b as a contiguous block
            a2 = torch.cat([a, a], dim=0)  # (2H,)
            # Try all rotations
            for shift in range(H):
                if torch.all(a2[shift : shift + H] == b):
                    return True
            # Try reversed b
            b_rev = torch.flip(b, dims=[0])
            for shift in range(H):
                if torch.all(a2[shift : shift + H] == b_rev):
                    return True
            return False

        consistency[i] = 1.0 if cyclic_equal(hull_seq, tour_seq) else 0.0

    return consistency, tour_seq  # (B,), 1 if convex hull property satisfied, else 0


def check_purity_order(seeds):
    """Compute the purity order K_p(e_{ij}) for every edge in a batch of
    TSP tours in traversal order.

    For each edge e_{ij} = (x_i, x_j) in the closed-loop tour, treat the
    edge as a diameter and count how many other cities lie inside the
    resulting circle::

        N_c(e_{ij}) = {x in X \\ {x_i, x_j} : (x_i - x)^T (x_j - x) < 0}
        K_p(e_{ij}) = |N_c(e_{ij})|

    An edge with K_p = 0 is "pure" — no other city lies inside its
    diameter circle — and is locally optimal in the diameter sense.

    Args:
        seeds: (B, N, 2) tensor of node coordinates in tour order. The
            tour is treated as a closed loop; the N-th edge wraps from
            position N-1 back to position 0.

    Returns:
        purity_order: (B, N) int64 per-edge tensor. ``purity_order[b, k]``
            is the purity of the k-th edge in batch instance b, i.e. the
            edge ``(seeds[b, k], seeds[b, (k + 1) % N])``. The second
            axis indexes edges (a closed tour of N nodes has exactly N
            edges), so each entry is a per-edge scalar. Values range
            over ``[0, N - 2]``.

    Note:
        The two endpoints of each edge are excluded by the strict ``<``
        in the dot-product test, so an edge whose diameter-circle
        contains no third city has K_p = 0. The maximum K_p = N - 2 is
        reached when every other city lies inside the edge's diameter
        circle. Complexity is O(B * N^2) memory and compute due to the
        all-pairs broadcast.
    """
    B, N, D = seeds.shape  # Batchsize, Number of nodes, Coords
    x_i = seeds.unsqueeze(2)  # (B, N, 1, 2)
    x_j = torch.roll(seeds, shifts=-1, dims=1).unsqueeze(
        2
    )  # (B, N, 1, 2), wrap-around handled
    x = seeds.unsqueeze(1)  # (B, 1, N, 2)

    # Make use of torch array boardcasting:
    # x_i[b, i, 0, :] is node x_i in tour order for batch b
    # x_j[b, i, 0, ;] is the next node x_j = x_{i+1} in tour order for batch b
    # x[b, 0, k, :] is the node x_k, candidate node to check whether in circle
    # Broadcasting:
    # Starting from the last dimension and moving left, dimensions must be either equal, or one of them must be 1.
    # If a dimension is 1, it is “stretched” (conceptually repeated) to match the other size.
    # Therefore:
    # diff_i[b, i, k, :] = x_i[b, i, 0, :] - x[b, 0, k, :] = x_i - x_k
    diff_i = x_i - x  # (B, N, N, 2)
    diff_j = x_j - x  # (B, N, N, 2)

    dot_product = (diff_i * diff_j).sum(-1)  # (B, N, N)
    purity_order = (dot_product < 0).int().sum(-1)  # (B, N), purity order for each edge

    return purity_order


def _summarize_purity_order(purity_order):
    """Compute the three scalar summaries of a (B, N) purity-order tensor.

    Aggregates a per-edge purity tensor over a batch of tours and returns
    three scalar metrics. All three are computed across the full
    ``B * N`` edges of the input.

    Args:
        purity_order: (B, N) tensor (typically the output of
            ``check_purity_order``). The second axis indexes the N
            per-edge scalars for each of the B tours. The tensor is
            detached and cast to float32 before aggregation, so it is
            safe to pass an autograd-connected tensor.

    Returns:
        dict with the following keys:

        - ``mean_purity_order`` (float): mean K_p over all ``B * N``
          edges.
        - ``fraction_pure`` (float): proportion of edges with
          ``K_p == 0`` (i.e. "pure" edges with no third city inside
          the diameter circle). In ``[0, 1]``.
        - ``mean_purity_order_nonpure`` (float or NaN): mean K_p
          restricted to edges with ``K_p > 0``. NaN when every edge is
          pure (``fraction_pure == 1.0``) or when the input is empty.

    Note:
        Edge cases: returns ``mean_purity_order = 0.0``,
        ``fraction_pure = 0.0``, and ``mean_purity_order_nonpure = NaN``
        when the input has zero elements.
    """
    po = purity_order.detach()
    total = po.numel()
    if total == 0:
        return {
            "mean_purity_order": 0.0,
            "fraction_pure": 0.0,
            "mean_purity_order_nonpure": float("nan"),
        }
    po_f = po.float()
    pure_mask = po == 0
    n_pure = int(pure_mask.sum().item())
    mean = float(po_f.mean().item())
    frac_pure = n_pure / total
    if n_pure < total:
        mean_nonpure = float(po_f[~pure_mask].mean().item())
    else:
        mean_nonpure = float("nan")
    return {
        "mean_purity_order": mean,
        "fraction_pure": frac_pure,
        "mean_purity_order_nonpure": mean_nonpure,
    }


def plot_tsp_tours(
    tours,
    has_intersection=None,
    num_intersections=None,
    intersection_results=None,
    consistency=None,
    out_dir="results",
    filename=None,
    max_plots=16,
    tag=None,
):
    """Render a PNG of every TSP tour that violates the no-self-intersection
    or convex-hull constraint. Edges involved in any crossing are drawn red;
    the convex hull is drawn as a dotted gray border; hull vertices are drawn
    as red filled circles when the consistency check fails. Returns the saved
    path, or None when every tour is clean (no file is written).

    Args:
        tours: (B, N, 2) tensor of node coordinates in tour order. The tour is
            treated as a closed loop: position N-1 is adjacent to position 0.
        has_intersection: optional (B,) bool tensor from
            :func:`check_no_self_intersection`. Recomputed if ``None``.
        num_intersections: optional (B,) int tensor; recomputed if ``None``.
        intersection_results: optional (B, N, N) bool tensor (upper-triangular,
            adjacent pairs zeroed out). Recomputed if ``None``.
        consistency: optional (B,) float tensor (1.0 = hull satisfied) from
            :func:`check_convex_hull`. Recomputed if ``None``.
        out_dir: directory to write the PNG into; created if missing.
        filename: explicit output filename; defaults to
            ``tour_diag_{tag}.png`` inside ``out_dir``.
        max_plots: cap on the number of subplots in the single figure, to
            keep the rendered PNG legible when many tours are bad.
        tag: short label used in the suptitle and default filename, e.g.
            ``"tsp200_w1"``. No filesystem sanitization is performed.
    """
    B, N, _ = tours.shape

    # Trivial tour: nothing meaningful to render.
    if N < 3:
        print("[plot_tsp_tours] N<3 — nothing to plot.")
        return None

    # Reuse the caller's diagnostics when available; otherwise recompute.
    if (
        has_intersection is None
        or num_intersections is None
        or intersection_results is None
    ):
        (
            has_intersection,
            num_intersections,
            _,
            _,
            intersection_results,
        ) = check_no_self_intersection(tours)
    if consistency is None:
        consistency, _ = check_convex_hull(tours)

    has_intersection = has_intersection.bool()
    hull_fail = consistency == 0
    problematic = has_intersection | hull_fail
    bad_idx = torch.nonzero(problematic, as_tuple=True)[0].tolist()

    if not bad_idx:
        print("[plot_tsp_tours] All tours clean — no PNG written.")
        return None

    # Per-edge intersection status. `inter` is upper-triangular with adjacent
    # edges masked, so the row of edge k holds crossings with edges j>k, and
    # the column holds crossings with edges i<k. An edge participates in any
    # crossing when it appears in either a row OR column entry.
    edge_intersects = intersection_results.any(dim=2) | intersection_results.any(dim=1)

    # Cap the number of subplots so the figure stays legible.
    total_bad = len(bad_idx)
    K = min(total_bad, max_plots)
    if K < total_bad:
        print(f"[plot_tsp_tours] {total_bad} failing tours; rendering first {K}.")

    # Grid layout: 1, 2, 3 -> 1 row; 4+ -> up to 4 columns, rows = ceil(K/cols).
    cols = min(4, K)
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.0 * cols, 4.0 * rows), squeeze=False
    )
    axes_flat = axes.flatten()

    for plot_pos, idx in enumerate(bad_idx[:K]):
        ax = axes_flat[plot_pos]
        tour_np = tours[idx].detach().cpu().numpy()
        # Close the loop by appending the first point at the end.
        tour_closed = np.concatenate([tour_np, tour_np[:1]], axis=0)
        edge_mask = edge_intersects[idx].cpu().numpy()

        # (a) Base tour: light blue closed loop.
        ax.plot(
            tour_closed[:, 0],
            tour_closed[:, 1],
            color="C0",
            linewidth=0.8,
            alpha=0.6,
            zorder=1,
        )

        # (b) Intersecting edges: red, drawn on top of the base loop.
        for k in range(N):
            if not edge_mask[k]:
                continue
            p1 = tour_np[k]
            p2 = tour_np[(k + 1) % N]
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="red",
                linewidth=1.6,
                alpha=0.9,
                zorder=3,
            )

        # (c) Node markers (small black dots at every tour vertex). Slightly
        # smaller than the hull-vertex markers so the hull points stand out.
        ax.scatter(
            tour_closed[:, 0],
            tour_closed[:, 1],
            s=6,
            color="black",
            zorder=2,
        )

        # (d) Convex hull: dotted gray border. Recompute per instance because
        # `check_convex_hull` only returns the last instance's hull_seq (the
        # Python-scalar loop clobbers earlier instances).
        try:
            hull_idx = ConvexHull(tour_np).vertices
        except Exception:
            hull_idx = None

        if hull_idx is not None and len(hull_idx) >= 3:
            hpts = tour_np[list(hull_idx) + [int(hull_idx[0])]]
            ax.plot(
                hpts[:, 0],
                hpts[:, 1],
                linestyle=":",
                color="gray",
                linewidth=1.2,
                alpha=0.8,
                zorder=1,
            )
            # Hull vertices are colored by consistency: blue when the
            # convex-hull constraint is satisfied, red when it is violated.
            # Regular marker size (s=14) to stay clean and not overpower
            # the tour edges.
            hull_verts = tour_np[hull_idx]
            if hull_fail[idx].item():
                ax.scatter(
                    hull_verts[:, 0],
                    hull_verts[:, 1],
                    s=14,
                    facecolor="red",
                    edgecolor="darkred",
                    linewidth=0.6,
                    zorder=4,
                )
            else:
                ax.scatter(
                    hull_verts[:, 0],
                    hull_verts[:, 1],
                    s=14,
                    facecolor="blue",
                    edgecolor="darkblue",
                    linewidth=0.6,
                    zorder=4,
                )

        # (e) Cosmetics.
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        n_int = int(num_intersections[idx].item())
        hull_lbl = "FAIL" if hull_fail[idx].item() else "OK"
        ax.set_title(f"#{idx}  inter={n_int}  hull={hull_lbl}", fontsize=9)

    # Hide any unused axes.
    for j in range(K, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"Problematic TSP tours — {tag or 'unlabeled'}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    os.makedirs(out_dir, exist_ok=True)
    fname = filename or f"tour_diag_{tag or 'tours'}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_tsp_tours_purity(
    tours,
    purity_order=None,
    has_intersection=None,
    num_intersections=None,
    intersection_results=None,
    consistency=None,
    out_dir="results",
    filename=None,
    max_plots=16,
    tag=None,
):
    """Render a grid PNG of TSP tours with each edge colored by its
    purity order K_p.

    Pairs naturally with :func:`plot_tsp_tours`, which marks problematic
    edges in red on top of a faint base tour; this function instead
    colors every edge by its purity category.

    Because K_p is an integer with a small effective range (most edges
    sit in {0, 1, 2+}), the colormap is **discrete** with three bins:

    - ``0``   (pure)            → C0 (matplotlib blue, the project's
      "base / neutral" color)
    - ``1``   (one interior)    → C1 (matplotlib orange, neutral)
    - ``2+``  (two or more)     → C3 (matplotlib red, the project's
      "problematic" color)

    The colorbar is a discrete step legend with three ticks. Tours are
    rendered in batch order, capped at ``max_plots``.

    Args:
        tours: (B, N, 2) tensor of node coordinates in tour order.
            The tour is treated as a closed loop (edge k is
            ``(tours[b, k], tours[b, (k+1) % N])``).
        purity_order: optional (B, N) per-edge tensor. If ``None`` it is
            computed via :func:`check_purity_order` on ``tours``.
        has_intersection: optional (B,) bool tensor. If ``None`` it is
            recomputed.
        num_intersections: optional (B,) int tensor. If ``None`` it is
            recomputed.
        intersection_results: optional (B, N, N) bool tensor. If
            ``None`` it is recomputed.
        consistency: optional (B,) float tensor (1.0 = hull satisfied).
            If ``None`` it is recomputed.
        out_dir: directory to write the PNG into. Created if missing.
        filename: explicit filename. Defaults to
            ``f"tour_purity_{tag or 'tours'}.png"``.
        max_plots: hard cap on the number of subplots. When B is
            larger, only the first ``max_plots`` tours are rendered.
        tag: short label used in the figure suptitle and default
            filename (e.g. ``"tsp500_w10"``).

    Returns:
        out_path (str): absolute path of the saved PNG, or ``None`` if
        there is nothing to plot (N < 3).
    """
    B, N, _ = tours.shape
    if N < 3:
        print("[plot_tsp_tours_purity] N < 3 — nothing to plot.")
        return None

    if purity_order is None:
        purity_order = check_purity_order(tours)
    if has_intersection is None or num_intersections is None or intersection_results is None:
        has_i, n_i, _, _, inter = check_no_self_intersection(tours)
        has_intersection = has_i if has_intersection is None else has_intersection
        num_intersections = n_i if num_intersections is None else num_intersections
        intersection_results = inter if intersection_results is None else intersection_results
    if consistency is None:
        consistency, _ = check_convex_hull(tours)

    K = min(B, max_plots)
    cols = min(4, K)
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.0 * cols, 4.0 * rows), squeeze=False
    )
    axes_flat = axes.flatten()

    # Three discrete purity categories: 0 / 1 / 2+. The color palette
    # matches the project's existing convention — C0 (blue) for the
    # base/neutral state, C1 (orange) for intermediate, C3 (red) for
    # the "problematic" state.
    category_colors = ["C0", "C1", "C3"]
    category_labels = ["0 (pure)", "1", "2+"]

    def _category_color(k_p_value: int) -> str:
        """Map an integer K_p value to one of the 3 category colors."""
        if k_p_value <= 0:
            return category_colors[0]
        if k_p_value == 1:
            return category_colors[1]
        return category_colors[2]

    for plot_pos in range(K):
        idx = plot_pos  # render in batch order, capped at K
        ax = axes_flat[plot_pos]
        tour_np = tours[idx].detach().cpu().numpy()
        po_np = purity_order[idx].detach().cpu().numpy()

        # (a) Faint base loop so the underlying tour structure is
        # visible when the colored edges overlap densely. C0 with
        # reduced alpha so the category colors still read clearly.
        tour_closed = np.concatenate([tour_np, tour_np[:1]], axis=0)
        ax.plot(
            tour_closed[:, 0],
            tour_closed[:, 1],
            color="lightgray",
            linewidth=0.6,
            alpha=0.5,
            zorder=1,
        )

        # (b) Edges colored by purity category. Drawn on top of the
        # base loop with linewidth 2.0 so each segment reads
        # individually. The closed-loop convention is preserved by
        # the (k+1) % N wrap.
        for k in range(N):
            p1 = tour_np[k]
            p2 = tour_np[(k + 1) % N]
            c = _category_color(int(po_np[k]))
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=c,
                linewidth=2.0,
                alpha=0.95,
                zorder=2,
            )

        # (c) Node markers (small black dots at every tour vertex).
        ax.scatter(
            tour_closed[:, 0],
            tour_closed[:, 1],
            s=6,
            color="black",
            zorder=3,
        )

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        n_pure = int((po_np == 0).sum())
        n_one = int((po_np == 1).sum())
        n_high = int((po_np >= 2).sum())
        ax.set_title(
            f"#{idx}  K_p=0:{n_pure}  K_p=1:{n_one}  K_p≥2:{n_high}/{N}",
            fontsize=9,
        )

    # Hide unused axes.
    for j in range(K, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"TSP tours colored by purity order — {tag or 'unlabeled'}",
        fontsize=11,
    )

    # Discrete colorbar with 3 category ticks. We use a manual
    # legend so the labels show the actual category names
    # ("0 (pure)", "1", "2+") rather than the bin edges. This is
    # placed as its own axes on the right side of the figure.
    cbar_ax = fig.add_axes((0.96, 0.15, 0.015, 0.7))
    patches = [
        mpatches.Patch(color=category_colors[i], label=category_labels[i])
        for i in range(3)
    ]
    cbar_ax.legend(
        handles=patches,
        loc="center",
        frameon=False,
        title="purity order K_p",
    )
    cbar_ax.axis("off")

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, right=0.94)

    os.makedirs(out_dir, exist_ok=True)
    fname = filename or f"tour_purity_{tag or 'tours'}.png"
    out_path = os.path.join(out_dir, fname)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# =====================================================================
# GLOP vs LKH-3 edge-diff visualization
# =====================================================================


def _tour_edge_set(permutation):
    """Closed-loop edge set of a TSP tour expressed as a permutation of
    the original 0-indexed node ids. Edge ``(perm[k], perm[k+1 mod N])``
    is the city-pair visited consecutively in the tour.

    Returns a set of canonical sorted tuples ``(min, max)`` so two
    tours describe equal edge sets when they share the same cyclic
    traversal of city-pairs regardless of starting position or
    direction. Used to compare GLOP and LKH tours on a TSP instance.
    """
    perm = np.asarray(permutation, dtype=np.int64).reshape(-1)
    n = perm.shape[0]
    if n < 2:
        return set()
    a = perm
    b = np.roll(perm, -1)  # successor of each position
    edges = np.stack([a, b], axis=1)
    # Canonical sorted tuple per edge — rotation/reversal invariant.
    canonical = np.sort(edges, axis=1)
    return {tuple(e) for e in canonical.tolist()}


def _recover_permutation_from_coords(tour_coords, original_coords, atol=1e-6):
    """Recover the integer permutation ``pi`` such that
    ``original_coords[pi[k]] == tour_coords[k]`` for every tour position
    ``k``.

    GLOP stores tours as ``(N, 2)`` coordinate tensors in traversal
    order; the original index of each visited city is recoverable by
    matching ``tour_coords[k]`` against the canonical ``original_coords``
    array. The match is exact because ``LCP_TSP`` and the 2-opt post-fix
    only ever re-order coordinates via ``torch.gather`` / slicing /
    ``torch.flip`` — no float-rounding in those operations.

    Args:
        tour_coords: (N, 2) numpy or torch tensor.
        original_coords: (N, 2) reference (the un-permuted dataset
            coordinate array).
        atol: absolute tolerance for the match (default 1e-6 covers
            any float storage/loader-side rounding).
    """
    t = np.asarray(tour_coords, dtype=np.float64).reshape(-1, 2)
    o = np.asarray(original_coords, dtype=np.float64).reshape(-1, 2)
    n = t.shape[0]
    assert o.shape[0] == n, f"N mismatch: tour has N={n}, coords have {o.shape[0]}"
    # distance matrix ``(N, N)``; pick the closest original index per
    # tour position. Each row has exactly one true-or-near-zero entry
    # because GLOP only re-orders, never drops or duplicates cities.
    diff = t[:, None, :] - o[None, :, :]
    d2 = (diff * diff).sum(axis=-1)
    pi = np.argmin(d2, axis=1)
    if atol is not None:
        # Sanity check: every match should be a true copy of the
        # original coordinate, within numerical noise.
        per_row_min = d2[np.arange(n), pi]
        if per_row_min.max() > (atol * atol):
            # One or more rows couldn't be matched within tolerance —
            # surface a clear error rather than silently producing a
            # wrong permutation.
            worst = int(per_row_min.argmax())
            raise ValueError(
                f"tour_pos[{worst}]={t[worst].tolist()} could not be "
                f"matched within atol={atol} against original_coords; "
                f"nearest match distance={per_row_min[worst] ** 0.5:.3e}. "
                "Was GLOP applied to a subset or transformed copy of "
                "the original coordinates?"
            )
    return pi


def plot_glop_lkh_diff(
    coords,
    glop_tour,
    lkh_tour_or_pi,
    gap_pct,
    glop_cost,
    lkh_cost,
    instance_idx,
    out_path,
    tag=None,
):
    """Render a side-by-side GLOP-vs-LKH-3 tour comparison with the
    symmetric edge-set diff highlighted.

    Each panel draws:
      - the panel's tour as a closed loop;
      - edges shared with the other solver in light gray (faint, so
        common structure recedes);
      - edges *unique* to this panel's solver in a saturated color
        (red for GLOP, blue for LKH);
      - all cities as small black dots.

    Args:
        coords: (N, 2) original city coordinates (numpy or torch) — the
            reference frame both tours live in.
        glop_tour: (N, 2) GLOP tour coordinates in traversal order, OR
            a length-N permutation of the original indices. The two
            cases are detected by ``ndim == 1``.
        lkh_tour_or_pi: same convention as ``glop_tour`` for LKH-3.
        gap_pct: float, 100 * (cost_glop - cost_lkh) / cost_lkh. Drawn
            in the suptitle; positive = GLOP is worse.
        glop_cost, lkh_cost: floats used as panel-title annotations.
        instance_idx: integer — used in suptitle and default filename
            suggestion.
        out_path: explicit destination path. Created; parent dirs are
            made automatically.
        tag: optional short label appended to the suptitle.
    """
    o = np.asarray(coords, dtype=np.float64).reshape(-1, 2)

    # Accept either a coordinate tour or a permutation. Coordinate
    # tours get recovered into the original-index permutation via an
    # exact-match lookup; permutation input is used directly.
    def _to_permutation(tour, ref):
        arr = np.asarray(tour)
        if arr.ndim == 1:
            return arr.astype(np.int64)
        return _recover_permutation_from_coords(arr, ref)

    pi_glop = _to_permutation(glop_tour, o)
    pi_lkh = _to_permutation(lkh_tour_or_pi, o)

    edges_glop = _tour_edge_set(pi_glop)
    edges_lkh = _tour_edge_set(pi_lkh)
    shared = edges_glop & edges_lkh
    glop_only = edges_glop - shared
    lkh_only = edges_lkh - shared

    n = o.shape[0]

    def _draw_panel(ax, perm, edges_only, color_other, color_unique, title, cost):
        # Build the closed-loop path in coordinate space. Coordinates
        # for each vertex come from ``original[perm]`` — using the
        # permutation keeps orientation consistent with which solver
        # produced the tour.
        order = np.asarray(perm, dtype=np.int64)
        path = o[order]
        # Close the loop.
        path_closed = np.concatenate([path, path[:1]], axis=0)

        # Shared edges first (faint, behind everything) so unique
        # edges read clearly on top.
        if shared:
            for i, j in shared:
                ax.plot(
                    [o[i, 0], o[j, 0]],
                    [o[i, 1], o[j, 1]],
                    color="lightgray",
                    linewidth=0.8,
                    alpha=0.5,
                    zorder=1,
                )

        # Base tour loop (medium prominence, sells the silhouette even
        # on edges that are shared).
        ax.plot(
            path_closed[:, 0],
            path_closed[:, 1],
            color=color_other,
            linewidth=0.6,
            alpha=0.35,
            zorder=2,
        )

        # Solver-specific unique edges, drawn on top in a saturated
        # accent color that contrasts with the base loop tint.
        for i, j in edges_only:
            ax.plot(
                [o[i, 0], o[j, 0]],
                [o[i, 1], o[j, 1]],
                color=color_unique,
                linewidth=2.0,
                alpha=0.95,
                zorder=3,
            )

        # Vertex markers (small black dots) — placed last so they
        # remain visible above the edge overlays.
        ax.scatter(
            path_closed[:, 0],
            path_closed[:, 1],
            s=4,
            color="black",
            zorder=4,
        )

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}\ncost={cost:.4f}", fontsize=9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), squeeze=False)
    ax_glop, ax_lkh = axes[0]

    _draw_panel(
        ax_glop,
        pi_glop,
        glop_only,
        color_other="C0",  # GLOP base loop: matplotlib default blue
        color_unique="C3",  # GLOP-only edges: red — distinct accent
        title=f"GLOP  (n={n}, |unique|={len(glop_only)})",
        cost=glop_cost,
    )
    _draw_panel(
        ax_lkh,
        pi_lkh,
        lkh_only,
        color_other="C1",  # LKH base loop: matplotlib default orange
        color_unique="C0",  # LKH-only edges: blue — distinct accent
        title=f"LKH-3  (n={n}, |unique|={len(lkh_only)})",
        cost=lkh_cost,
    )

    suptitle_tag = f"  [{tag}]" if tag else ""
    fig.suptitle(
        f"Instance #{instance_idx}  gap={gap_pct:+.2f}%{suptitle_tag}\n"
        f"|common|={len(shared)}  |glop-only|={len(glop_only)}  "
        f"|lkh-only|={len(lkh_only)}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    out_dir = os.path.dirname(out_path) or "."
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
