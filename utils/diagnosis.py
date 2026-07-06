# Check if heuristics for optimal TSP tour are valid. Indicating whether decompose-on-edge is req'd

import os

import numpy as np
import torch
from scipy.spatial import ConvexHull

import matplotlib
matplotlib.use("Agg")  # headless backend; safe even when no DISPLAY is present
import matplotlib.pyplot as plt


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
    """
    initial_tours = tours.detach().clone()
    B, N, _ = tours.shape

    def _closed_loop_cost(t):
        return (
            (t[:, 1:] - t[:, :-1]).norm(p=2, dim=2).sum(1)
            + (t[:, 0] - t[:, -1]).norm(p=2, dim=1)
        )

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

    for it in range(max_iter):
        iters_used = it + 1
        has_inter, n_inter, _, _, inter = check_no_self_intersection(tours)

        if not has_inter.any():
            final_intersections_count = 0
            break

        any_change = False
        for b in range(B):
            if not has_inter[b]:
                continue

            pairs = torch.nonzero(inter[b], as_tuple=False)
            if pairs.numel() == 0:
                continue

            i_all = pairs[:, 0]
            j_all = pairs[:, 1]
            # Defensive: skip any pair that is not strictly non-adjacent.
            # The detector's upper-triangular + adj-mask already guarantees
            # this, but a future refactor could relax the assumption.
            mask = (j_all > i_all + 1)
            i_all = i_all[mask]
            j_all = j_all[mask]
            if i_all.numel() == 0:
                continue

            for k in range(i_all.numel()):
                i = int(i_all[k].item())
                j = int(j_all[k].item())
                tours[b, i + 1 : j + 1] = torch.flip(
                    tours[b, i + 1 : j + 1], dims=[0]
                )
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
        print(
            f"[plot_tsp_tours] {total_bad} failing tours; rendering first {K}."
        )

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
