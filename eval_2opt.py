"""Evaluate and compare GLOP with 2-opt post-processing.

Runs the SAME TSP instances through several configurations and reports both the
final tour cost and the per-iteration convergence of each:

  1. baseline            — GLOP only (no 2-opt)
  2. final               — GLOP, then full 2-opt once at the end of the pipeline
  3. per_iter            — GLOP with full 2-opt applied after every revisor iter
  4. knn_final           — GLOP, then KNN-sparse 2-opt once at the end
  5. knn_per_iter        — GLOP with KNN-sparse 2-opt after every revisor iter
  6. radius_final        — GLOP, then tour-position-radius 2-opt at the end
  7. radius_per_iter     — GLOP with tour-position-radius 2-opt per iter
  8. range_final         — GLOP, then range-radius 2-opt with [2, 5] at the end
  9. range_per_iter      — GLOP with range-radius 2-opt [2, 5] per iter
 10. range_wide_final    — GLOP, then range-radius 2-opt [2, N//10] at the end
 11. range_wide_per_iter — GLOP with range-radius 2-opt [2, N//10] per iter

Outputs:
  * a printed summary table of final performance (avg / best cost, duration)
  * a figure (results/twoopt_compare_<tag>.png) with two panels:
      - per-iteration convergence curve, one line per mode
      - final-cost bar chart with % improvement vs. baseline

The runs reuse a single set of loaded revisers and reset the RNG seed before
each run, so the warm-start tours are identical across modes and the
comparison is fair.

Example:
  python eval_2opt.py --problem_size 100 --revision_lens 50 20 \
      --revision_iters 10 5 --width 4 --eval_batch_size 8 --val_size 8 \
      --decode_strategy greedy --two_opt_iters 30 --two_opt_knn_k 15
"""

import argparse
import copy
import math
import os
import time

import numpy as np
import torch

from utils import load_model

# Reuse the exact eval machinery from main.py so this script tracks the pipeline.
from main import _eval_dataset, _aggregate_solver_curve


def _is_oom_error(exc):
    """True if `exc` looks like a CUDA / CPU out-of-memory error.

    Used by ``_safe_run_mode`` to decide whether to swallow the exception
    (skip the mode, keep evaluating the rest) or let it propagate.
    """
    # CUDA OOM, available as a distinct subclass since PyTorch 1.13.
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    # Some CUDA OOMs surface as a plain RuntimeError on older / custom builds.
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "cuda out of memory" in msg or "out of memory" in msg:
            return True
    # CPU OOM (numpy.memmap etc.).
    if isinstance(exc, MemoryError):
        return True
    return False


# Deterministic per-label fallback colors for plot dicts. Hashing the label
# string gives every unknown label a stable, distinct color across panels
# (line plot, bar chart, swap-distance histogram) without requiring an entry
# in the hand-picked ``colors`` dict — useful when users add custom mode
# rows like ``decomp_per_iter_5`` with radius baked into the name.
_RANDOM_MARKERS = ("o", "s", "D", "v", "^", "P", "X", "p", "h", "*", "d", "<", ">")


def _random_color(label):
    """Stable hex color for an unknown plot label.

    Hashes the label so the same label always gets the same color across
    all panels of the comparison figure, while different labels produce
    visually distinct hues.
    """
    import hashlib

    return "#" + hashlib.md5(label.encode("utf-8")).hexdigest()[:6]


def _random_marker(label):
    """Stable matplotlib marker for an unknown plot label.

    Pairs with ``_random_color`` to keep unknown-label series visually
    distinguishable in the line plot.
    """
    import hashlib

    return _RANDOM_MARKERS[
        int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % len(_RANDOM_MARKERS)
    ]


def _safe_run_mode(
    label,
    use_2opt,
    two_opt_mode,
    two_opt_kind,
    two_opt_knn_k,
    two_opt_radius,
    two_opt_radius_min,
    two_opt_radius_max,
    two_opt_decomp_radius,
    two_opt_sampling_base,
    two_opt_sampling_r,
    base_opts,
    revisers,
):
    """Run one configuration; on OOM, return a sentinel row instead of raising.

    Returns the same dict shape as ``run_mode`` with two extra keys:
        ``skipped`` (bool): True iff this row is an OOM fallback.
        ``error``   (str):  textual error message when skipped.

    Other exception types (FileNotFoundError, user errors, etc.) are
    propagated untouched.
    """
    try:
        return run_mode(
            label,
            use_2opt,
            two_opt_mode,
            two_opt_kind,
            two_opt_knn_k,
            two_opt_radius,
            two_opt_radius_min,
            two_opt_radius_max,
            two_opt_decomp_radius,
            two_opt_sampling_base,
            two_opt_sampling_r,
            base_opts,
            revisers,
        )
    except BaseException as exc:
        if not _is_oom_error(exc):
            raise
        # Free what we can so subsequent modes have a chance to fit.
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print(
            f"\n[ERROR] OOM in mode={label} "
            f"(kind={two_opt_kind}, knn_k={two_opt_knn_k}, "
            f"radius={two_opt_radius}, "
            f"r_min={two_opt_radius_min}, r_max={two_opt_radius_max}, "
            f"decomp_r={two_opt_decomp_radius}, "
            f"samp_base={two_opt_sampling_base}, samp_r={two_opt_sampling_r}); "
            f"skipping this mode. error={exc!r}\n",
            flush=True,
        )
        return {
            "label": label,
            "two_opt_kind": two_opt_kind,
            "two_opt_knn_k": two_opt_knn_k,
            "two_opt_radius": two_opt_radius,
            "two_opt_radius_min": two_opt_radius_min,
            "two_opt_radius_max": two_opt_radius_max,
            "two_opt_decomp_radius": two_opt_decomp_radius,
            "two_opt_sampling_base": two_opt_sampling_base,
            "two_opt_sampling_r": two_opt_sampling_r,
            "avg": float("nan"),
            "best": float("nan"),
            "duration": float("nan"),
            "curve": {},
            "swap_dists": [],
            "skipped": True,
            "error": str(exc),
        }


# Modes to compare:
# (label, use_2opt, two_opt_mode, two_opt_kind,
#  two_opt_knn_k, two_opt_radius, two_opt_radius_min, two_opt_radius_max,
#  two_opt_decomp_radius, two_opt_sampling_base, two_opt_sampling_r)
#
# ``two_opt_radius`` is only consulted when ``two_opt_kind == 'radius'``;
# ``None`` makes the dispatcher fall back to ``max(2, N // 10)`` per instance.
# ``two_opt_radius_min`` / ``two_opt_radius_max`` are only consulted when
# ``two_opt_kind == 'range_radius'``; ``None`` for either makes the dispatcher
# fall back to ``(2, max(2, N // 10))``.
# ``two_opt_decomp_radius`` is only consulted when ``two_opt_kind == 'decomp'``;
# ``None`` makes the dispatcher fall back to ``max(2, revision_len // 10)``.
# ``two_opt_sampling_base`` / ``two_opt_sampling_r`` are only consulted when
# ``two_opt_kind == 'sampling_radius'``; defaults are ``2`` and ``None`` with
# ``None`` for ``r`` making the dispatcher fall back to
# ``max(2, N // 10)`` per instance.
def make_modes(opts):
    """Build the comparison table once ``opts.problem_size`` is known.

    The wide-window range-radius rows scale with the instance size, so the
    MODES table is built per-call rather than at module load.

    Returns a list of 11-tuples, one per comparison row. Each tuple has the
    following positional fields:

      0. ``label`` (str) — short name shown in the printed summary table,
         figure legend, swap-distance histogram, and used as the filename
         tag component. Must be unique across rows. The ``print_table``
         helper also infers the ``two_opt_kind`` from the label prefix:
         ``knn_*`` → ``knn``, ``range_*`` → ``range_radius``,
         ``sampling_*`` → ``sampling_radius``, ``radius_*`` → ``radius``,
         ``decomp_*`` → ``decomp``, anything else → ``full``.

      1. ``use_2opt`` (bool) — master switch. When ``False`` the pipeline
         skips the optional 2-opt step entirely (this is the GLOP-only
         baseline). When ``True``, the pipeline invokes the variant
         selected by ``two_opt_kind``.

      2. ``two_opt_mode`` (``'final'`` | ``'per_iter'``) — controls *when*
         the 2-opt runs. ``'final'`` runs it once after the whole revisor
         chain; ``'per_iter'`` runs it after every revisor iteration.
         ``'decomp'`` is only valid with ``'per_iter'``.

      3. ``two_opt_kind`` (``'full'`` | ``'knn'`` | ``'radius'`` |
         ``'range_radius'`` | ``'sampling_radius'`` | ``'decomp'``) —
         selects the algorithm. Unknown values trigger a ``UserWarning``
         and fall back to ``'full'``.

      4. ``two_opt_knn_k`` (int | None) — ``k`` for KNN-sparse 2-opt.
         Only consulted when ``two_opt_kind == 'knn'``. ``None`` means the
         dispatcher uses its default (currently 20).

      5. ``two_opt_radius`` (int | None) — ``r`` for fixed-radius
         tour-position 2-opt. Only consulted when ``two_opt_kind ==
         'radius'``. ``None`` means the dispatcher falls back to
         ``max(2, N // 10)`` per instance.

      6. ``two_opt_radius_min`` (int | None) — inclusive lower bound of
         the offset window for ``two_opt_kind == 'range_radius'``. Only
         consulted for range-radius rows. ``None`` means the dispatcher
         falls back to ``2``. Must be ``>= 2`` if supplied (offsets 0
         and ±1 are invalid 2-opt moves).

      7. ``two_opt_radius_max`` (int | None) — inclusive upper bound of
         the offset window for ``two_opt_kind == 'range_radius'``. Only
         consulted for range-radius rows. ``None`` means the dispatcher
         falls back to ``max(2, N // 10)``. If the dispatcher finds
         ``r_min > r_max`` it emits a ``UserWarning`` and silently swaps
         the two values.

      8. ``two_opt_decomp_radius`` (int | None) — half-width of the seam
         neighbourhood along the tour for ``two_opt_kind == 'decomp'``.
         Only consulted for ``decomp`` rows. ``None`` means the dispatcher
         falls back to ``max(2, revision_len // 10)``.

      9. ``two_opt_sampling_base`` (int | None) — inclusive starting
         offset for ``two_opt_kind == 'sampling_radius'``. Only consulted
         for sampling-radius rows. ``None`` means the dispatcher falls
         back to ``2``. Must be ``>= 2`` if supplied (offsets 0 and ±1
         are invalid 2-opt moves).

     10. ``two_opt_sampling_r`` (int | None) — window size beyond
         ``two_opt_sampling_base`` for ``two_opt_kind == 'sampling_radius'``.
         Only consulted for sampling-radius rows. ``None`` means the
         dispatcher falls back to ``max(2, N // 10)``. If the dispatcher
         finds ``base > r`` it emits a ``UserWarning`` and silently swaps
         the two values.

    Downstream consumers all read rows positionally, so any change here is
    the only edit needed to add / remove / retune a comparison row:

      * ``colors`` / ``markers`` dicts in ``plot_comparison`` and
        ``plot_swap_distance_histograms`` colour-code each label (new
        labels fall back to matplotlib's default cycle if absent).
      * ``_build_tag`` and the figure suptitle suffix append a
        ``kind=`` / ``k=`` / ``r=`` / ``r=[min,max]`` suffix derived from
        ``opts.two_opt_kind``.
    """
    N = max(2, opts.problem_size // 10)  # problem_size_based
    C = max(2, opts.revision_lens[0])
    return [
        (
            "baseline",
            False,
            "final",
            "full",
            20,
            None,
            None,
            None,
            None,
            None,
            None,  # sampling_base, sampling_r
        ),
        # ("final", True, "final", "full", 20, None, None, None, None, None, None),
        # ("per_iter", True, "per_iter", "full", 20, None, None, None, None, None, None),
        # ("knn_final", True, "final", "knn", 20, None, None, None, None, None, None),
        # ("knn_per_iter", True, "per_iter", "knn", 20, None, None, None, None, None, None),
        # ("radius_final", True, "final", "radius", 20, None, 0.5 * C, 1.5 * C, None, None, None),
        # ("radius_per_iter", True, "per_iter", "radius", 20, None, N, 2 * N, None, None, None),
        # ("range_final", True, "final", "range_radius", 20, None, N, 3 * N, None, None, None),
        # ("range_per_iter", True, "per_iter", "range_radius", 20, None, 0.5 * N, 5 * N, None, None, None,),
        # ("decomp_per_iter_5",True,"per_iter", "decomp",20,None,None,None,5,  # Decomp radius sizeNone,None,  # sampling_base, sampling_r),
        (
            "sampling_per_iter_wide",
            True,
            "per_iter",
            "sampling_radius",
            20,
            None,
            None,
            None,
            None,
            50,  # sampling_base
            100,  # explicit wide window (N = 100)
        ),
    ]


# Each row's `use_2opt` flags whether the pipeline invokes the optional 2-opt
# step; `two_opt_kind` selects full vs. knn vs. radius vs. range_radius in
# ``maybe_two_opt`` (post_process.py).


def build_base_opts():
    """Argparse for the subset of options relevant to a TSP 2-opt comparison.

    Field names mirror main.py so the reused pipeline code finds what it needs.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--problem_size", type=int, default=100)
    p.add_argument(
        "--problem_type",
        type=str,
        default="tsp",
        choices=["tsp"],
        help="This comparison script targets TSP.",
    )
    p.add_argument(
        "--path",
        type=str,
        default="",
        help="Test dataset path; defaults to data/tsp/tsp<size>_test.pkl",
    )
    p.add_argument("--val_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--width", type=int, default=4)
    p.add_argument("--revision_lens", nargs="+", type=int, default=[50, 20])
    p.add_argument("--revision_iters", nargs="+", type=int, default=[10, 5])
    p.add_argument(
        "--decode_strategy",
        type=str,
        default="greedy",
        help="'greedy' recommended for a deterministic comparison",
    )
    p.add_argument(
        "--two_opt_iters",
        type=int,
        default=30,
        help="Max 2-opt sweeps per invocation (final and per_iter modes)",
    )
    p.add_argument(
        "--two_opt_kind",
        type=str,
        default="full",
        choices=["full", "knn", "radius", "range_radius", "sampling_radius", "decomp"],
        help="2-opt algorithm variant: 'full' (dense), "
        "'knn' (k-NN-sparse; uses --two_opt_knn_k), "
        "'radius' (tour-position-sparse; uses --two_opt_radius), "
        "'range_radius' (tour-position-sparse over "
        "[r_min, r_max]; uses --two_opt_radius_min / "
        "--two_opt_radius_max), "
        "'sampling_radius' (shifting-window tour-position-sparse; "
        "uses --two_opt_sampling_base / --two_opt_sampling_r), or "
        "'decomp' (decomposition-aware; only valid with "
        "--two_opt_mode=per_iter; uses --two_opt_decomp_radius).",
    )
    p.add_argument(
        "--two_opt_knn_k",
        type=int,
        default=20,
        help="k for KNN-sparse 2-opt (only used when --two_opt_kind=knn)",
    )
    p.add_argument(
        "--two_opt_radius",
        type=int,
        default=None,
        help="r for radius-sparse 2-opt (only used when "
        "--two_opt_kind=radius). Default: 10%% of "
        "--problem_size (floored at 2).",
    )
    p.add_argument(
        "--two_opt_radius_min",
        type=int,
        default=2,
        help="r_min for range-radius 2-opt (only used when "
        "--two_opt_kind=range_radius). Default: 2.",
    )
    p.add_argument(
        "--two_opt_radius_max",
        type=int,
        default=None,
        help="r_max for range-radius 2-opt (only used when "
        "--two_opt_kind=range_radius). Default: 10%% of "
        "--problem_size (floored at max(r_min, 2)).",
    )
    p.add_argument(
        "--two_opt_sampling_base",
        type=int,
        default=2,
        help="Starting offset for sampling-radius 2-opt (only used when "
        "--two_opt_kind=sampling_radius). Default: 2.",
    )
    p.add_argument(
        "--two_opt_sampling_r",
        type=int,
        default=None,
        help="Window size beyond base for sampling-radius 2-opt (only "
        "used when --two_opt_kind=sampling_radius). Default: 10%% of "
        "--problem_size (floored at max(base, 2)).",
    )
    p.add_argument(
        "--two_opt_decomp_radius",
        type=int,
        default=None,
        help="r for decomp 2-opt — the seam neighbourhood "
        "half-width along the tour (only used when "
        "--two_opt_kind=decomp). Default: "
        "max(2, revision_len // 10).",
    )
    p.add_argument(
        "--two_opt_debug",
        action="store_true",
        help="Print per-sweep 2-opt phase timings to stdout for "
        "performance investigation.",
    )
    p.add_argument(
        "--record_two_opt_swaps",
        action="store_true",
        help="Record |i_star - j_star| for every accepted 2-opt "
        "move and emit a swap-distance histogram PNG "
        "(results/twoopt_swap_dists_<tag>.png).",
    )
    p.add_argument("--no_aug", action="store_true")
    p.add_argument("--no_prune", action="store_true")
    p.add_argument("--no_progress_bar", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="results")

    opts = p.parse_args()

    # Fields the reused pipeline reads but this CLI does not expose.
    opts.n_subset = 1
    opts.n_partition = 1
    opts.ckpt_path = ""
    opts.tsp_aug = False  # _eval_dataset may flip this for problem_size <= 100

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.device = torch.device(f"cuda:{opts.device_id}" if use_cuda else "cpu")

    if opts.path == "":
        opts.path = f"data/tsp/tsp{opts.problem_size}_test.pkl"
    return opts


def load_revisers(opts):
    revisers = []
    for reviser_size in opts.revision_lens:
        reviser_path = f"pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt"
        reviser, _ = load_model(reviser_path, is_local=True)
        reviser.to(opts.device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)
        revisers.append(reviser)
    return revisers


def final_costs(results):
    """Extract final tour costs (per instance) from _eval_dataset results (TSP)."""
    costs_revised = torch.cat([r[1] for r in results], dim=0)
    return costs_revised


def run_mode(
    label,
    use_2opt,
    two_opt_mode,
    two_opt_kind,
    two_opt_knn_k,
    two_opt_radius,
    two_opt_radius_min,
    two_opt_radius_max,
    two_opt_decomp_radius,
    two_opt_sampling_base,
    two_opt_sampling_r,
    base_opts,
    revisers,
):
    """Run one configuration on a fresh copy of opts with a reset RNG."""
    opts = copy.deepcopy(base_opts)
    opts.use_2opt = use_2opt
    opts.two_opt_mode = two_opt_mode
    opts.two_opt_kind = two_opt_kind
    opts.two_opt_knn_k = two_opt_knn_k
    opts.two_opt_radius = two_opt_radius
    opts.two_opt_radius_min = two_opt_radius_min
    opts.two_opt_radius_max = two_opt_radius_max
    opts.two_opt_decomp_radius = two_opt_decomp_radius
    opts.two_opt_sampling_base = two_opt_sampling_base
    opts.two_opt_sampling_r = two_opt_sampling_r
    opts.two_opt_debug = getattr(base_opts, "two_opt_debug", False)
    opts.record_two_opt_swaps = getattr(base_opts, "record_two_opt_swaps", False)
    # Always allocate a sink so downstream code can read it unconditionally;
    # the actual recording inside full_2opt / knn_2opt is gated on the flag.
    swap_dists: list = []
    opts.two_opt_swap_sink = swap_dists

    # Reset RNG so every mode sees identical warm-start tours / sampling draws.
    torch.manual_seed(base_opts.seed)
    np.random.seed(base_opts.seed)

    print(
        f"\n===================== running mode: {label} "
        f"(use_2opt={use_2opt}, mode={two_opt_mode}, "
        f"kind={two_opt_kind}, knn_k={two_opt_knn_k}, "
        f"radius={two_opt_radius}, "
        f"r_min={two_opt_radius_min}, r_max={two_opt_radius_max}, "
        f"decomp_r={two_opt_decomp_radius}, "
        f"samp_base={two_opt_sampling_base}, samp_r={two_opt_sampling_r}) "
        f"====================="
    )
    results, duration, all_stats = _eval_dataset(opts.path, opts, opts.device, revisers)

    costs = final_costs(results)
    return {
        "label": label,
        "two_opt_kind": two_opt_kind,
        "two_opt_knn_k": two_opt_knn_k,
        "two_opt_radius": two_opt_radius,
        "two_opt_radius_min": two_opt_radius_min,
        "two_opt_radius_max": two_opt_radius_max,
        "two_opt_decomp_radius": two_opt_decomp_radius,
        "two_opt_sampling_base": two_opt_sampling_base,
        "two_opt_sampling_r": two_opt_sampling_r,
        "avg": costs.mean().item(),
        "best": costs.min().item(),
        "duration": duration,
        "curve": _aggregate_solver_curve(all_stats),  # {layer_id: {iters,best,avg}}
        "swap_dists": swap_dists,
    }


def flatten_curve(curve):
    """Concatenate per-layer 'best' series into one global convergence sequence.

    Returns (xs, ys, boundaries) where boundaries are the global x positions at
    which a new revisor layer starts (for drawing separators).
    """
    xs, ys, boundaries = [], [], []
    gx = 0
    for lid in sorted(curve.keys()):
        data = curve[lid]
        if gx > 0:
            boundaries.append(gx + 0.5)
        for b in data["best"]:
            gx += 1
            xs.append(gx)
            ys.append(b)
    return xs, ys, boundaries


def plot_comparison(runs, opts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline": "#7f7f7f",
        "final": "#1f77b4",
        "per_iter": "#d62728",
        "knn_final": "#2ca02c",
        "knn_per_iter": "#9467bd",
        "radius_final": "#ff7f0e",
        "radius_per_iter": "#8c564b",
        "range_final": "#e377c2",
        "range_per_iter": "#bcbd22",
        "range_wide_final": "#17becf",
        "range_wide_per_iter": "#9edae5",
        "sampling_per_iter_default": "#aec7e8",
        "sampling_per_iter_wide": "#ffbb78",
        "decomp_per_iter": "#dbdb8d",
    }
    markers = {
        "baseline": "o",
        "final": "s",
        "per_iter": "^",
        "knn_final": "D",
        "knn_per_iter": "v",
        "radius_final": "P",
        "radius_per_iter": "X",
        "range_final": "p",
        "range_per_iter": "h",
        "range_wide_final": "<",
        "range_wide_per_iter": ">",
        "sampling_per_iter_default": "*",
        "sampling_per_iter_wide": "+",
        "decomp_per_iter": "d",
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]}
    )

    # --- Panel 1: per-iteration convergence ---
    all_boundaries = []
    max_x = 0
    for r in runs:
        xs, ys, boundaries = flatten_curve(r["curve"])
        if not xs:
            continue
        all_boundaries = boundaries  # same layout across modes
        max_x = max(max_x, xs[-1])
        ax1.plot(
            xs,
            ys,
            "-",
            color=colors.get(r["label"], _random_color(r["label"])),
            marker=markers.get(r["label"], _random_marker(r["label"])),
            markersize=4,
            linewidth=1.8,
            label=f"{r['label']} (per-iter)",
        )
        # Terminal marker = the mode's FINAL cost after the whole pipeline
        # (captures the end-of-pipeline 2-opt drop for the 'final' mode).
        ax1.plot(
            [xs[-1], xs[-1] + 1],
            [ys[-1], r["avg"]],
            ":",
            color=colors.get(r["label"], _random_color(r["label"])),
            linewidth=1.2,
        )
        ax1.scatter(
            [xs[-1] + 1],
            [r["avg"]],
            color=colors.get(r["label"], _random_color(r["label"])),
            marker="*",
            s=130,
            zorder=5,
        )

    for bx in all_boundaries:
        ax1.axvline(bx, color="k", linestyle=":", alpha=0.25)
    ax1.set_xlabel("Global revisor iteration (layers concatenated;  ★ = final cost)")
    ax1.set_ylabel("Mean tour cost over eval set\n(best across --width)")
    ax1.set_title("Cost at every iteration")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    # --- Panel 2: final performance bar chart ---
    labels = [r["label"] for r in runs]
    avgs = [r["avg"] for r in runs]
    bests = [r["best"] for r in runs]
    base_avg = next((r["avg"] for r in runs if r["label"] == "baseline"), avgs[0])
    x = np.arange(len(labels))
    bars = ax2.bar(
        x,
        avgs,
        color=[colors.get(l, _random_color(l)) for l in labels],
        alpha=0.85,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Final mean tour cost")
    ax2.set_title("Final performance")
    ax2.grid(True, axis="y", alpha=0.3)
    lo = min(avgs) * 0.995
    hi = max(avgs) * 1.005
    ax2.set_ylim(lo, hi)
    for xi, (_, avg, best) in enumerate(zip(bars, avgs, bests)):
        impr = 100.0 * (base_avg - avg) / base_avg if base_avg else 0.0
        ax2.annotate(
            f"avg {avg:.3f}\nbest {best:.3f}\n({impr:+.2f}%)",
            xy=(xi, avg),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    has_knn = any(r["label"].startswith("knn") for r in runs)
    has_radius = any(r["label"].startswith("radius") for r in runs)
    has_range = any(r["label"].startswith("range") for r in runs)
    has_sampling = any(r["label"].startswith("sampling") for r in runs)
    has_decomp = any(r["label"].startswith("decomp") for r in runs)
    kind_suffix = (
        f", kind={opts.two_opt_kind}"
        + (f", k={opts.two_opt_knn_k}" if opts.two_opt_kind == "knn" else "")
        + (
            f", r={opts.two_opt_radius}"
            if (
                opts.two_opt_kind == "radius"
                and getattr(opts, "two_opt_radius", None) is not None
            )
            else ""
        )
        + (
            f", r=[{getattr(opts, 'two_opt_radius_min', 2)},"
            f"{getattr(opts, 'two_opt_radius_max', None)}]"
            if (
                opts.two_opt_kind == "range_radius"
                and getattr(opts, "two_opt_radius_min", None) is not None
                and getattr(opts, "two_opt_radius_max", None) is not None
            )
            else ""
        )
        + (
            f", samp=[{getattr(opts, 'two_opt_sampling_base', 2)},"
            f"{getattr(opts, 'two_opt_sampling_r', None)}]"
            if (
                opts.two_opt_kind == "sampling_radius"
                and getattr(opts, "two_opt_sampling_base", None) is not None
                and getattr(opts, "two_opt_sampling_r", None) is not None
            )
            else ""
        )
        + (
            f", decomp_r={opts.two_opt_decomp_radius}"
            if (
                opts.two_opt_kind == "decomp"
                and getattr(opts, "two_opt_decomp_radius", None) is not None
            )
            else ""
        )
        if (has_knn or has_radius or has_range or has_sampling or has_decomp)
        else ""
    )
    fig.suptitle(
        f"GLOP + 2-opt comparison — tsp{opts.problem_size}, width={opts.width}, "
        f"val_size={opts.val_size}, lens={opts.revision_lens}, iters={opts.revision_iters}, "
        f"2opt_iters={opts.two_opt_iters}{kind_suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    os.makedirs(opts.out_dir, exist_ok=True)
    has_knn = any(r["label"].startswith("knn") for r in runs)
    has_radius = any(r["label"].startswith("radius") for r in runs)
    has_range = any(r["label"].startswith("range") for r in runs)
    has_sampling = any(r["label"].startswith("sampling") for r in runs)
    has_decomp = any(r["label"].startswith("decomp") for r in runs)
    kind_tag = (
        f"_kind{opts.two_opt_kind}"
        + (f"_k{opts.two_opt_knn_k}" if opts.two_opt_kind == "knn" else "")
        + (
            f"_r{opts.two_opt_radius}"
            if (
                opts.two_opt_kind == "radius"
                and getattr(opts, "two_opt_radius", None) is not None
            )
            else ""
        )
        + (
            f"_rmin{opts.two_opt_radius_min}_rmax{opts.two_opt_radius_max}"
            if (
                opts.two_opt_kind == "range_radius"
                and getattr(opts, "two_opt_radius_min", None) is not None
                and getattr(opts, "two_opt_radius_max", None) is not None
            )
            else ""
        )
        + (
            f"_sampb{opts.two_opt_sampling_base}_sampr{opts.two_opt_sampling_r}"
            if (
                opts.two_opt_kind == "sampling_radius"
                and getattr(opts, "two_opt_sampling_base", None) is not None
                and getattr(opts, "two_opt_sampling_r", None) is not None
            )
            else ""
        )
        + (
            f"_decompr{opts.two_opt_decomp_radius}"
            if (
                opts.two_opt_kind == "decomp"
                and getattr(opts, "two_opt_decomp_radius", None) is not None
            )
            else ""
        )
        if (has_knn or has_radius or has_range or has_sampling or has_decomp)
        else ""
    )
    tag = (
        f"tsp{opts.problem_size}_w{opts.width}"
        f"_lens{'-'.join(map(str, opts.revision_lens))}"
        f"_iters{'-'.join(map(str, opts.revision_iters))}_2opt{opts.two_opt_iters}"
        f"{kind_tag}"
    )
    out_path = os.path.join(opts.out_dir, f"twoopt_compare_{tag}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _build_tag(opts):
    """Reconstruct the deterministic filename tag shared by every PNG in this run.

    Mirrors the tag built inside ``plot_comparison`` so the swap-distance
    histogram lines up with the comparison figure on disk.
    """
    runs_for_tag = getattr(opts, "_runs_for_tag", [])
    has_knn = any(r["label"].startswith("knn") for r in runs_for_tag)
    has_radius = any(r["label"].startswith("radius") for r in runs_for_tag)
    has_range = any(r["label"].startswith("range") for r in runs_for_tag)
    has_sampling = any(r["label"].startswith("sampling") for r in runs_for_tag)
    has_decomp = any(r["label"].startswith("decomp") for r in runs_for_tag)
    kind_tag = (
        f"_kind{opts.two_opt_kind}"
        + (f"_k{opts.two_opt_knn_k}" if opts.two_opt_kind == "knn" else "")
        + (
            f"_r{opts.two_opt_radius}"
            if (
                opts.two_opt_kind == "radius"
                and getattr(opts, "two_opt_radius", None) is not None
            )
            else ""
        )
        + (
            f"_rmin{opts.two_opt_radius_min}_rmax{opts.two_opt_radius_max}"
            if (
                opts.two_opt_kind == "range_radius"
                and getattr(opts, "two_opt_radius_min", None) is not None
                and getattr(opts, "two_opt_radius_max", None) is not None
            )
            else ""
        )
        + (
            f"_sampb{opts.two_opt_sampling_base}_sampr{opts.two_opt_sampling_r}"
            if (
                opts.two_opt_kind == "sampling_radius"
                and getattr(opts, "two_opt_sampling_base", None) is not None
                and getattr(opts, "two_opt_sampling_r", None) is not None
            )
            else ""
        )
        + (
            f"_decompr{opts.two_opt_decomp_radius}"
            if (
                opts.two_opt_kind == "decomp"
                and getattr(opts, "two_opt_decomp_radius", None) is not None
            )
            else ""
        )
        if (has_knn or has_radius or has_range or has_sampling or has_decomp)
        else ""
    )
    return (
        f"tsp{opts.problem_size}_w{opts.width}"
        f"_lens{'-'.join(map(str, opts.revision_lens))}"
        f"_iters{'-'.join(map(str, opts.revision_iters))}_2opt{opts.two_opt_iters}"
        f"{kind_tag}"
    )


def plot_swap_distance_histograms(runs, opts):
    """Plot the swap-distance distribution per mode as a sibling PNG.

    Distance is the cyclic short-arc hop along the closed tour:
    ``min(|i-j|, N - |i-j|)``. Bounded by ``N // 2`` since the smaller
    of the two arcs is at most half the tour.

    Two panels:
      * absolute short-arc distance (0..N/2) with a dashed line at
        ``two_opt_knn_k`` to surface the KNN candidate cap;
      * same data normalised to tour length (0..0.5) so cross-problem-size
        comparisons are meaningful.

    Skipped (returns ``None``) when no run recorded any swaps, i.e.
    ``--record_two_opt_swaps`` was not set or every 2-opt mode was OOM-skipped.
    """
    # Only consider runs that actually have data.
    plotted = [r for r in runs if r.get("swap_dists")]
    if not plotted:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline": "#7f7f7f",
        "final": "#1f77b4",
        "per_iter": "#d62728",
        "knn_final": "#2ca02c",
        "knn_per_iter": "#9467bd",
        "radius_final": "#ff7f0e",
        "radius_per_iter": "#8c564b",
        "range_final": "#e377c2",
        "range_per_iter": "#bcbd22",
        "range_wide_final": "#17becf",
        "range_wide_per_iter": "#9edae5",
        "sampling_per_iter_default": "#aec7e8",
        "sampling_per_iter_wide": "#ffbb78",
        "decomp_per_iter": "#dbdb8d",
    }

    fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(13, 5))

    N = opts.problem_size
    abs_max = max(1, N // 2)
    norm_max = 0.5
    n_bins_abs = min(abs_max, 50)
    n_bins_norm = 30

    for r in plotted:
        d = r["swap_dists"]
        color = colors.get(r["label"], _random_color(r["label"]))
        ax_abs.hist(
            d,
            bins=n_bins_abs,
            range=(0, abs_max),
            density=True,
            alpha=0.5,
            color=color,
            label=f"{r['label']} (n={len(d)})",
        )
        ax_norm.hist(
            [x / N for x in d],
            bins=n_bins_norm,
            range=(0.0, norm_max),
            density=True,
            alpha=0.5,
            color=color,
            label=f"{r['label']} (n={len(d)})",
        )

    # Vertical dashed line on both panels marking the KNN candidate cap.
    # (Also drawn when the radius kind is active so radius-only comparisons
    # still show a candidate-cap reference line at --two_opt_knn_k.)
    ax_abs.axvline(
        opts.two_opt_knn_k,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=f"knn k={opts.two_opt_knn_k}",
    )
    ax_norm.axvline(
        opts.two_opt_knn_k / N,
        color="gray",
        linestyle="--",
        alpha=0.6,
        label=f"knn k={opts.two_opt_knn_k}",
    )

    ax_abs.set_xlabel("min(|i-j|, N-|i-j|)  (short-arc hop distance)")
    ax_abs.set_ylabel("density")
    ax_abs.set_title("Accepted swap distances (short-arc)")
    ax_abs.grid(True, alpha=0.3)
    ax_abs.legend(loc="best", fontsize=9)

    ax_norm.set_xlabel("min(|i-j|, N-|i-j|) / N  (normalised to tour length)")
    ax_norm.set_ylabel("density")
    ax_norm.set_title("Accepted swap distances (normalised)")
    ax_norm.grid(True, alpha=0.3)
    ax_norm.legend(loc="best", fontsize=9)

    fig.suptitle(
        f"GLOP 2-opt swap-distance histogram — tsp{opts.problem_size}, "
        f"width={opts.width}, lens={opts.revision_lens}, "
        f"iters={opts.revision_iters}, 2opt_iters={opts.two_opt_iters}, "
        f"kind={opts.two_opt_kind}, knn_k={opts.two_opt_knn_k}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    os.makedirs(opts.out_dir, exist_ok=True)
    # Stash runs on opts so _build_tag can compute has_knn without threading
    # another parameter through this stack.
    opts._runs_for_tag = runs
    tag = _build_tag(opts)
    out_path = os.path.join(opts.out_dir, f"twoopt_swap_dists_{tag}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def print_table(runs):
    # Pick a non-OOM baseline; fall back to the first run if even baseline OOMed.
    base_avg = None
    for r in runs:
        if r["label"] == "baseline" and not r.get("skipped"):
            base_avg = r["avg"]
            break
    if base_avg is None or (isinstance(base_avg, float) and math.isnan(base_avg)):
        for r in runs:
            if not r.get("skipped") and not math.isnan(r["avg"]):
                base_avg = r["avg"]
                break
    print("\n================= Final performance =================")
    header = (
        f"{'mode':<14}{'kind':<6}{'avg cost':>12}{'best cost':>12}"
        f"{'impr% vs base':>16}{'time (s)':>12}"
    )
    print(header)
    print("-" * len(header))
    # Infer kind from the mode label if not already on the record.
    for r in runs:
        if "two_opt_kind" not in r:
            if r["label"].startswith("knn"):
                r["two_opt_kind"] = "knn"
            elif r["label"].startswith("range"):
                r["two_opt_kind"] = "range_radius"
            elif r["label"].startswith("sampling"):
                r["two_opt_kind"] = "sampling_radius"
            elif r["label"].startswith("decomp"):
                r["two_opt_kind"] = "decomp"
            elif r["label"].startswith("radius"):
                r["two_opt_kind"] = "radius"
            else:
                r["two_opt_kind"] = "full"
    for r in runs:
        if r.get("skipped") or math.isnan(r["avg"]):
            # OOM rows: render placeholders so the table stays aligned.
            print(
                f"{r['label']:<14}{r['two_opt_kind']:<6}{'OOM':>12}"
                f"{'OOM':>12}{'-':>16}{'-':>12}"
            )
            continue
        impr = (
            100.0 * (base_avg - r["avg"]) / base_avg
            if base_avg and not math.isnan(base_avg)
            else 0.0
        )
        print(
            f"{r['label']:<14}{r['two_opt_kind']:<6}{r['avg']:>12.4f}"
            f"{r['best']:>12.4f}{impr:>15.2f}%{r['duration']:>12.2f}"
        )
    print("=====================================================")


def main():
    opts = build_base_opts()
    print("using device:", opts.device)
    print("dataset:", opts.path)
    assert os.path.exists(opts.path), f"dataset not found: {opts.path}"

    revisers = load_revisers(opts)

    t0 = time.time()
    # Use the OOM-safe wrapper so a single mode running out of memory does
    # not abort the whole comparison. Other exception types still propagate.
    modes = make_modes(opts)
    runs = [
        _safe_run_mode(
            label,
            use2,
            mode,
            kind,
            knn_k,
            radius,
            r_min,
            r_max,
            decomp_r,
            sampling_base,
            sampling_r,
            opts,
            revisers,
        )
        for (
            label,
            use2,
            mode,
            kind,
            knn_k,
            radius,
            r_min,
            r_max,
            decomp_r,
            sampling_base,
            sampling_r,
        ) in modes
    ]
    skipped = [r["label"] for r in runs if r.get("skipped")]
    if skipped:
        print(f'\n[NOTE] Skipped modes due to OOM: {", ".join(skipped)}', flush=True)
    print_table(runs)
    out_path = plot_comparison(runs, opts)
    print(f"\n=== Comparison figure saved to: {out_path} ===")
    swap_path = plot_swap_distance_histograms(runs, opts)
    if swap_path is not None:
        print(f"=== Swap-distance histogram saved to: {swap_path} ===")
    print(f"=== Total wall-clock: {time.time() - t0:.2f}s ===")


if __name__ == "__main__":
    main()
