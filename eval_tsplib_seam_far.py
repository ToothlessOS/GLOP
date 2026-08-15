"""Evaluate seam_far 2-opt on the heterogeneous TSPLib benchmark set.

The TSPLib set (``data/tsp/tsplib49.pkl``) contains 49 instances of
varying sizes (~100 to ~85k nodes), so we cannot reuse the standard
``main._eval_dataset`` pipeline directly — it assumes a single fixed
``problem_size`` and a uniform-shape DataLoader batch. Instead, this
script walks each instance one at a time (batch_size=1) and selects
the revisor subset / revision_lens / revision_iters based on the
instance's actual size, mirroring the size-bucketed logic that
``eval_2opt_seam_far.py`` and ``eval_tsplib.py`` use at the top of
their inner loop.

It runs two configurations on every TSPLib instance and reports a
side-by-side comparison:

  1. baseline           — GLOP only (no 2-opt)
  2. seam_far_2opt_m_100 — GLOP + decomp_sampling 2-opt
                            (--two_opt_kind=decomp_sampling,
                             base=50, seam_radius=20, candidate_r=100)
                            applied per revisor iteration.

Outputs:

  * a printed summary table (per-instance cost + mean cost + impr% vs.
    baseline + wall-clock)
  * a per-instance CSV at ``results/tsplib_seam_far_<tag>.csv``
    (``instance_idx, size, baseline_cost, seam_far_cost``) for cross-checking
    against LKH3 / known optima
  * a single-panel bar chart at ``results/tsplib_seam_far_<tag>.png``
    showing the mean cost across all evaluated instances per mode.

Example:

  # CPU smoke run on the first 4 TSPLib instances:
  python eval_tsplib_seam_far.py --val_size 4 --no_cuda --no_progress_bar \\
      --decode_strategy greedy

  # Full 49-instance run on GPU:
  python eval_tsplib_seam_far.py --decode_strategy greedy
"""

import argparse
import copy
import math
import os
import pickle
import time

import numpy as np
import torch

from utils import load_model
from utils.functions import reconnect
from utils.insertion import random_insertion_parallel


# ---------------------------------------------------------------------------
# Comparison modes.
# Each row is a 14-tuple matching eval_2opt_seam_far.py's `make_modes` schema:
#
#   0  label
#   1  use_2opt                  (bool)
#   2  two_opt_mode              ('final' | 'per_iter')
#   3  two_opt_kind              ('full' | 'knn' | 'radius' |
#                                 'range_radius' | 'sampling_radius' |
#                                 'hop_radius' | 'decomp' |
#                                 'decomp_sampling')
#   4  two_opt_knn_k             (int|None)
#   5  two_opt_radius            (int|None)
#   6  two_opt_radius_min        (int|None)
#   7  two_opt_radius_max        (int|None)
#   8  two_opt_decomp_radius     (int|None)
#   9  two_opt_sampling_base     (int|None)
#  10  two_opt_sampling_r        (int|None)
#  11  two_opt_decomp_sampling_base         (int|None)
#  12  two_opt_decomp_sampling_seam_radius  (int|None)
#  13  two_opt_decomp_sampling_candidate_r  (int|None)
# ---------------------------------------------------------------------------
MODES = [
    # GLOP-only baseline; no 2-opt.
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
        None,
        None,
        None,
        None,
    ),
    # GLOP + decomp_sampling 2-opt applied per revisor iteration.
    # Same parameters as eval_2opt_seam_far.py:347-362 ("seam_far_2opt_m_100").
    (
        "seam_far_2opt_m_100",
        True,
        "per_iter",
        "decomp_sampling",
        20,
        None,
        None,
        None,
        None,
        None,
        None,
        50,
        20,
        100,
    ),
]


def _load_tsplib_pickle(path):
    """Load a TSPLib-style pkl as a list of (N_i, 2) float tensors.

    The TSPLib pickle at ``data/tsp/tsplib49.pkl`` is a Python list of
    raw coordinate arrays (one per instance) — NOT a torch.save dump —
    so we cannot use ``torch.load``. Each instance is converted to a
    ``FloatTensor`` and stacked into a list. Instances are heterogeneous
    in size; callers must iterate with batch_size=1.
    """
    assert os.path.splitext(path)[1] == ".pkl", f"expected .pkl, got {path}"
    with open(path, "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, list), f"TSPLib pkl must be a list, got {type(data)}"
    instances = [torch.FloatTensor(np.asarray(row, dtype=np.float32)) for row in data]
    sizes = [inst.shape[0] for inst in instances]
    print(
        f"[TSPLib] loaded {len(instances)} instances from {path} "
        f"(min={min(sizes)}, max={max(sizes)}, median={int(np.median(sizes))})"
    )
    return instances


def pick_size_bucket(problem_size, full_revisers):
    """Return (subset_revisers, revision_lens, revision_iters, no_aug) for `problem_size`.

    Ported verbatim from the basic ``eval_2opt_seam_far.py`` /
    ``eval_tsplib.py`` size-bucket logic. The caller passes in the full
    revisers list ordered by descending size, e.g. ``[100, 50, 20, 10]``
    (each entry is a torch.nn.Module trained for that subproblem length).

    Buckets:
        N  <  20  → AssertionError
        20 ≤ N <  50 → [20, 10]
        50 ≤ N < 100 → [50, 20]
        100 ≤ N < 150 → [100, 50, 20]
        N ≥ 150  → [100, 50, 20]

    The (width=4) augmentation in main.py is automatically disabled
    above N=100 (matches the original logic), but we leave the actual
    augmentation toggle to the caller via ``no_aug``.
    """
    assert problem_size >= 20, (
        f"TSPLib instance has problem_size={problem_size} < 20; "
        f"the smallest revisor is trained for L=10, so this script "
        f"requires N >= 20."
    )
    if problem_size < 50:
        slice_ = slice(2, None)
        revision_lens = [20, 10]
        revision_iters = [10, 5]
        no_aug = True
    elif problem_size < 100:
        slice_ = slice(1, 3)
        revision_lens = [50, 20]
        revision_iters = [10, 5]
        no_aug = True
    elif problem_size < 150:
        slice_ = slice(0, 3)
        revision_lens = [100, 50, 20]
        revision_iters = [10, 5, 5]
        no_aug = True
    else:
        slice_ = slice(0, 3)
        revision_lens = [100, 50, 20]
        revision_iters = [10, 10, 5]
        no_aug = False

    subset_revisers = full_revisers[slice_]
    assert len(subset_revisers) == len(revision_lens), (
        f"revisers list missing a required entry for problem_size={problem_size}. "
        f"Pass --revision_lens that covers all sizes {revision_lens}."
    )
    return subset_revisers, revision_lens, revision_iters, no_aug


def build_base_opts():
    """Argparse mirror of ``eval_2opt_seam_far.py:build_base_opts``.

    Defaults are tuned for TSPLib: ``--path`` points to the
    heterogeneous pickle, ``--val_size=49`` uses every instance,
    ``--eval_batch_size=1`` (mandatory for mixed-size batches),
    ``--width=4`` runs diversified insertion warm-starts.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--problem_type", type=str, default="tsp", choices=["tsp"])
    p.add_argument(
        "--path",
        type=str,
        default="data/tsp/tsplib49.pkl",
        help="TSPLib pickle path (heterogeneous list of instances).",
    )
    p.add_argument(
        "--val_size",
        type=int,
        default=49,
        help="Number of TSPLib instances to evaluate (capped to len(instances)).",
    )
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Must be 1 for heterogeneous TSPLib; per-instance batches.",
    )
    p.add_argument(
        "--width",
        type=int,
        default=4,
        help="Number of diversified warm-start tours per instance.",
    )
    p.add_argument(
        "--revision_lens",
        nargs="+",
        type=int,
        default=[100, 50, 20, 10],
        help="Reviser sizes to load, in descending order. pick_size_bucket "
        "slices this list per instance.",
    )
    p.add_argument(
        "--revision_iters",
        nargs="+",
        type=int,
        default=[10, 10, 5, 5],
        help="Iteration counts per revisor (overridden per-bucket).",
    )
    p.add_argument(
        "--decode_strategy",
        type=str,
        default="greedy",
        help="'greedy' recommended for deterministic comparisons.",
    )
    p.add_argument(
        "--two_opt_iters",
        type=int,
        default=30,
        help="Max 2-opt sweeps per invocation.",
    )
    p.add_argument(
        "--two_opt_kind",
        type=str,
        default="decomp_sampling",
        choices=[
            "full",
            "knn",
            "radius",
            "range_radius",
            "sampling_radius",
            "hop_radius",
            "decomp",
            "decomp_sampling",
        ],
        help="2-opt algorithm variant; the MODES table passes its own "
        "value per row, so the CLI default only matters when running "
        "a single mode.",
    )
    p.add_argument("--two_opt_knn_k", type=int, default=20)
    p.add_argument("--two_opt_radius", type=int, default=None)
    p.add_argument("--two_opt_radius_min", type=int, default=2)
    p.add_argument("--two_opt_radius_max", type=int, default=None)
    p.add_argument("--two_opt_sampling_base", type=int, default=2)
    p.add_argument("--two_opt_sampling_r", type=int, default=None)
    p.add_argument("--two_opt_hop_base", type=int, default=2)
    p.add_argument("--two_opt_hop_h", type=int, default=None)
    p.add_argument("--two_opt_decomp_radius", type=int, default=None)
    p.add_argument(
        "--two_opt_decomp_sampling_base",
        type=int,
        default=50,
        help="Default for decomp_sampling 2-opt (matches MODES row 2).",
    )
    p.add_argument(
        "--two_opt_decomp_sampling_seam_radius",
        type=int,
        default=20,
        help="Default for decomp_sampling 2-opt (matches MODES row 2).",
    )
    p.add_argument(
        "--two_opt_decomp_sampling_candidate_r",
        type=int,
        default=100,
        help="Default for decomp_sampling 2-opt (matches MODES row 2).",
    )
    p.add_argument("--two_opt_debug", action="store_true")
    p.add_argument("--no_aug", action="store_true")
    p.add_argument("--no_prune", action="store_true")
    p.add_argument("--no_progress_bar", action="store_true")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_dir", type=str, default="results")

    opts = p.parse_args()

    # Fields that `reconnect` / `LCP_TSP` may consult; not exposed on the CLI.
    opts.n_subset = 1
    opts.n_partition = 1
    opts.ckpt_path = ""
    opts.tsp_aug = False  # always off — heterogeneous TSPLib skips aug

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.device = torch.device(
        f"cuda:{opts.device_id}" if use_cuda else "cpu"
    )

    assert opts.eval_batch_size == 1, (
        "Heterogeneous TSPLib requires batch_size=1 (instances have "
        "different sizes). Pass --eval_batch_size=1."
    )
    assert os.path.exists(opts.path), f"dataset not found: {opts.path}"
    return opts


def load_revisers(opts):
    """Load every revisor in ``opts.revision_lens`` onto ``opts.device``.

    Mirrors ``eval_2opt_seam_far.py:load_revisers`` (572-581) — the
    revisers must be loaded in descending size order so that
    ``pick_size_bucket`` can slice them by index.
    """
    revisers = []
    for reviser_size in opts.revision_lens:
        reviser_path = f"pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt"
        reviser, _ = load_model(reviser_path, is_local=True)
        reviser.to(opts.device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)
        revisers.append(reviser)
    return revisers


# Cost function reused by reconnect().
from utils.functions import load_problem  # noqa: E402  (after argparse defs)


def _get_cost_func():
    problem = load_problem("tsp")
    return lambda input, pi: problem.get_costs(input, pi, return_local=True)


def eval_one_instance(instance_tensor, opts, device, revisers, mode_tuple):
    """Run one (instance, mode) pair; return (warm_cost, revised_cost)."""
    (
        _label,
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
        ds_base,
        ds_seam_r,
        ds_cand_r,
    ) = mode_tuple

    # Per-instance, per-mode opts (don't mutate the caller's copy).
    inst_opts = copy.copy(opts)
    inst_opts.use_2opt = use_2opt
    inst_opts.two_opt_mode = two_opt_mode
    inst_opts.two_opt_kind = two_opt_kind
    inst_opts.two_opt_knn_k = two_opt_knn_k
    inst_opts.two_opt_radius = two_opt_radius
    inst_opts.two_opt_radius_min = two_opt_radius_min
    inst_opts.two_opt_radius_max = two_opt_radius_max
    inst_opts.two_opt_decomp_radius = two_opt_decomp_radius
    inst_opts.two_opt_sampling_base = two_opt_sampling_base
    inst_opts.two_opt_sampling_r = two_opt_sampling_r
    inst_opts.two_opt_hop_base = getattr(opts, "two_opt_hop_base", 2)
    inst_opts.two_opt_hop_h = getattr(opts, "two_opt_hop_h", None)
    inst_opts.two_opt_decomp_sampling_base = ds_base
    inst_opts.two_opt_decomp_sampling_seam_radius = ds_seam_r
    inst_opts.two_opt_decomp_sampling_candidate_r = ds_cand_r
    inst_opts.two_opt_debug = getattr(opts, "two_opt_debug", False)
    # Per-iter swap recording is not relevant for this script.
    inst_opts.two_opt_swap_sink = []
    inst_opts.record_two_opt_swaps = False

    problem_size = instance_tensor.shape[0]

    # (1, N, 2) batch — random_insertion_parallel expects a stacked batch.
    batch = instance_tensor.unsqueeze(0).to(device)
    # Drop the revisors that are larger than this instance (revisors are
    # size-specific — a 100-L revisor cannot process an L=60 instance).
    _revisers, revision_lens, revision_iters, no_aug = pick_size_bucket(
        problem_size, revisers
    )
    inst_opts.revision_lens = revision_lens
    inst_opts.revision_iters = revision_iters
    inst_opts.no_aug = no_aug

    width = opts.width
    # Match the original eval_2opt_seam_far.py / eval_tsplib.py behaviour:
    # for small instances (N <= 100) the warm-start width is reduced so the
    # outer 4x augmentation below brings it back to a comparable breadth.
    if problem_size <= 100 and width >= 4:
        width //= 4

    orders = [torch.randperm(problem_size) for _ in range(width)]
    # random_insertion_parallel expects a single order per call (the C
    # lib iterates cities along axis 0 and reorders via that one order).
    # Each call returns a (1, N) permutation; stack into (width, N).
    # The C library consumes numpy arrays, so the cities tensor is moved
    # to CPU first; the resulting pi_batch is therefore CPU-resident —
    # move it back to `device` before gathering into the (GPU) batch_rep.
    pi_list = [random_insertion_parallel(batch.cpu(), order) for order in orders]
    pi_batch = torch.tensor(
        np.asarray(pi_list, dtype=np.int64)
    ).reshape(width, problem_size)
    pi_batch = pi_batch.to(device, non_blocking=True)

    if batch.shape[0] == 1:
        batch_rep = batch.repeat(width, 1, 1)
    else:
        batch_rep = batch

    seed = batch_rep.gather(1, pi_batch.unsqueeze(-1).expand_as(batch_rep))
    seed = seed.to(device)

    cost_ori = (
        (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1)
        + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
    )
    # best-of-width warm-start cost for this single instance
    warm_cost = cost_ori.min().item()

    # Optional tsp_aug (mirrors main.py:151-155); disabled by default
    # for heterogeneous TSPLib, but kept for parity with the standard
    # small-N behaviour.
    if problem_size <= 100 and inst_opts.tsp_aug:
        seed2 = torch.cat((1 - seed[:, :, [0]], seed[:, :, [1]]), dim=2)
        seed3 = torch.cat((seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
        seed4 = torch.cat((1 - seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
        seed = torch.cat((seed, seed2, seed3, seed4), dim=0)

    get_cost_func = _get_cost_func()
    _, costs_revised = reconnect(
        get_cost_func=get_cost_func,
        batch=seed,
        opts=inst_opts,
        revisers=_revisers,
        stats_list=None,
    )
    # costs_revised shape: (width or width*4_aug, eval_batch_size)
    # After eval_batch_size=1, pick best across width restarts.
    if costs_revised.dim() == 0:
        revised_cost = costs_revised.item()
    else:
        revised_cost = costs_revised.reshape(-1).min().item()
    return warm_cost, revised_cost


def run_mode(mode_tuple, base_opts, revisers, instances):
    """Run one mode across all instances; return dict with per-instance costs."""
    (
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
        ds_base,
        ds_seam_r,
        ds_cand_r,
    ) = mode_tuple

    print(
        f"\n===================== running mode: {label} "
        f"(use_2opt={use_2opt}, mode={two_opt_mode}, "
        f"kind={two_opt_kind}, knn_k={two_opt_knn_k}, "
        f"ds_base={ds_base}, ds_seam_r={ds_seam_r}, ds_cand_r={ds_cand_r}) "
        f"====================="
    )

    n = len(instances)
    warm_costs = np.zeros(n, dtype=np.float64)
    revised_costs = np.zeros(n, dtype=np.float64)
    sizes = np.zeros(n, dtype=np.int64)
    t_start = time.time()
    for i, inst in enumerate(instances):
        # Reproducible per-instance warm-start: reset the RNG before
        # random_insertion_parallel so different modes share the same
        # warm-start tours.
        torch.manual_seed(base_opts.seed + i)
        np.random.seed(base_opts.seed + i)
        warm, revised = eval_one_instance(
            inst, base_opts, base_opts.device, revisers, mode_tuple
        )
        warm_costs[i] = warm
        revised_costs[i] = revised
        sizes[i] = inst.shape[0]
        if not base_opts.no_progress_bar:
            print(
                f"  [{i+1}/{n}] N={inst.shape[0]:>5d}  "
                f"warm={warm:>10.4f}  revised={revised:>10.4f}  "
                f"Δ={(revised - warm):+.4f}"
            )
    duration = time.time() - t_start

    return {
        "label": label,
        "two_opt_kind": two_opt_kind,
        "use_2opt": use_2opt,
        "two_opt_mode": two_opt_mode,
        "warm_costs": warm_costs,
        "revised_costs": revised_costs,
        "sizes": sizes,
        "avg": float(revised_costs.mean()),
        "best": float(revised_costs.min()),
        "duration": duration,
    }


def print_tsplib_table(runs, n_instances):
    """Print a per-mode summary table with impr% vs. baseline."""
    base = next((r for r in runs if r["label"] == "baseline"), runs[0])
    base_avg = base["avg"]
    print("\n================= TSPLib seam_far comparison =================")
    header = (
        f"{'mode':<24}{'kind':<18}{'avg cost':>14}{'best cost':>14}"
        f"{'impr% vs base':>16}{'time (s)':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        impr = (
            100.0 * (base_avg - r["avg"]) / base_avg
            if base_avg > 0
            else 0.0
        )
        print(
            f"{r['label']:<24}{r['two_opt_kind']:<18}{r['avg']:>14.4f}"
            f"{r['best']:>14.4f}{impr:>15.2f}%{r['duration']:>12.2f}"
        )
    print("=" * len(header))
    print(f"Evaluated {n_instances} TSPLib instances")


def plot_tsplib_comparison(runs, opts, sizes):
    """Single-panel bar chart: mean cost across all instances per mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline": "#7f7f7f",
        "seam_far_2opt_m_100": "#41b6c4",
    }
    labels = [r["label"] for r in runs]
    avgs = [r["avg"] for r in runs]
    bests = [r["best"] for r in runs]
    base_avg = next((r["avg"] for r in runs if r["label"] == "baseline"), avgs[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        avgs,
        color=[colors.get(l, _random_color(l)) for l in labels],
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean final tour cost (across TSPLib instances)")
    ax.set_title("seam_far 2-opt on TSPLib")
    ax.grid(True, axis="y", alpha=0.3)
    if avgs:
        lo = min(avgs) * 0.995
        hi = max(avgs) * 1.005
        ax.set_ylim(lo, hi)
    for xi, (_, avg, best) in enumerate(zip(bars, avgs, bests)):
        impr = 100.0 * (base_avg - avg) / base_avg if base_avg else 0.0
        ax.annotate(
            f"avg {avg:.3f}\nbest {best:.3f}\n({impr:+.2f}%)",
            xy=(xi, avg),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    size_summary = (
        f"N range [{int(sizes.min())}..{int(sizes.max())}], "
        f"median={int(np.median(sizes))}"
    )
    fig.suptitle(
        f"GLOP + seam_far on TSPLib — {len(sizes)} instances, {size_summary}, "
        f"width={opts.width}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    os.makedirs(opts.out_dir, exist_ok=True)
    tag = (
        f"tsplib_n{len(sizes)}_w{opts.width}"
        f"_lens{'-'.join(map(str, opts.revision_lens))}"
    )
    out_path = os.path.join(opts.out_dir, f"tsplib_seam_far_{tag}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _random_color(label):
    """Deterministic fallback colour for unknown mode names."""
    import hashlib
    return "#" + hashlib.md5(label.encode("utf-8")).hexdigest()[:6]


def save_csv(runs, opts, out_path):
    """Per-instance cost comparison as CSV for downstream analysis."""
    import csv

    base_run = next((r for r in runs if r["label"] == "baseline"), runs[0])
    seam_run = next((r for r in runs if r["label"] != "baseline"), None)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["instance_idx", "size"]
        for r in runs:
            header += [f"{r['label']}_cost", f"{r['label']}_warm"]
        if seam_run is not None:
            header.append("seam_far_impr%")
        w.writerow(header)
        for i in range(len(base_run["sizes"])):
            row = [i, int(base_run["sizes"][i])]
            for r in runs:
                row.append(f"{r['revised_costs'][i]:.6f}")
                row.append(f"{r['warm_costs'][i]:.6f}")
            if seam_run is not None:
                base_c = base_run["revised_costs"][i]
                seam_c = seam_run["revised_costs"][i]
                if base_c > 0:
                    impr = 100.0 * (base_c - seam_c) / base_c
                else:
                    impr = 0.0
                row.append(f"{impr:.4f}")
            w.writerow(row)


def main():
    opts = build_base_opts()
    print("using device:", opts.device)
    print("dataset:", opts.path)

    instances = _load_tsplib_pickle(opts.path)
    if opts.val_size < len(instances):
        instances = instances[: opts.val_size]
        print(f"[TSPLib] truncating to --val_size={opts.val_size} instances")
    n_instances = len(instances)
    sizes = np.array([inst.shape[0] for inst in instances], dtype=np.int64)

    revisers = load_revisers(opts)

    t0 = time.time()
    runs = [run_mode(mode, opts, revisers, instances) for mode in MODES]
    print(f"\n=== Total wall-clock: {time.time() - t0:.2f}s ===")

    print_tsplib_table(runs, n_instances)

    out_png = plot_tsplib_comparison(runs, opts, sizes)
    print(f"\n=== Comparison figure saved to: {out_png} ===")

    tag = (
        f"tsplib_n{n_instances}_w{opts.width}"
        f"_lens{'-'.join(map(str, opts.revision_lens))}"
    )
    csv_path = os.path.join(opts.out_dir, f"tsplib_seam_far_{tag}.csv")
    save_csv(runs, opts, csv_path)
    print(f"=== Per-instance CSV saved to: {csv_path} ===")


if __name__ == "__main__":
    main()