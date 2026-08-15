"""Evaluate and compare GLOP with seam-far 2-opt on the TSP1000 benchmark set.

This is a TSP1000-flavoured counterpart to ``eval_2opt_seam_far.py`` that
consumes the five distribution datasets shipped under ``data/tsp1000/``:

    uniform, clustered1, clustered2, explosion, implosion

Each ``data/tsp1000/tsp1000_<dist>.txt`` file holds 200 TSP instances of
1000 cities each. The matching LKH-3 reference tours live in
``data/tsp1000_sol/tsp1000_<dist>`` (one tour + cost + LKH runtime per
line). The data is bridged into GLOP's existing pipeline via
``utils.tsp1000_data.resolve_tsp1000_pkl``, which materialises a
GLOP-compatible ``.pkl`` on disk (cached) and lets the unmodified
``main._eval_dataset`` flow consume it.

The comparison table mirrors the original script's two-row setup:

  1. baseline         — GLOP only (no 2-opt)
  2. seam_far_2opt_m_100 — GLOP with decomp_sampling 2-opt
                            (base=50, seam_radius=20, candidate_r=100)
                            applied after every revisor iteration.

Outputs:
  * printed summary table with avg/best cost, time, AND gap% vs the
    LKH-3 optimum for each row (per distribution).
  * ``results/twoopt_compare_tsp1000_<dist>_<tag>.png`` — three panels:
      - per-iteration convergence curve, one line per mode
      - final-cost bar chart with % improvement vs. baseline
      - gap-to-optimum % bar chart for the same modes
  * (optional) ``results/twoopt_swap_dists_<tag>.png`` when
    ``--record_two_opt_swaps`` is set.
  * (when ``--tsp1000_distribution all``) one final combined table at
    the end aggregating per-instance gap% over all 5 distributions
    (~``val_size * 5`` instances, macro-instance mean).

Reviser checkpoints are loaded from ``--reviser_root`` (default
``pretrained/Reviser-stage2``). Pass ``--reviser_root
pretrained/Reviser-stage2-tuned`` to evaluate the finetuned revisers
and A/B against the stage2 defaults. The checkpoint-path layout must be
``<root>/reviser_<size>/epoch-299.pt`` for each ``--revision_lens`` entry.

Example:
  python eval_2opt_seam_far_tsp1000.py \\
      --tsp1000_distribution uniform --val_size 4 --eval_batch_size 1 \\
      --revision_lens 100 50 20 --revision_iters 2 1 1 \\
      --width 1 --no_aug --no_progress_bar

  # Sweep all five distributions (one PNG + table per distribution,
  # plus a combined table at the end):
  python eval_2opt_seam_far_tsp1000.py --tsp1000_distribution all

  # Compare finetuned revisers vs the stage2 defaults (run twice,
  # diff the combined tables / PNG suffixes):
  python eval_2opt_seam_far_tsp1000.py --tsp1000_distribution all \\
      --reviser_root pretrained/Reviser-stage2
  python eval_2opt_seam_far_tsp1000.py --tsp1000_distribution all \\
      --reviser_root pretrained/Reviser-stage2-tuned
"""

import argparse
import copy
import io
import math
import os
import re
import time
from contextlib import redirect_stdout

import numpy as np
import torch

from utils import load_model

# Reuse the exact eval machinery from main.py so this script tracks the pipeline.
from main import _eval_dataset, _aggregate_solver_curve

# TSP1000-specific data plumbing (parsing, pkl cache, gap-to-optimum).
from utils.tsp1000_data import (
    DISTRIBUTIONS,
    DEFAULT_DATA_DIR,
    DEFAULT_SOL_DIR,
    DEFAULT_CACHE_DIR,
    load_tsp1000_optimal_costs,
    resolve_tsp1000_pkl,
    summarise_distribution_costs,
    gap_to_optimum_pct,
)


def _is_oom_error(exc):
    """True if `exc` looks like a CUDA / CPU out-of-memory error.

    Used by ``_safe_run_mode`` to decide whether to swallow the exception
    (skip the mode, keep evaluating the rest) or let it propagate.
    """
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "cuda out of memory" in msg or "out of memory" in msg:
            return True
    if isinstance(exc, MemoryError):
        return True
    return False


# Deterministic per-label fallback colors for plot dicts.
_RANDOM_MARKERS = ("o", "s", "D", "v", "^", "P", "X", "p", "h", "*", "d", "<", ">")


def _random_color(label):
    """Stable hex color for an unknown plot label."""
    import hashlib

    return "#" + hashlib.md5(label.encode("utf-8")).hexdigest()[:6]


def _random_marker(label):
    """Stable matplotlib marker for an unknown plot label."""
    import hashlib

    return _RANDOM_MARKERS[
        int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % len(_RANDOM_MARKERS)
    ]


def _sanitize_reviser_root(root):
    """Stable filename slug for a reviser-checkpoint root directory.

    Strips a leading ``Reviser-`` prefix and lowercases alphanumerics so
    different checkpoint trees don't collide on PNG filenames:

        pretrained/Reviser-stage2          -> stage2
        pretrained/Reviser-stage2-tuned    -> stage2_tuned
        foo/bar                            -> bar
    """
    leaf = os.path.basename(os.path.normpath(root))
    leaf = re.sub(r"^Reviser-", "", leaf)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", leaf).strip("_").lower()
    return slug or "root"


# ---------------------------------------------------------------------------
# 2-opt dispatcher / per-mode runner — copied verbatim from eval_2opt_seam_far.py
# so the comparison surface is identical.
# ---------------------------------------------------------------------------


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
    two_opt_decomp_sampling_base,
    two_opt_decomp_sampling_seam_radius,
    two_opt_decomp_sampling_candidate_r,
    base_opts,
    revisers,
):
    """Run one configuration; on OOM, return a sentinel row instead of raising."""
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
            two_opt_decomp_sampling_base,
            two_opt_decomp_sampling_seam_radius,
            two_opt_decomp_sampling_candidate_r,
            base_opts,
            revisers,
        )
    except BaseException as exc:
        if not _is_oom_error(exc):
            raise
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
            f"samp_base={two_opt_sampling_base}, samp_r={two_opt_sampling_r}, "
            f"decomp_samp_base={two_opt_decomp_sampling_base}, "
            f"decomp_samp_seam_r={two_opt_decomp_sampling_seam_radius}, "
            f"decomp_samp_cand_r={two_opt_decomp_sampling_candidate_r}); "
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
            "two_opt_decomp_sampling_base": two_opt_decomp_sampling_base,
            "two_opt_decomp_sampling_seam_radius": two_opt_decomp_sampling_seam_radius,
            "two_opt_decomp_sampling_candidate_r": two_opt_decomp_sampling_candidate_r,
            "avg": float("nan"),
            "best": float("nan"),
            "duration": float("nan"),
            "curve": {},
            "swap_dists": [],
            "skipped": True,
            "error": str(exc),
            "costs_pred": None,
            "tours": None,
        }


# ---------------------------------------------------------------------------
# Comparison table — identical two-row setup to eval_2opt_seam_far.py.
# ---------------------------------------------------------------------------


def make_modes(opts):
    """Two-row comparison: baseline + seam_far_2opt_m_100.

    Mirrors ``eval_2opt_seam_far.py:make_modes`` so the pipeline runs the
    same configurations across both scripts.
    """
    N = max(2, opts.problem_size // 10)
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
            None,
            None,
            None,
            None,
        ),
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_base_opts():
    """Argparse for TSP1000 + 2-opt comparison.

    Defaults override the parent script for N=1000 (see plan):
      * problem_size=1000, val_size=200, eval_batch_size=1
      * revision_lens=[100, 50, 20], revision_iters=[10, 10, 5]
      * width=1 (RAM-safe default for TSP1000)
      * no_aug defaults to True (irrelevant for N>100 anyway)
      * --path is left empty; the new --tsp1000_distribution flag
        resolves to a pkl via utils.tsp1000_data.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--problem_size", type=int, default=1000)
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
        help="Override the auto-resolved TSP1000 pkl path. Leave empty to "
        "let --tsp1000_distribution resolve one.",
    )
    p.add_argument("--val_size", type=int, default=200)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--width", type=int, default=1)
    p.add_argument("--revision_lens", nargs="+", type=int, default=[100, 50, 20])
    p.add_argument("--revision_iters", nargs="+", type=int, default=[10, 10, 5])
    p.add_argument(
        "--decode_strategy",
        type=str,
        default="greedy",
        help="'greedy' recommended for a deterministic comparison",
    )
    # TSP1000-specific dataset selection.
    p.add_argument(
        "--tsp1000_distribution",
        type=str,
        default="uniform",
        choices=list(DISTRIBUTIONS) + ["all"],
        help="Which TSP1000 distribution to evaluate. 'all' sweeps all five.",
    )
    p.add_argument(
        "--tsp1000_data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory holding tsp1000_<dist>.txt instance files.",
    )
    p.add_argument(
        "--tsp1000_sol_dir",
        type=str,
        default=DEFAULT_SOL_DIR,
        help="Directory holding tsp1000_<dist> solution files.",
    )
    p.add_argument(
        "--tsp1000_cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Where to write the converted .pkl caches.",
    )
    p.add_argument(
        "--no_pkl_cache",
        action="store_true",
        help="Bypass the on-disk pkl cache (uses a temp file instead).",
    )
    p.add_argument(
        "--reviser_root",
        type=str,
        default="pretrained/Reviser-stage2",
        help="Directory root for reviser checkpoints. The script expects "
        "the layout <root>/reviser_<size>/epoch-299.pt for each "
        "--revision_lens entry. Pass e.g. "
        "'pretrained/Reviser-stage2-tuned' to evaluate the finetuned "
        "revisers and A/B against the stage2 defaults.",
    )
    # 2-opt knobs — same as eval_2opt_seam_far.py.
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
        help="2-opt algorithm variant (see eval_2opt_seam_far.py for full docs).",
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
    p.add_argument("--two_opt_decomp_sampling_base", type=int, default=2)
    p.add_argument(
        "--two_opt_decomp_sampling_seam_radius", type=int, default=None
    )
    p.add_argument(
        "--two_opt_decomp_sampling_candidate_r", type=int, default=None
    )
    p.add_argument(
        "--two_opt_debug",
        action="store_true",
        help="Print per-sweep 2-opt phase timings to stdout.",
    )
    p.add_argument(
        "--record_two_opt_swaps",
        action="store_true",
        help="Record |i_star - j_star| for every accepted 2-opt move and "
        "emit a swap-distance histogram PNG.",
    )
    p.add_argument("--no_aug", action="store_true", default=True)
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
    return opts


def load_revisers(opts):
    revisers = []
    for reviser_size in opts.revision_lens:
        reviser_path = f"{opts.reviser_root}/reviser_{reviser_size}/epoch-299.pt"
        reviser, _ = load_model(reviser_path, is_local=True)
        reviser.to(opts.device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)
        revisers.append(reviser)
    return revisers


# ---------------------------------------------------------------------------
# Per-mode runner — same as eval_2opt_seam_far.py, plus it stashes the
# per-instance predicted-cost tensor on the run dict so the gap-to-optimum
# column can be populated after the mode loop.
# ---------------------------------------------------------------------------


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
    two_opt_decomp_sampling_base,
    two_opt_decomp_sampling_seam_radius,
    two_opt_decomp_sampling_candidate_r,
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
    opts.two_opt_decomp_sampling_base = two_opt_decomp_sampling_base
    opts.two_opt_decomp_sampling_seam_radius = two_opt_decomp_sampling_seam_radius
    opts.two_opt_decomp_sampling_candidate_r = two_opt_decomp_sampling_candidate_r
    if two_opt_kind == "hop_radius":
        opts.two_opt_hop_base = two_opt_sampling_base
        opts.two_opt_hop_h = two_opt_sampling_r
    else:
        opts.two_opt_hop_base = getattr(base_opts, "two_opt_hop_base", 2)
        opts.two_opt_hop_h = getattr(base_opts, "two_opt_hop_h", None)
    opts.two_opt_debug = getattr(base_opts, "two_opt_debug", False)
    opts.record_two_opt_swaps = getattr(base_opts, "record_two_opt_swaps", False)
    swap_dists: list = []
    opts.two_opt_swap_sink = swap_dists

    torch.manual_seed(base_opts.seed)
    np.random.seed(base_opts.seed)

    print(
        f"\n===================== running mode: {label} "
        f"(use_2opt={use_2opt}, mode={two_opt_mode}, "
        f"kind={two_opt_kind}, knn_k={two_opt_knn_k}, "
        f"radius={two_opt_radius}, "
        f"r_min={two_opt_radius_min}, r_max={two_opt_radius_max}, "
        f"decomp_r={two_opt_decomp_radius}, "
        f"samp_base={two_opt_sampling_base}, samp_r={two_opt_sampling_r}, "
        f"hop_base={opts.two_opt_hop_base}, hop_h={opts.two_opt_hop_h}, "
        f"decomp_samp_base={two_opt_decomp_sampling_base}, "
        f"decomp_samp_seam_r={two_opt_decomp_sampling_seam_radius}, "
        f"decomp_samp_cand_r={two_opt_decomp_sampling_candidate_r}) "
        f"====================="
    )
    results, duration, all_stats = _eval_dataset(opts.path, opts, opts.device, revisers)

    costs = final_costs(results)
    # Also stash the per-instance cost tensor for gap-to-optimum reporting.
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
        "two_opt_hop_base": opts.two_opt_hop_base,
        "two_opt_hop_h": opts.two_opt_hop_h,
        "two_opt_decomp_sampling_base": two_opt_decomp_sampling_base,
        "two_opt_decomp_sampling_seam_radius": two_opt_decomp_sampling_seam_radius,
        "two_opt_decomp_sampling_candidate_r": two_opt_decomp_sampling_candidate_r,
        "avg": costs.mean().item(),
        "best": costs.min().item(),
        "duration": duration,
        "curve": _aggregate_solver_curve(all_stats),
        "swap_dists": swap_dists,
        "costs_pred": costs.detach().cpu(),
    }


# ---------------------------------------------------------------------------
# Per-distribution driver
# ---------------------------------------------------------------------------


def _resolve_distribution_paths(opts, distribution):
    """Resolve the pkl path + sol-costs for one distribution.

    If ``opts.path`` was explicitly set, that wins (escape hatch). Otherwise
    build the pkl from the tsp1000_*.txt file.
    """
    if opts.path:
        pkl_path = opts.path
    else:
        pkl_path = resolve_tsp1000_pkl(
            distribution,
            data_dir=opts.tsp1000_data_dir,
            cache_dir=opts.tsp1000_cache_dir,
            num_samples=opts.val_size,
            offset=0,
            use_cache=not opts.no_pkl_cache,
        )
    sol_path = os.path.join(opts.tsp1000_sol_dir, f"tsp1000_{distribution}")
    return pkl_path, sol_path


def _eval_one_distribution(distribution, opts, revisers):
    """Run all comparison modes for one distribution; return (runs, summary).

    ``summary`` carries the per-instance predicted-cost tensors from each
    mode plus the optimal-cost vector from ``data/tsp1000_sol/``, ready for
    gap-to-optimum computation and plot annotation.
    """
    pkl_path, sol_path = _resolve_distribution_paths(opts, distribution)
    assert os.path.exists(pkl_path), f"dataset not found: {pkl_path}"
    assert os.path.exists(sol_path), f"solution file not found: {sol_path}"

    # Point the reused pipeline at the resolved pkl.
    opts_dist = copy.deepcopy(opts)
    opts_dist.path = pkl_path
    print(f"\n----- TSP1000 distribution = {distribution} -----")
    print(f"dataset (pkl): {pkl_path}")
    print(f"solution (opt): {sol_path}")

    modes = make_modes(opts_dist)
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
            decomp_sampling_base,
            decomp_sampling_seam_radius,
            decomp_sampling_candidate_r,
            opts_dist,
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
            decomp_sampling_base,
            decomp_sampling_seam_radius,
            decomp_sampling_candidate_r,
        ) in modes
    ]
    skipped = [r["label"] for r in runs if r.get("skipped")]
    if skipped:
        print(f'[NOTE] Skipped modes due to OOM: {", ".join(skipped)}', flush=True)

    # Optimal costs for the same val_size slice.
    costs_opt, _runtimes = load_tsp1000_optimal_costs(
        sol_path, num_samples=opts.val_size, offset=0
    )
    if costs_opt.numel() != opts.val_size:
        # Defensive: shouldn't happen given the slice validation upstream.
        print(
            f"[WARN] sol file has {costs_opt.numel()} lines but val_size={opts.val_size}; "
            "truncating optimal-cost vector to match."
        )
        costs_opt = costs_opt[: opts.val_size]

    summary = {
        "distribution": distribution,
        "costs_opt": costs_opt,
        "n": int(costs_opt.numel()),
    }
    return runs, summary, opts_dist


# ---------------------------------------------------------------------------
# Console reporting (gap-aware)
# ---------------------------------------------------------------------------


def _print_table_with_gap(runs, summary):
    """Print the comparison table with gap% columns populated from summary."""
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

    costs_opt = summary["costs_opt"]
    print("\n================= Final performance =================")
    header = (
        f"{'mode':<22}{'kind':<18}{'avg cost':>12}{'best cost':>12}"
        f"{'impr% vs base':>16}{'gap% avg':>11}{'gap% best':>11}{'time (s)':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in runs:
        if "two_opt_kind" not in r:
            if r["label"].startswith("knn"):
                r["two_opt_kind"] = "knn"
            elif r["label"].startswith("range"):
                r["two_opt_kind"] = "range_radius"
            elif r["label"].startswith("sampling"):
                r["two_opt_kind"] = "sampling_radius"
            elif r["label"].startswith("hop"):
                r["two_opt_kind"] = "hop_radius"
            elif r["label"].startswith("decomp"):
                r["two_opt_kind"] = "decomp"
            elif r["label"].startswith("seam_far"):
                r["two_opt_kind"] = "decomp_sampling"
            elif r["label"].startswith("radius"):
                r["two_opt_kind"] = "radius"
            else:
                r["two_opt_kind"] = "full"
    for r in runs:
        if r.get("skipped") or math.isnan(r["avg"]):
            print(
                f"{r['label']:<22}{r['two_opt_kind']:<18}{'OOM':>12}"
                f"{'OOM':>12}{'-':>16}{'-':>11}{'-':>11}{'-':>12}"
            )
            continue
        impr = (
            100.0 * (base_avg - r["avg"]) / base_avg
            if base_avg and not math.isnan(base_avg)
            else 0.0
        )
        # Gap-to-optimum requires the per-instance predicted-cost tensor.
        costs_pred = r.get("costs_pred")
        if costs_pred is None or costs_pred.numel() != costs_opt.numel():
            gap_avg_str = "-"
            gap_best_str = "-"
        else:
            gap = gap_to_optimum_pct(costs_pred, costs_opt)
            gap_avg_str = f"{gap.mean().item():.2f}%"
            # "best" = the smallest predicted cost in the slice, vs opt of
            # the same instance — i.e. gap at the oracle instance.
            best_idx = int(costs_pred.argmin().item())
            gap_best_val = gap[best_idx].item()
            gap_best_str = f"{gap_best_val:.2f}%"
        print(
            f"{r['label']:<22}{r['two_opt_kind']:<18}{r['avg']:>12.4f}"
            f"{r['best']:>12.4f}{impr:>15.2f}%{gap_avg_str:>11}"
            f"{gap_best_str:>11}{r['duration']:>12.2f}"
        )
    print("=====================================================")


def _print_combined_table(all_runs_per_mode, all_costs_opt, opts, modes):
    """Print one combined table aggregating across all TSP1000 distributions.

    Macro-instance mean: per-instance predicted costs are concatenated
    across all distributions, then ``gap_to_optimum_pct`` is computed
    once on the resulting vector. Every instance is weighted equally.

    Args:
        all_runs_per_mode: list of length ``len(modes)``; each entry is a
            list of length ``len(distributions)`` holding either a
            per-instance cost ``Tensor`` or ``None`` for OOM-skipped rows.
        all_costs_opt: list of length ``len(distributions)``; each entry
            is a per-instance optimal-cost ``Tensor``.
        opts: parsed args (only ``opts.val_size`` is consulted).
        modes: the comparison rows (same as ``make_modes(opts)``) — used
            for the per-row ``label`` and ``two_opt_kind``.
    """
    n_dists = len(all_costs_opt)

    # Resolve base-mode combined avg for the impr% column.
    base_avg = None
    for i, mode in enumerate(modes):
        if mode[0] == "baseline":
            tensors = [t for t in all_runs_per_mode[i] if t is not None]
            if tensors:
                base_avg = torch.cat(tensors, dim=0).mean().item()
            break
    if base_avg is None or (isinstance(base_avg, float) and math.isnan(base_avg)):
        for i in range(len(modes)):
            tensors = [t for t in all_runs_per_mode[i] if t is not None]
            if tensors:
                base_avg = torch.cat(tensors, dim=0).mean().item()
                break

    print(
        "\n================= Combined across distributions "
        f"({n_dists} datasets) ================="
    )
    header = (
        f"{'mode':<22}{'kind':<18}{'avg cost':>12}{'best cost':>12}"
        f"{'impr% vs base':>16}{'gap% avg':>11}{'gap% best':>11}{'time (s)':>12}"
    )
    print(header)
    print("-" * len(header))
    for i, mode in enumerate(modes):
        label, _, _, kind = mode[0], mode[1], mode[2], mode[3]
        tensors = all_runs_per_mode[i]
        kept = [t for t in tensors if t is not None]
        skipped_count = sum(1 for t in tensors if t is None)
        if not kept:
            print(
                f"{label:<22}{kind:<18}{'OOM':>12}{'OOM':>12}"
                f"{'-':>16}{'-':>11}{'-':>11}{'-':>12}"
            )
            continue
        pred = torch.cat(kept, dim=0)
        # Match the OOM-skipped distributions' optimal vectors only.
        kept_opt = [
            all_costs_opt[k] for k, t in enumerate(tensors) if t is not None
        ]
        opt = torch.cat(kept_opt, dim=0)
        gap = gap_to_optimum_pct(pred, opt)
        avg = pred.mean().item()
        best = pred.min().item()
        impr = (
            100.0 * (base_avg - avg) / base_avg
            if base_avg and not (isinstance(base_avg, float) and math.isnan(base_avg))
            else 0.0
        )
        gap_avg = gap.mean().item()
        gap_best = gap[pred.argmin()].item()
        # Sum per-dist durations where the mode wasn't OOM-skipped.
        # ``opts._combined_durations[i]`` is a dict {dist_idx: duration}
        # populated by main() (see ``all_durations_per_mode``).
        durations_i = getattr(opts, "_combined_durations", [{}] * len(modes))[i]
        duration = sum(
            durations_i[k] for k in range(n_dists) if tensors[k] is not None
        )
        suffix = (
            f"  (n={pred.numel()}"
            + (f", skipped={skipped_count})" if skipped_count else ")")
        )
        print(
            f"{label:<22}{kind:<18}{avg:>12.4f}{best:>12.4f}"
            f"{impr:>15.2f}%{gap_avg:>10.2f}%{gap_best:>10.2f}%"
            f"{duration:>12.2f}{suffix}"
        )
    print("=====================================================")


# ---------------------------------------------------------------------------
# Plotting (3 panels: convergence + final-cost + gap-to-optimum)
# ---------------------------------------------------------------------------


def flatten_curve(curve):
    """Concatenate per-layer 'best' series into one global convergence sequence."""
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


def plot_comparison(runs, opts, summary):
    """Three-panel comparison figure with the gap-to-optimum added.

    Panel 1: per-iteration convergence curve (same as parent script).
    Panel 2: final-cost bar chart with % improvement vs. baseline.
    Panel 3: gap-to-optimum % bar chart (new).
    """
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
        "hop_per_iter_default": "#8c6d31",
        "hop_per_iter_wide": "#b29966",
        "decomp_per_iter": "#dbdb8d",
        "seam_far_per_iter": "#c7e9b4",
        "seam_far_2opt_k_2": "#c7e9b4",
        "seam_far_2opt_k_5": "#7fcdbb",
        "seam_far_2opt_m_50": "#41b6c4",
        "seam_far_2opt_m_100": "#1d91c0",
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
        "hop_per_iter_default": "P",
        "hop_per_iter_wide": "X",
        "decomp_per_iter": "d",
        "seam_far_per_iter": "P",
        "seam_far_2opt_k_2": "P",
        "seam_far_2opt_k_5": "X",
        "seam_far_2opt_m_50": "p",
        "seam_far_2opt_m_100": "*",
    }

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [2, 1, 1]}
    )

    # --- Panel 1: per-iteration convergence ---
    all_boundaries = []
    for r in runs:
        xs, ys, boundaries = flatten_curve(r["curve"])
        if not xs:
            continue
        all_boundaries = boundaries
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

    # --- Panel 2: final-cost bar chart ---
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
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Final mean tour cost")
    ax2.set_title("Final performance")
    ax2.grid(True, axis="y", alpha=0.3)
    finite_avgs = [a for a in avgs if not (isinstance(a, float) and math.isnan(a))]
    if finite_avgs:
        lo = min(finite_avgs) * 0.995
        hi = max(finite_avgs) * 1.005
        ax2.set_ylim(lo, hi)
    for xi, (_, avg, best) in enumerate(zip(bars, avgs, bests)):
        if math.isnan(avg):
            ax2.annotate(
                "OOM", xy=(xi, 0), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8,
            )
            continue
        impr = 100.0 * (base_avg - avg) / base_avg if base_avg else 0.0
        ax2.annotate(
            f"{avg:.3f}\n({impr:+.2f}%)",
            xy=(xi, avg), xytext=(0, 3),
            textcoords="offset points", ha="center", va="bottom", fontsize=8,
        )

    # --- Panel 3: gap-to-optimum % bar chart ---
    costs_opt = summary["costs_opt"]
    gap_avgs = []
    gap_bests = []
    for r in runs:
        cp = r.get("costs_pred")
        if r.get("skipped") or cp is None or cp.numel() != costs_opt.numel():
            gap_avgs.append(float("nan"))
            gap_bests.append(float("nan"))
            continue
        g = gap_to_optimum_pct(cp, costs_opt)
        gap_avgs.append(g.mean().item())
        gap_bests.append(g[g.argmin()].item())  # gap at the oracle instance
    bars3 = ax3.bar(
        x,
        gap_avgs,
        color=[colors.get(l, _random_color(l)) for l in labels],
        alpha=0.85,
    )
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=15, ha="right")
    ax3.set_ylabel("Gap to LKH optimum (%)")
    ax3.set_title("Gap-to-optimum")
    ax3.grid(True, axis="y", alpha=0.3)
    finite_gaps = [g for g in gap_avgs if not (isinstance(g, float) and math.isnan(g))]
    if finite_gaps:
        lo3 = min(min(finite_gaps), 0.0) * 1.05
        hi3 = max(finite_gaps) * 1.10
        if hi3 == 0.0:
            hi3 = 1.0
        ax3.set_ylim(lo3, hi3)
    for xi, (g_avg, g_best) in enumerate(zip(gap_avgs, gap_bests)):
        if math.isnan(g_avg):
            ax3.annotate(
                "OOM", xy=(xi, 0), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8,
            )
            continue
        ax3.annotate(
            f"{g_avg:.2f}%\n(min inst {g_best:.2f}%)",
            xy=(xi, g_avg), xytext=(0, 3),
            textcoords="offset points", ha="center", va="bottom", fontsize=8,
        )

    fig.suptitle(
        f"GLOP + 2-opt on TSP1000 ({summary['distribution']}) — "
        f"N={opts.problem_size}, width={opts.width}, "
        f"val_size={opts.val_size}, lens={opts.revision_lens}, "
        f"iters={opts.revision_iters}, 2opt_iters={opts.two_opt_iters}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    os.makedirs(opts.out_dir, exist_ok=True)
    slug = _sanitize_reviser_root(opts.reviser_root)
    tag = (
        f"tsp1000_{summary['distribution']}_w{opts.width}"
        f"_lens{'-'.join(map(str, opts.revision_lens))}"
        f"_iters{'-'.join(map(str, opts.revision_iters))}"
        f"_2opt{opts.two_opt_iters}"
        f"_root{slug}"
    )
    out_path = os.path.join(opts.out_dir, f"twoopt_compare_{tag}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Swap-distance histogram (kept identical to parent script's behaviour).
# ---------------------------------------------------------------------------


def _build_tag(opts):
    runs_for_tag = getattr(opts, "_runs_for_tag", [])
    slug = _sanitize_reviser_root(opts.reviser_root)
    return (
        f"tsp1000_{getattr(opts, '_tsp1000_distribution_for_tag', 'all')}"
        f"_w{opts.width}"
        f"_lens{'-'.join(map(str, opts.revision_lens))}"
        f"_iters{'-'.join(map(str, opts.revision_iters))}"
        f"_2opt{opts.two_opt_iters}"
        f"_root{slug}"
    )


def plot_swap_distance_histograms(runs, opts):
    """Plot the swap-distance distribution per mode as a sibling PNG."""
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
        "hop_per_iter_default": "#8c6d31",
        "hop_per_iter_wide": "#b29966",
        "decomp_per_iter": "#dbdb8d",
        "seam_far_per_iter": "#c7e9b4",
        "seam_far_2opt_k_2": "#c7e9b4",
        "seam_far_2opt_k_5": "#7fcdbb",
        "seam_far_2opt_m_50": "#41b6c4",
        "seam_far_2opt_m_100": "#1d91c0",
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
            d, bins=n_bins_abs, range=(0, abs_max), density=True,
            alpha=0.5, color=color, label=f"{r['label']} (n={len(d)})",
        )
        ax_norm.hist(
            [x / N for x in d], bins=n_bins_norm, range=(0.0, norm_max),
            density=True, alpha=0.5, color=color, label=f"{r['label']} (n={len(d)})",
        )

    ax_abs.axvline(
        opts.two_opt_knn_k, color="gray", linestyle="--", alpha=0.6,
        label=f"knn k={opts.two_opt_knn_k}",
    )
    ax_norm.axvline(
        opts.two_opt_knn_k / N, color="gray", linestyle="--", alpha=0.6,
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
    opts._runs_for_tag = runs
    tag = _build_tag(opts)
    out_path = os.path.join(opts.out_dir, f"twoopt_swap_dists_{tag}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    opts = build_base_opts()
    print("using device:", opts.device)
    print("tsp1000 distribution selector:", opts.tsp1000_distribution)
    print("reviser root:", opts.reviser_root)
    print(
        "result buffering: enabled (per-distribution tables are "
        "deferred to the end-of-run summary so early results don't "
        "scroll out of view; progress logs stream inline as usual)"
    )

    revisers = load_revisers(opts)

    t0 = time.time()
    distributions = (
        list(DISTRIBUTIONS) if opts.tsp1000_distribution == "all"
        else [opts.tsp1000_distribution]
    )
    modes = make_modes(opts)

    # Accumulators for the combined table (only used when --tsp1000_distribution all).
    all_runs_per_mode = [[] for _ in modes]
    all_costs_opt = []
    all_durations_per_mode = [dict() for _ in modes]
    # Per-distribution table capture — only the dense result tables are
    # deferred; progress logs (distribution banner, mode banners, figure
    # save paths, OOM messages) still stream through to the terminal.
    dist_table_outputs: list[tuple[str, str]] = []

    for dist in distributions:
        runs, summary, opts_dist = _eval_one_distribution(dist, opts, revisers)
        # Capture only the table for end-of-run display.
        table_buf = io.StringIO()
        with redirect_stdout(table_buf):
            _print_table_with_gap(runs, summary)
        dist_table_outputs.append((dist, table_buf.getvalue()))
        # Stash distribution tag for the swap-distance histogram filename.
        opts_dist._tsp1000_distribution_for_tag = dist
        out_path = plot_comparison(runs, opts_dist, summary)
        print(f"\n=== Comparison figure saved to: {out_path} ===")
        swap_path = plot_swap_distance_histograms(runs, opts_dist)
        if swap_path is not None:
            print(f"=== Swap-distance histogram saved to: {swap_path} ===")

        if opts.tsp1000_distribution == "all":
            all_costs_opt.append(summary["costs_opt"])
            for i, r in enumerate(runs):
                all_runs_per_mode[i].append(r.get("costs_pred"))
                all_durations_per_mode[i][len(all_costs_opt) - 1] = r.get("duration", 0.0)

    total_seconds = time.time() - t0

    # ----- End-of-run summary: dump all per-distribution tables, then
    # the combined table, then the wall-clock. Tables only — progress
    # logs already streamed during the loop above.
    print(
        "\n\n================= End-of-run summary "
        f"({len(distributions)} distribution(s)) ================="
    )
    for dist, table_text in dist_table_outputs:
        print(f"\n----- TSP1000 distribution: {dist} -----")
        # Strip a trailing newline so the banner doesn't get doubled.
        print(table_text.rstrip())
    if opts.tsp1000_distribution == "all" and any(all_runs_per_mode):
        opts._combined_durations = all_durations_per_mode
        _print_combined_table(all_runs_per_mode, all_costs_opt, opts, modes)
    print(f"\n=== Total wall-clock: {total_seconds:.2f}s ===")


if __name__ == "__main__":
    main()