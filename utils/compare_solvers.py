#!/usr/bin/env python3
"""Head-to-head compare GLOP and LKH-3 on shared TSP instances.

Reads:
  - GLOP tours + coords dumped by ``main.py --save_tours <dir>`` (a
    ``meta.json`` index plus one ``*_<stage>.pt`` per stage);
  - LKH-3 outputs from
    ``LKH3_eval/run_diagnosis.py --keep-tempfiles --workdir <lkh_workdir>``
    (a ``summary.json`` plus per-instance ``*.tour`` files).

For every matched ``(dataset, instance_idx)`` pair this script
recovers the GLOP permutation by exact-match lookup against the
canonical reference coordinates, parses the LKH tour from the
``.tour`` file, builds each side's closed-loop edge set in original
node-index space, computes the per-instance cost gap, and emits:

    <out_dir>/gap_summary.png              bar chart of mean gap per stage
    <out_dir>/<dataset>/all_gaps.png       per-dataset gap histogram
    <out_dir>/<dataset>/instance_NNNN_diff.png  side-by-side diffs (top-K)
    <out_dir>/index.txt                    per-instance summary table

The DAG of work:
    _load_glop_dump      +   _load_lkh_records      -->  join on (dataset, idx)
                                          |
                                          v
                              _compute_records_for_each_match
                                          |
                  +-----------------------+-----------------------+
                  v                       v                       v
          _write_summary_bar   _write_per_dataset_hist     _write_instance_diffs

All plotting uses the ``Agg`` matplotlib backend (consistent with
``utils/diagnosis.plot_tsp_tours``) so the script is headless-friendly.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

# Make ``utils`` importable whether this file is run as
# ``python utils/compare_solvers.py`` or ``python -m utils.compare_solvers``.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.lkh import calc_tsp_length, read_tsplib  # noqa: E402
from utils.diagnosis import (  # noqa: E402
    _recover_permutation_from_coords,
    _tour_edge_set,
    plot_glop_lkh_diff,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------


def _load_glop_dump(glop_dump_dir, stage):
    """Load GLOP tour dump + meta.json. Returns the chosen stage by
    name, or picks the most-processed stage if ``stage == "auto"``.

    Returns a dict with keys:
        meta      : full meta.json contents
        coords    : (M, N, 2) torch.Tensor — canonical reference coords
        tours     : (M, N, 2) torch.Tensor — GLOP tours in traversal order
        stage     : str — stage actually loaded
        tag       : str — original meta.json tag
    """
    meta_path = os.path.join(glop_dump_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"No meta.json in {glop_dump_dir}. Did you run "
            f"`python main.py --save_tours {glop_dump_dir}` first?"
        )
    with open(meta_path, "r") as f:
        meta = json.load(f)
    stages = meta.get("stages", {})
    if stage == "auto":
        for cand in ("postrev", "postfix", "raw"):
            if cand in stages:
                stage = cand
                break
        else:
            raise ValueError(
                f"meta.json has no known stages; available: {list(stages)}"
            )
    if stage not in stages:
        available = list(stages)
        raise ValueError(
            f"Stage '{stage}' not in meta.json (available: {available}). "
            "Did you forget to pass --save_tours when running main.py?"
        )
    snap_file = stages[stage]["file"]
    snap_path = os.path.join(glop_dump_dir, snap_file)
    payload = torch.load(snap_path, weights_only=False)
    coords = payload.get("coords")
    tours = payload["tours"]
    return {
        "meta": meta,
        "coords": coords,
        "tours": tours,
        "stage": stage,
        "tag": meta.get("tag", os.path.splitext(snap_file)[0]),
    }


def _load_lkh_records(lkh_workdir, dataset_filter=None):
    """Read ``summary.json`` + per-instance ``*.tour`` files in
    ``lkh_workdir``. Returns a dict keyed by ``(dataset, instance_idx)``:

        key -> {dataset, instance_idx, pi, cost_lkh}

    ``summary.json`` (as written by ``run_diagnosis.py:626-647``) is a
    single dict with an ``instances`` list of per-instance records.
    Each record carries ``dataset`` (basename) and ``instance`` (0-based
    index) — not ``instance_idx``. The tour file lives at
    ``<lkh_workdir>/<dataset>/<name>.tour`` where
    ``<name> = f"{dataset}_inst{instance:04d}"``. The .tour files exist
    on disk only when ``--keep-tempfiles`` was passed; if any expected
    tour is missing, that instance is silently skipped.
    """
    summary_path = os.path.join(lkh_workdir, "summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f"No summary.json in {lkh_workdir}. Did you run "
            f"`python LKH3_eval/run_diagnosis.py --keep-tempfiles "
            f"--workdir {lkh_workdir}` first?"
        )
    with open(summary_path, "r") as f:
        summary = json.load(f)

    # Handle both shapes gracefully: a flat {"instances": [...]} as the
    # current run_diagnosis.py writes, OR a future dataset-keyed shape.
    if isinstance(summary, dict) and "instances" in summary:
        records = summary["instances"]
    elif isinstance(summary, dict):
        # Backwards-compat fallback: dataset-name -> [records]
        records = []
        for _, recs in summary.items():
            if isinstance(recs, list):
                records.extend(recs)
    elif isinstance(summary, list):
        records = summary
    else:
        raise ValueError(
            f"Unrecognized summary.json shape: top-level type "
            f"{type(summary).__name__}"
        )

    by_key = {}
    for rec in records:
        if not rec.get("ok", False):
            continue
        ds = rec.get("dataset")
        inst = rec.get("instance")
        if ds is None or inst is None:
            continue
        if dataset_filter and ds not in dataset_filter:
            continue
        # Tour file path: <lkh_workdir>/<dataset>/<name>.tour, where
        # name = f"{dataset}_inst{instance:04d}" by run_diagnosis.py:240.
        name = rec.get("name") or f"{ds}_inst{int(inst):04d}"
        tour_path = os.path.join(lkh_workdir, ds, f"{name}.tour")
        if not os.path.isfile(tour_path):
            # --keep-tempfiles was off, or a different naming convention
            # was used — skip this instance.
            continue
        try:
            pi = read_tsplib(tour_path)  # 0-indexed permutation
        except Exception:
            continue
        cost_lkh = rec.get("lkh_cost_unscaled")
        if cost_lkh is None:
            cost_lkh = rec.get("our_cost")
        if cost_lkh is None or not np.isfinite(cost_lkh) or cost_lkh <= 0:
            continue
        by_key[(ds, int(inst))] = {
            "dataset": ds,
            "instance_idx": int(inst),
            "pi": pi,
            "cost_lkh": float(cost_lkh),
        }
    return by_key


# ---------------------------------------------------------------------
# Joining + per-record analysis
# ---------------------------------------------------------------------


def _match_records(glop_dump, lkh_records):
    """For each LKH instance, find the GLOP tour by instance index and
    produce a per-record dict ready for plotting.

    Returns a list of dicts:
        {dataset, instance_idx, coords (N,2), pi_glop, pi_lkh,
         edges_glop, edges_lkh, shared, glop_only, lkh_only,
         cost_glop, cost_lkh, gap_pct}
    """
    coords_full = glop_dump["coords"]
    tours_full = glop_dump["tours"]
    if coords_full is None:
        raise ValueError(
            "GLOP dump has no 'coords' tensor. The .pt payload is "
            "missing the canonical reference frame; rerun main.py "
            "with --save_tours to regenerate it."
        )

    records = []
    for key, lkh in lkh_records.items():
        base, inst_idx = key
        if inst_idx >= tours_full.shape[0]:
            # LKH run solved more instances than GLOP produced (e.g.
            # different val_size). Skip the unmatched ones silently.
            continue
        coords = coords_full[inst_idx].cpu().numpy()
        glop_tour = tours_full[inst_idx].cpu().numpy()

        # Recover GLOP permutation via exact-match lookup. This is O(N^2)
        # per instance but tiny in absolute terms (e.g. ~1ms for N=500).
        pi_glop = _recover_permutation_from_coords(glop_tour, coords)
        pi_lkh = np.asarray(lkh["pi"], dtype=np.int64)
        if pi_lkh.shape[0] != coords.shape[0]:
            # LKH instance size mismatch — skip rather than crash.
            continue

        edges_glop = _tour_edge_set(pi_glop)
        edges_lkh = _tour_edge_set(pi_lkh)
        shared = edges_glop & edges_lkh
        glop_only = edges_glop - shared
        lkh_only = edges_lkh - shared

        cost_glop = float(calc_tsp_length(coords, pi_glop.tolist()))
        cost_lkh = float(lkh["cost_lkh"])
        if not np.isfinite(cost_lkh) or cost_lkh <= 0:
            continue
        gap_pct = 100.0 * (cost_glop - cost_lkh) / cost_lkh

        records.append(
            {
                "dataset": base,
                "instance_idx": inst_idx,
                "coords": coords,
                "pi_glop": pi_glop,
                "pi_lkh": pi_lkh,
                "edges_glop": edges_glop,
                "edges_lkh": edges_lkh,
                "shared": shared,
                "glop_only": glop_only,
                "lkh_only": lkh_only,
                "cost_glop": cost_glop,
                "cost_lkh": cost_lkh,
                "gap_pct": gap_pct,
                "n_shared": len(shared),
                "n_glop_only": len(glop_only),
                "n_lkh_only": len(lkh_only),
            }
        )
    return records


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def _dataset_label_sort(label):
    """Sort key for dataset names so tsp{size}_test cluster by size."""
    m = re.search(r"(\d+)", label)
    return (0, int(m.group(1)), label) if m else (1, 0, label)


def _write_summary_bar(records, out_path):
    """Mean gap ± std per dataset, single bar chart. Records from all
    stages are aggregated; when only one stage is loaded the bar is
    unambiguous.
    """
    by_ds = defaultdict(list)
    for r in records:
        by_ds[r["dataset"]].append(r["gap_pct"])
    names = sorted(by_ds, key=_dataset_label_sort)
    if not names:
        return
    means = [np.mean(by_ds[n]) for n in names]
    stds = [np.std(by_ds[n]) for n in names]
    counts = [len(by_ds[n]) for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(names))
    bars = ax.bar(xs, means, yerr=stds, capsize=4, color="C0", alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n(n={c})" for n, c in zip(names, counts)],
                       fontsize=9)
    ax.set_ylabel("GLOP vs LKH-3 gap (%)")
    ax.set_title("Mean optimality gap per dataset\n(negative = GLOP better)")
    for x, m in zip(xs, means):
        ax.annotate(
            f"{m:+.2f}%",
            xy=(x, m),
            xytext=(0, 3 if m >= 0 else -10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_edge_suboptimality_by_length(records, out_path, n_bins=20,
                                        stage_tag=""):
    """Histogram of GLOP suboptimality rate as a function of edge length.

    For every matched instance we look at every edge in the GLOP tour,
    measure its Euclidean length, and flag it as "suboptimal" iff LKH-3
    did *not* also pick that edge. The rates are then pooled across all
    instances and binned by length. The hypothesis behind this plot is
    that GLOP's loss concentrates on long-range bridges: short edges
    tend to be local greedy choices that the LKH solver agrees with,
    while the long edges are the ones where global reasoning matters
    and a learned policy makes a sub-optimal trade.

    Output is a single PNG with twin y-axes:

      - bars (red, left axis):  P(LKH did NOT pick | length in bin)
      - line+markers (blue, right axis):  total GLOP-tour-edge count per
        length bin — sample size, so a high rate on a low-count bin is
        visibly less reliable than one on a populous bin.

    Also writes a sibling ``edge_suboptimality.csv`` next to the PNG
    with the binned counts and rates, so a downstream notebook can
    re-plot or test the trend without re-running the pipeline.
    """
    if not records:
        return

    lengths = []
    is_sub = []
    for r in records:
        coords = r["coords"]
        shared = r["shared"]
        glop_edges = r["edges_glop"]
        for (i, j) in glop_edges:
            d = float(np.linalg.norm(coords[i] - coords[j]))
            lengths.append(d)
            is_sub.append(0 if (i, j) in shared else 1)

    lengths = np.asarray(lengths, dtype=np.float64)
    is_sub = np.asarray(is_sub, dtype=np.int64)
    if lengths.size == 0:
        return

    # Choose bin range. For unit-square TSP instances the longest edge
    # is at most sqrt(2) ≈ 1.414, but instances can push beyond (LKH
    # has produced a few). Clip at the 99.5th percentile so a handful
    # of pathological outliers do not squash the meaningful range into
    # one thick bar at the right edge.
    p995 = float(np.percentile(lengths, 99.5))
    bin_max = max(float(lengths.max()), p995, 0.1)
    bin_edges = np.linspace(0.0, bin_max, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    counts_glop = np.zeros(n_bins, dtype=np.int64)
    counts_sub = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = (lengths >= bin_edges[b]) & (lengths < bin_edges[b + 1])
        counts_glop[b] = int(mask.sum())
        counts_sub[b] = int(is_sub[mask].sum())

    rates = np.where(counts_glop > 0, counts_sub / counts_glop, np.nan)

    # --- CSV sidecar for downstream analysis ---
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    with open(csv_path, "w") as f:
        f.write(
            "length_lo,length_hi,length_center,n_glop_edges,"
            "n_suboptimal_edges,suboptimal_rate\n"
        )
        for b in range(n_bins):
            f.write(
                f"{bin_edges[b]:.6f},{bin_edges[b + 1]:.6f},"
                f"{centers[b]:.6f},{int(counts_glop[b])},"
                f"{int(counts_sub[b])},"
                f"{('nan' if np.isnan(rates[b]) else f'{rates[b]:.6f}')}\n"
            )

    # --- PNG ---
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    width = (bin_max / n_bins) * 0.92
    ax1.bar(
        centers, rates,
        width=width,
        color="C3", alpha=0.85, edgecolor="white",
        label="P(LKH did NOT pick | GLOP did)",
    )
    ax1.set_xlabel("edge length (in unit-square coordinates)")
    ax1.set_ylabel(
        "GLOP-suboptimal rate per length bin",
        color="C3",
    )
    ax1.tick_params(axis="y", labelcolor="C3")
    ymax = float(np.nanmax(rates)) if np.any(~np.isnan(rates)) else 0.05
    # Add headroom + floor so an empty/near-zero bin doesn't flatten the bars.
    ax1.set_ylim(0.0, max(0.05, ymax * 1.25))
    # Overall baseline so the trend reads against a reference.
    overall = float(counts_sub.sum()) / max(int(counts_glop.sum()), 1)
    ax1.axhline(overall, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
    ax1.annotate(
        f"overall={overall:.3f}",
        xy=(bin_max, overall),
        xytext=(-5, 4),
        textcoords="offset points",
        ha="right",
        color="gray", fontsize=8,
    )
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        centers, counts_glop,
        "o-", color="C0", linewidth=1.5, markersize=4, alpha=0.85,
        label="# GLOP-tour edges in bin",
    )
    ax2.set_ylabel("# GLOP-tour edges", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")

    suptitle_tag = f"  [{stage_tag}]" if stage_tag else ""
    fig.suptitle(
        f"GLOP suboptimality vs edge length{suptitle_tag}\n"
        f"{len(records)} instances, "
        f"{int(counts_glop.sum())} GLOP-tour edges analyzed",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_per_dataset_histogram(records, dataset, out_path, stage_tag):
    """Distribution of per-instance gaps for a single dataset."""
    gaps = [r["gap_pct"] for r in records if r["dataset"] == dataset]
    if not gaps:
        return
    gaps = np.array(gaps)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gaps, bins=min(20, max(5, len(gaps) // 3)), color="C0", alpha=0.85,
            edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=0.6)
    mean = float(np.mean(gaps))
    ax.axvline(mean, color="C3", linestyle="--", linewidth=1.2)
    ax.annotate(
        f"mean={mean:+.2f}%",
        xy=(mean, ax.get_ylim()[1] * 0.9),
        xytext=(5, 0),
        textcoords="offset points",
        color="C3",
        fontsize=9,
    )
    ax.set_xlabel("gap (%) = 100 × (cost_glop − cost_lkh) / cost_lkh")
    ax.set_ylabel("instances")
    ax.set_title(
        f"{dataset}  •  GLOP vs LKH-3 gap distribution  •  stage={stage_tag}  "
        f"•  n={len(gaps)}"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_instance_diff(record, out_path, tag=None):
    """One PNG per instance: side-by-side GLOP vs LKH-3 with the
    edge-set symmetric diff highlighted. Delegates to
    ``utils.diagnosis.plot_glop_lkh_diff``.
    """
    plot_glop_lkh_diff(
        coords=record["coords"],
        glop_tour=record["pi_glop"],
        lkh_tour_or_pi=record["pi_lkh"],
        gap_pct=record["gap_pct"],
        glop_cost=record["cost_glop"],
        lkh_cost=record["cost_lkh"],
        instance_idx=record["instance_idx"],
        out_path=out_path,
        tag=tag,
    )


# ---------------------------------------------------------------------
# Index + driver
# ---------------------------------------------------------------------


def _format_table(records):
    """Plain-text tabular index for human / grep consumption."""
    lines = [
        "instance  dataset            idx     cost_glop     cost_lkh      gap%  "
        "  #common  #glop-only  #lkh-only",
        "-" * 96,
    ]
    for r in sorted(records, key=lambda x: (-x["gap_pct"], x["dataset"], x["instance_idx"])):
        lines.append(
            f"{r['dataset'] + '/' + str(r['instance_idx']):<22s}  "
            f"glop={r['cost_glop']:.4f}  lkh={r['cost_lkh']:.4f}  "
            f"gap={r['gap_pct']:+.3f}%   "
            f"common={r['n_shared']:<4d}  "
            f"glop-only={r['n_glop_only']:<4d}  "
            f"lkh-only={r['n_lkh_only']:<4d}"
        )
    return "\n".join(lines) + "\n"


def _format_summary_stats(records):
    """Aggregate stats across all matched records (single dataset block)."""
    if not records:
        return "(no matched records)\n"
    gaps = np.array([r["gap_pct"] for r in records])
    n_shared = np.array([r["n_shared"] for r in records])
    n_glop = np.array([r["n_glop_only"] for r in records])
    n_lkh = np.array([r["n_lkh_only"] for r in records])
    n_total = n_shared + n_glop  # = N edges in GLOP tour
    lines = [
        f"matched instances: {len(records)}",
        f"gap (%): mean={gaps.mean():+.3f}  std={gaps.std():.3f}  "
        f"min={gaps.min():+.3f}  max={gaps.max():+.3f}  median={np.median(gaps):+.3f}",
        f"shared edges:  mean={n_shared.mean():.1f}/{n_total.mean():.1f}  "
        f"({100.0 * n_shared.mean() / n_total.mean():.1f}%)",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Compare GLOP and LKH-3 TSP tours and render "
        "per-instance edge-diff figures and a gap summary.",
    )
    parser.add_argument(
        "--glop_dump",
        required=True,
        help="Directory written by `main.py --save_tours <DIR>`; "
        "contains meta.json + one `*_<stage>.pt` per stage.",
    )
    parser.add_argument(
        "--lkh_workdir",
        required=True,
        help="Directory written by `LKH3_eval/run_diagnosis.py "
        "--keep-tempfiles --workdir <DIR>`; contains summary.json + "
        "per-instance `*.tour` files.",
    )
    parser.add_argument(
        "--stage",
        default="auto",
        choices=["auto", "raw", "postfix", "postrev"],
        help="Which GLOP snapshot stage to compare against LKH-3 "
        "(default auto picks the most-processed available).",
    )
    parser.add_argument(
        "--out_dir",
        default="",
        help="Output directory (default: results/compare_<tag>).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="Per-dataset count of worst-gap instances to render in "
        "side-by-side diff PNGs (default 8).",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Restrict to specific dataset names (e.g. tsp500_test). "
        "Default: every dataset present in LKH summary.json.",
    )
    args = parser.parse_args()

    glop_dump = _load_glop_dump(args.glop_dump, args.stage)
    lkh_records = _load_lkh_records(args.lkh_workdir, args.datasets)
    records = _match_records(glop_dump, lkh_records)
    if not records:
        print("[compare_solvers] no matched (dataset, instance_idx) pairs "
              "between GLOP dump and LKH workdir — nothing to render.",
              file=sys.stderr)
        sys.exit(2)

    out_dir = args.out_dir or os.path.join(
        "results", f"compare_{re.sub(r'[^A-Za-z0-9._-]', '_', glop_dump['tag'])}"
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"[compare_solvers] GLOP stage = {glop_dump['stage']}  "
          f"({glop_dump['tag']})")
    print(f"[compare_solvers] matched {len(records)} instances across "
          f"{len({r['dataset'] for r in records})} datasets")
    print(_format_summary_stats(records))

    # Per-dataset subdir for histograms and per-instance diff PNGs.
    summary_bar_path = os.path.join(out_dir, "gap_summary.png")
    _write_summary_bar(records, summary_bar_path)
    print(f"[compare_solvers] wrote {summary_bar_path}")

    # Aggregate view: GLOP suboptimality rate as a function of edge
    # length. Tests the hypothesis that GLOP's loss concentrates on
    # long-range bridges that LKH-3 prefers to skip.
    edge_subopt_path = os.path.join(
        out_dir, f"edge_suboptimality_{glop_dump['stage']}.png"
    )
    _write_edge_suboptimality_by_length(
        records,
        edge_subopt_path,
        stage_tag=f"{glop_dump['stage']}/{glop_dump['tag']}",
    )
    print(
        f"[compare_solvers] wrote {edge_subopt_path} "
        f"(and {(edge_subopt_path[:-4] + '.csv') if edge_subopt_path.endswith('.png') else ''})"
    )

    by_ds = defaultdict(list)
    for r in records:
        by_ds[r["dataset"]].append(r)

    for ds, ds_records in by_ds.items():
        ds_dir = os.path.join(out_dir, ds)
        os.makedirs(ds_dir, exist_ok=True)
        hist_path = os.path.join(ds_dir, "all_gaps.png")
        _write_per_dataset_histogram(
            ds_records, ds, hist_path, stage_tag=glop_dump["stage"]
        )
        print(f"[compare_solvers] wrote {hist_path}")

        # Top-K worst gaps within this dataset.
        worst = sorted(ds_records, key=lambda x: -x["gap_pct"])[: args.top_k]
        for r in worst:
            png = os.path.join(
                ds_dir, f"instance_{r['instance_idx']:04d}_diff.png"
            )
            _write_instance_diff(
                r, png, tag=f"{ds}/{r['instance_idx']} {glop_dump['stage']}"
            )
        print(f"[compare_solvers] wrote {len(worst)} diff PNGs into {ds_dir}")

    index_path = os.path.join(out_dir, "index.txt")
    with open(index_path, "w") as f:
        f.write(
            f"GLOP tag: {glop_dump['tag']}\n"
            f"GLOP stage: {glop_dump['stage']}\n"
            f"LKH workdir: {args.lkh_workdir}\n"
            f"Top-K per dataset: {args.top_k}\n\n"
            f"=== aggregate ===\n"
            f"{_format_summary_stats(records)}\n"
            f"=== per-instance (sorted by descending gap%) ===\n"
        )
        f.write(_format_table(records))
    print(f"[compare_solvers] wrote {index_path}")


if __name__ == "__main__":
    main()
