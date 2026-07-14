"""Render the GLOP sub-TSP solver convergence curve from a JSONL log.

Reads a JSONL file produced by ``LCP_TSP`` when the ``--iter_cost_log``
flag is set on ``main.py`` (or any other caller of
:func:`utils.functions.LCP_TSP`). Groups the per-iter records by
``layer_id``, aggregates across the per-batch records that share the
same ``(layer_id, iter_id)`` (weighted by ``count``), orders the layers
by ``layer_id`` ascending (Layer 1, Layer 2, Layer 3 from left to
right), and produces a one-PNG-per-file figure with one subplot per
revisor layer.

Conventions:

- x-axis: ``iter_id + 1`` (1-indexed iteration within the layer).
- y-axis (left): closed-loop tour cost.
- Two curves per subplot: ``best`` (solid line, square markers) and
  ``avg`` (dashed line, circle markers), each aggregated across the
  ``--width`` restarts.
- One subplot per layer; ``sharey=False`` because per-layer costs can
  differ by an order of magnitude.
- Color ramp: viridis.
- Suptitle has two lines:
    1. ``"GLOP sub-TSP solver convergence — {problem_type}{problem_size}, "
       "width={width}, val_size={val_size}, seed={seed}, tag={tag}"``
    2. ``"total time: {total_elapsed_s:.1f}s, "
       "final cost (best): {final_best:.4f}, "
       "final cost (avg): {final_avg:.4f}, "
       "do_block_2opt={do_block_2opt_any}"``

Usage:

    # Default invocation (output goes to results/solver_curve_<tag>.png):
    python scripts/plot_solver_curve.py results/iter_costs.jsonl

    # Custom output directory and tag override:
    python scripts/plot_solver_curve.py results/iter_costs.jsonl \\
            --out_dir figures/ --out_tag smoke

The script is import-safe: ``scripts.plot_solver_curve.plot(jsonl_path,
out_dir, out_tag=None)`` returns the path to the saved PNG so it can be
called from a notebook or wired into a downstream pipeline.

See also: ``main.py --iter_cost_log <path>``, the docstring of
:func:`utils.functions.LCP_TSP` for the JSONL record schema.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Use a headless backend so the script works on a server without a display.
# Mirrors the pattern used by utils/diagnosis.py and utils/compare_solvers.py.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# JSONL loading
# --------------------------------------------------------------------------- #


def load_iter_costs(jsonl_path: str) -> List[Dict[str, Any]]:
    """Read a ``--iter_cost_log`` JSONL file and return its records.

    Each line must be a valid JSON object matching the schema documented
    in :func:`utils.functions.LCP_TSP`. Lines that fail to parse are
    skipped with a warning so a corrupted trailing line does not abort
    the whole plot.

    Args:
        jsonl_path: path to the JSONL file produced by ``--iter_cost_log``.

    Returns:
        list of per-iter record dicts. Order is preserved (file order).
    """
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(
            f"iter-cost log not found: {jsonl_path!r}. "
            "Run `python main.py ... --iter_cost_log <path>` first."
        )

    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r") as fh:
        for line_no, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as err:
                print(
                    f"[plot_solver_curve] WARN: skipping malformed line "
                    f"{line_no} of {jsonl_path}: {err}",
                    file=sys.stderr,
                )
    if not records:
        raise ValueError(
            f"iter-cost log is empty (or all lines malformed): {jsonl_path!r}"
        )
    return records


def group_by_layer(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Group records by ``layer_id`` and return layers + run metadata.

    The first record's run-level metadata is returned as a separate dict
    so the plot can stamp it into the suptitle. The function assumes all
    records share the same run metadata (which is the case for any single
    ``--iter_cost_log`` run).

    Because ``_eval_dataset`` calls ``reconnect`` once per data batch and
    each call appends a per-iter record, a single ``(layer_id, iter_id)``
    typically has multiple records (one per batch). Those records are
    aggregated by taking the weighted mean of ``best`` / ``avg`` /
    ``cost_before_2opt_best`` / ``cost_before_2opt_avg`` across records
    that share the same ``(layer_id, iter_id)``, weighted by ``count``.
    This is the correct reduction when batch sizes differ.

    The final layer's ``total_elapsed_s_last`` is the max across all
    per-iter records in that layer (the cumulative time, not the
    per-batch sum).

    Args:
        records: list of records from :func:`load_iter_costs`.

    Returns:
        A 2-tuple ``(layers, run_meta)``:

        - ``layers``: list of dicts ``{"layer_id", "revision_len",
          "revision_iter", "do_block_2opt", "iters": [int],
          "best": [float], "avg": [float], "total_elapsed_s_last":
          float, "count_last": int}``, sorted by ``layer_id`` ascending
          (Layer 1, Layer 2, Layer 3 from left to right).
        - ``run_meta``: the run-level metadata dict taken from the first
          record (used to annotate the suptitle).
    """
    run_meta: Dict[str, Any] = {
        k: v
        for k, v in records[0].items()
        if k
        not in {
            "ts",
            "layer_id",
            "iter_id",
            "revision_len",
            "revision_iter",
            "do_block_2opt",
            "best",
            "avg",
            "count",
            "cost_before_2opt_best",
            "cost_before_2opt_avg",
            "iter_elapsed_s",
            "total_elapsed_s",
        }
    }

    by_layer: Dict[Any, List[Dict[str, Any]]] = {}
    for r in records:
        by_layer.setdefault(r.get("layer_id"), []).append(r)

    def _weighted_mean(values: List[float], weights: List[float]) -> float:
        total_w = sum(weights)
        if total_w <= 0:
            # Fall back to a plain mean if every weight is non-positive.
            return float(sum(values) / max(len(values), 1))
        return float(sum(v * w for v, w in zip(values, weights)) / total_w)

    layers: List[Dict[str, Any]] = []
    for lid, recs in by_layer.items():
        if not recs:
            continue
        # Group by iter_id, then weighted-average across the per-batch records.
        by_iter: Dict[int, List[Dict[str, Any]]] = {}
        for r in recs:
            by_iter.setdefault(int(r.get("iter_id", 0)), []).append(r)

        iters_sorted = sorted(by_iter.keys())
        iters: List[int] = []
        best: List[float] = []
        avg: List[float] = []
        max_elapsed: float = 0.0
        count_last: int = 0
        for it in iters_sorted:
            iter_recs = by_iter[it]
            weights = [max(int(r.get("count", 0)), 1) for r in iter_recs]
            iters.append(it)
            best.append(
                _weighted_mean(
                    [float(r.get("best", 0.0)) for r in iter_recs], weights
                )
            )
            avg.append(
                _weighted_mean(
                    [float(r.get("avg", 0.0)) for r in iter_recs], weights
                )
            )
            # Cumulative ``total_elapsed_s`` is monotonic within an LCP_TSP
            # call, so the maximum is the meaningful end-of-iter value.
            for r in iter_recs:
                max_elapsed = max(max_elapsed, float(r.get("total_elapsed_s", 0.0)))
            count_last = max(int(r.get("count", 0)) for r in iter_recs)

        layers.append(
            {
                "layer_id": lid,
                "revision_len": recs[0].get("revision_len"),
                "revision_iter": recs[0].get("revision_iter"),
                "do_block_2opt": bool(recs[0].get("do_block_2opt", False)),
                "iters": iters,
                "best": best,
                "avg": avg,
                "total_elapsed_s_last": max_elapsed,
                "count_last": count_last,
            }
        )

    # Sort by layer_id ascending — the user wants Layer 1, Layer 2, Layer 3
    # from left to right, in the order the revisors ran.
    layers.sort(
        key=lambda d: (
            d.get("layer_id") if d.get("layer_id") is not None else 0,
        )
    )
    return layers, run_meta


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def _format_float(x: float, digits: int = 3) -> str:
    """Format a float compactly; fall back to ``%g`` if ``:.Nf`` overflows."""
    try:
        s = f"{x:.{digits}f}"
        return s
    except (ValueError, OverflowError):
        return f"{x:g}"


def plot(
    jsonl_path: str,
    out_dir: str = "results",
    out_tag: Optional[str] = None,
    dpi: int = 120,
) -> str:
    """Render the per-layer convergence plot and save it to a PNG.

    Args:
        jsonl_path: path to the JSONL sidecar written by ``--iter_cost_log``.
        out_dir: directory for the output PNG; created if missing. Default
            ``"results"`` (matches the rest of the project's output
            convention).
        out_tag: tag to embed in the output filename. Default ``None``
            derives a tag from the JSONL path's stem, falling back to
            ``"run"`` if the stem is empty.
        dpi: dots-per-inch for the saved PNG. Default ``120``.

    Returns:
        Absolute path to the saved PNG file.

    Raises:
        FileNotFoundError: if ``jsonl_path`` does not exist.
        ValueError: if the JSONL is empty or every line is malformed.

    See also:
        :func:`utils.functions.LCP_TSP` — the JSONL record schema.
        ``main.py --iter_cost_log <path>`` — the producer of the JSONL.
    """
    records = load_iter_costs(jsonl_path)
    layers, run_meta = group_by_layer(records)

    if not layers:
        raise ValueError(
            f"no per-iter records found in {jsonl_path!r}; nothing to plot."
        )

    n_layers = len(layers)
    fig, axes = plt.subplots(
        1, n_layers, figsize=(5 * n_layers, 4.5), sharey=False
    )
    if n_layers == 1:
        axes = [axes]
    colors = plt.cm.viridis(
        [i / max(n_layers - 1, 1) for i in range(n_layers)]
    )

    for ax, layer, color in zip(axes, layers, colors):
        x = [it + 1 for it in layer["iters"]]
        n_iters = (
            max(layer["iters"]) + 1
            if layer["iters"]
            else 0
        )
        # ``best`` (solid) and ``avg`` (dashed) lines; both drawn even when
        # they overlap (e.g. width=1) so the legend stays consistent.
        ax.plot(
            x,
            layer["avg"],
            "--",
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5,
            label="avg across --width",
        )
        ax.plot(
            x,
            layer["best"],
            "-",
            color=color,
            linewidth=2.0,
            marker="s",
            markersize=5,
            label="best across --width",
        )
        ax.set_title(
            f"Layer {layer['layer_id'] + 1}: L={layer['revision_len']} "
            f"({n_iters} iters, do_block_2opt={layer['do_block_2opt']})"
        )
        ax.set_xlabel("Iteration within layer")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        # Annotate first and last ``best`` value.
        if layer["best"]:
            first_x, first_y = x[0], layer["best"][0]
            last_x, last_y = x[-1], layer["best"][-1]
            ax.annotate(
                f"{_format_float(first_y)}",
                xy=(first_x, first_y),
                xytext=(3, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
            ax.annotate(
                f"{_format_float(last_y)}",
                xy=(last_x, last_y),
                xytext=(-30, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

    axes[0].set_ylabel("Closed-loop tour cost\n(eval-dataset mean)")

    # Suptitle: two lines. Line 1 = run identity (mirrors the LCP run
    # header in the README / main.py). Line 2 = the headline metrics the
    # user asked for: total time, final cost (best + avg), and the
    # block-level 2-opt toggle that applied during this run.
    total_elapsed_last = layers[-1]["total_elapsed_s_last"]
    final_best = layers[-1]["best"][-1] if layers[-1]["best"] else float("nan")
    final_avg = layers[-1]["avg"][-1] if layers[-1]["avg"] else float("nan")
    do_block_2opt_any = any(layer["do_block_2opt"] for layer in layers)
    suptitle = (
        "GLOP sub-TSP solver convergence — "
        f"{run_meta.get('problem_type', '?')}{run_meta.get('problem_size', '?')}, "
        f"width={run_meta.get('width', '?')}, "
        f"val_size={run_meta.get('val_size', '?')}, "
        f"seed={run_meta.get('seed', '?')}, "
        f"tag={run_meta.get('tag', '?')}\n"
        f"total time: {_format_float(total_elapsed_last, 1)}s   |   "
        f"final cost (best): {_format_float(final_best, 4)}   |   "
        f"final cost (avg): {_format_float(final_avg, 4)}   |   "
        f"do_block_2opt={do_block_2opt_any}"
    )
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.78)

    os.makedirs(out_dir, exist_ok=True)
    if out_tag is None:
        stem = os.path.splitext(os.path.basename(jsonl_path))[0]
        # Strip a trailing "_iter_costs" so the output PNG doesn't carry
        # the same suffix twice (the project convention is
        # ``solver_curve_<tag>.png``).
        if stem.endswith("_iter_costs"):
            stem = stem[: -len("_iter_costs")]
        if not stem:
            stem = "run"
        out_tag = stem
    out_path = os.path.join(out_dir, f"solver_curve_{out_tag}.png")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"=== Solver curve saved to: {out_path} ===")
    return out_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a GLOP sub-TSP convergence plot from a JSONL log. "
            "The log is produced by `python main.py ... --iter_cost_log <path>`."
        )
    )
    parser.add_argument(
        "jsonl_path",
        type=str,
        help="Path to the iter-cost JSONL log (one JSON object per line).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help=(
            "Directory to write the PNG into. Created if missing. "
            "Defaults to 'results' to match the rest of the project's "
            "output convention."
        ),
    )
    parser.add_argument(
        "--out_tag",
        type=str,
        default=None,
        help=(
            "Tag for the output filename. The PNG is named "
            "`solver_curve_<out_tag>.png`. Defaults to the JSONL path's "
            "stem (with any trailing '_iter_costs' stripped), so "
            "`results/run_iter_costs.jsonl` becomes "
            "`results/solver_curve_run.png`."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Dots-per-inch for the saved PNG. Default 120.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        plot(args.jsonl_path, out_dir=args.out_dir, out_tag=args.out_tag, dpi=args.dpi)
    except (FileNotFoundError, ValueError) as err:
        print(f"[plot_solver_curve] ERROR: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
