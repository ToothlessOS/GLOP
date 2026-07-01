"""Plot objective vs iteration for GLOP baseline vs heatmap-guided.

Reads `{in_dir}/tsp{N}/raw.json` and writes `{in_dir}/tsp{N}/cost_vs_iter.png`
for each problem size `N` it finds under `in_dir`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np

# matplotlib is optional — only required to render the chart. Import
# lazily so the rest of the experiment (heatmap, align, revision,
# benchmark) works in environments without matplotlib (e.g. the
# headless `glop` conda env).
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:  # pragma: no cover
    _HAVE_MPL = False


def aggregate(history: List[dict]) -> np.ndarray:
    """Reduce a per-instance cost history to a (n_iter,) mean curve.

    Each entry in `history` is a dict with `cost: List[float]`
    (one float per instance in the eval batch). We take the mean over
    the instance axis to get a single curve per iter.
    """
    return np.array([np.mean(h["cost"]) for h in history])


def plot_problem_size(
    problem_size: int, raw_path: str, out_path: str
) -> None:
    if not _HAVE_MPL:
        print(
            f"[!] matplotlib not installed — skipping {out_path}. "
            f"Install with `pip install matplotlib` to enable plotting."
        )
        return
    with open(raw_path) as f:
        raw = json.load(f)

    cost_baseline = aggregate(raw["baseline"]["history"])
    cost_guided = aggregate(raw["guided"]["history"])
    cost_ori_mean = float(np.mean(raw["cost_ori"]))

    # Stage boundaries (only used to draw the stage shading on the
    # guided curve — the baseline shading is symmetric).
    stages_guided = [h["stage"] for h in raw["guided"]["history"]]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    iters_b = np.arange(1, len(cost_baseline) + 1)
    iters_g = np.arange(1, len(cost_guided) + 1)

    ax.plot(iters_b, cost_baseline, label="GLOP baseline", color="tab:blue", lw=2)
    ax.plot(iters_g, cost_guided, label="Heatmap-guided", color="tab:red", lw=2)
    ax.axhline(
        cost_ori_mean, color="gray", linestyle="--", lw=1, label="warm start"
    )

    # Shade the cascade stages
    revision_lens = raw["revision_lens"]
    n_stages = len(revision_lens)
    for stage_id, reviser_size in enumerate(revision_lens):
        # Find the first iter of this stage in the guided history
        starts = [i for i, s in enumerate(stages_guided) if s == stage_id]
        if not starts:
            continue
        a = starts[0] + 1
        b = starts[-1] + 1 if stage_id < n_stages - 1 else iters_g[-1] + 1
        ax.axvspan(a, b, alpha=0.08, color="orange")
        ax.text(
            (a + b) / 2,
            ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0,
            f"reviser {reviser_size}",
            ha="center",
            va="top",
            fontsize=8,
            color="darkorange",
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("tour cost")
    ax.set_title(f"TSP-{problem_size}  (val_size={raw['val_size']}, width={raw['width']})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[*] Wrote {out_path}")


def main(opts) -> None:
    in_dir = opts.in_dir
    out_dir = opts.out_dir or in_dir
    os.makedirs(out_dir, exist_ok=True)
    for sub in sorted(os.listdir(in_dir)):
        sub_path = os.path.join(in_dir, sub)
        raw_path = os.path.join(sub_path, "raw.json")
        if not os.path.isfile(raw_path):
            continue
        try:
            problem_size = int(sub.replace("tsp", ""))
        except ValueError:
            continue
        out_path = os.path.join(sub_path, "cost_vs_iter.png")
        plot_problem_size(problem_size, raw_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="experiments/heatmap_guided/results")
    parser.add_argument("--out_dir", type=str, default="")
    opts = parser.parse_args()
    main(opts)
