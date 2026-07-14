"""Evaluate and compare GLOP with 2-opt post-processing.

Runs the SAME TSP instances through three configurations and reports both the
final tour cost and the per-iteration convergence of each:

  1. baseline  — GLOP only (no 2-opt)
  2. final     — GLOP, then full 2-opt once at the end of the pipeline
  3. per_iter  — GLOP with 2-opt applied after every revisor iteration

Outputs:
  * a printed summary table of final performance (avg / best cost, duration)
  * a figure (results/twoopt_compare_<tag>.png) with two panels:
      - per-iteration convergence curve, one line per mode
      - final-cost bar chart with % improvement vs. baseline

The three runs reuse a single set of loaded revisers and reset the RNG seed
before each run, so the warm-start tours are identical across modes and the
comparison is fair.

Example:
  python eval_2opt.py --problem_size 100 --revision_lens 50 20 \
      --revision_iters 10 5 --width 4 --eval_batch_size 8 --val_size 8 \
      --decode_strategy greedy --two_opt_iters 30
"""

import argparse
import copy
import os
import time

import numpy as np
import torch

from utils import load_model
# Reuse the exact eval machinery from main.py so this script tracks the pipeline.
from main import _eval_dataset, _aggregate_solver_curve


# Modes to compare: (label, use_2opt, two_opt_mode)
MODES = [
    ('baseline', False, 'final'),
    ('final', True, 'final'),
    ('per_iter', True, 'per_iter'),
]


def build_base_opts():
    """Argparse for the subset of options relevant to a TSP 2-opt comparison.

    Field names mirror main.py so the reused pipeline code finds what it needs.
    """
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--problem_size', type=int, default=100)
    p.add_argument('--problem_type', type=str, default='tsp', choices=['tsp'],
                   help='This comparison script targets TSP.')
    p.add_argument('--path', type=str, default='',
                   help='Test dataset path; defaults to data/tsp/tsp<size>_test.pkl')
    p.add_argument('--val_size', type=int, default=8)
    p.add_argument('--eval_batch_size', type=int, default=8)
    p.add_argument('--width', type=int, default=4)
    p.add_argument('--revision_lens', nargs='+', type=int, default=[50, 20])
    p.add_argument('--revision_iters', nargs='+', type=int, default=[10, 5])
    p.add_argument('--decode_strategy', type=str, default='greedy',
                   help="'greedy' recommended for a deterministic comparison")
    p.add_argument('--two_opt_iters', type=int, default=30,
                   help='Max 2-opt sweeps per invocation (final and per_iter modes)')
    p.add_argument('--no_aug', action='store_true')
    p.add_argument('--no_prune', action='store_true')
    p.add_argument('--no_progress_bar', action='store_true')
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--device_id', type=int, default=0)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--out_dir', type=str, default='results')

    opts = p.parse_args()

    # Fields the reused pipeline reads but this CLI does not expose.
    opts.n_subset = 1
    opts.n_partition = 1
    opts.ckpt_path = ''
    opts.tsp_aug = False  # _eval_dataset may flip this for problem_size <= 100

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.device = torch.device(f'cuda:{opts.device_id}' if use_cuda else 'cpu')

    if opts.path == '':
        opts.path = f'data/tsp/tsp{opts.problem_size}_test.pkl'
    return opts


def load_revisers(opts):
    revisers = []
    for reviser_size in opts.revision_lens:
        reviser_path = f'pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
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


def run_mode(label, use_2opt, two_opt_mode, base_opts, revisers):
    """Run one configuration on a fresh copy of opts with a reset RNG."""
    opts = copy.deepcopy(base_opts)
    opts.use_2opt = use_2opt
    opts.two_opt_mode = two_opt_mode

    # Reset RNG so every mode sees identical warm-start tours / sampling draws.
    torch.manual_seed(base_opts.seed)
    np.random.seed(base_opts.seed)

    print(f'\n===================== running mode: {label} '
          f'(use_2opt={use_2opt}, mode={two_opt_mode}) =====================')
    results, duration, all_stats = _eval_dataset(opts.path, opts, opts.device, revisers)

    costs = final_costs(results)
    return {
        'label': label,
        'avg': costs.mean().item(),
        'best': costs.min().item(),
        'duration': duration,
        'curve': _aggregate_solver_curve(all_stats),  # {layer_id: {iters,best,avg}}
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
        for b in data['best']:
            gx += 1
            xs.append(gx)
            ys.append(b)
    return xs, ys, boundaries


def plot_comparison(runs, opts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'baseline': '#7f7f7f', 'final': '#1f77b4', 'per_iter': '#d62728'}
    markers = {'baseline': 'o', 'final': 's', 'per_iter': '^'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={'width_ratios': [2, 1]})

    # --- Panel 1: per-iteration convergence ---
    all_boundaries = []
    max_x = 0
    for r in runs:
        xs, ys, boundaries = flatten_curve(r['curve'])
        if not xs:
            continue
        all_boundaries = boundaries  # same layout across modes
        max_x = max(max_x, xs[-1])
        ax1.plot(xs, ys, '-', color=colors[r['label']], marker=markers[r['label']],
                 markersize=4, linewidth=1.8, label=f"{r['label']} (per-iter)")
        # Terminal marker = the mode's FINAL cost after the whole pipeline
        # (captures the end-of-pipeline 2-opt drop for the 'final' mode).
        ax1.plot([xs[-1], xs[-1] + 1], [ys[-1], r['avg']], ':',
                 color=colors[r['label']], linewidth=1.2)
        ax1.scatter([xs[-1] + 1], [r['avg']], color=colors[r['label']],
                    marker='*', s=130, zorder=5)

    for bx in all_boundaries:
        ax1.axvline(bx, color='k', linestyle=':', alpha=0.25)
    ax1.set_xlabel('Global revisor iteration (layers concatenated;  ★ = final cost)')
    ax1.set_ylabel('Mean tour cost over eval set\n(best across --width)')
    ax1.set_title('Cost at every iteration')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)

    # --- Panel 2: final performance bar chart ---
    labels = [r['label'] for r in runs]
    avgs = [r['avg'] for r in runs]
    bests = [r['best'] for r in runs]
    base_avg = next((r['avg'] for r in runs if r['label'] == 'baseline'), avgs[0])
    x = np.arange(len(labels))
    bars = ax2.bar(x, avgs, color=[colors[l] for l in labels], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Final mean tour cost')
    ax2.set_title('Final performance')
    ax2.grid(True, axis='y', alpha=0.3)
    lo = min(avgs) * 0.995
    hi = max(avgs) * 1.005
    ax2.set_ylim(lo, hi)
    for xi, (bar, avg, best) in enumerate(zip(bars, avgs, bests)):
        impr = 100.0 * (base_avg - avg) / base_avg if base_avg else 0.0
        ax2.annotate(f"avg {avg:.3f}\nbest {best:.3f}\n({impr:+.2f}%)",
                     xy=(xi, avg), xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8)

    fig.suptitle(
        f"GLOP + 2-opt comparison — tsp{opts.problem_size}, width={opts.width}, "
        f"val_size={opts.val_size}, lens={opts.revision_lens}, iters={opts.revision_iters}, "
        f"2opt_iters={opts.two_opt_iters}", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    os.makedirs(opts.out_dir, exist_ok=True)
    tag = (f"tsp{opts.problem_size}_w{opts.width}"
           f"_lens{'-'.join(map(str, opts.revision_lens))}"
           f"_iters{'-'.join(map(str, opts.revision_iters))}_2opt{opts.two_opt_iters}")
    out_path = os.path.join(opts.out_dir, f'twoopt_compare_{tag}.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def print_table(runs):
    base_avg = next((r['avg'] for r in runs if r['label'] == 'baseline'), runs[0]['avg'])
    print('\n================= Final performance =================')
    header = f"{'mode':<10}{'avg cost':>12}{'best cost':>12}{'impr% vs base':>16}{'time (s)':>12}"
    print(header)
    print('-' * len(header))
    for r in runs:
        impr = 100.0 * (base_avg - r['avg']) / base_avg if base_avg else 0.0
        print(f"{r['label']:<10}{r['avg']:>12.4f}{r['best']:>12.4f}{impr:>15.2f}%{r['duration']:>12.2f}")
    print('=====================================================')


def main():
    opts = build_base_opts()
    print('using device:', opts.device)
    print('dataset:', opts.path)
    assert os.path.exists(opts.path), f'dataset not found: {opts.path}'

    revisers = load_revisers(opts)

    t0 = time.time()
    runs = [run_mode(label, use2, mode, opts, revisers) for (label, use2, mode) in MODES]
    print_table(runs)
    out_path = plot_comparison(runs, opts)
    print(f'\n=== Comparison figure saved to: {out_path} ===')
    print(f'=== Total wall-clock: {time.time() - t0:.2f}s ===')


if __name__ == '__main__':
    main()
