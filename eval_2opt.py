"""Evaluate and compare GLOP with 2-opt post-processing.

Runs the SAME TSP instances through five configurations and reports both the
final tour cost and the per-iteration convergence of each:

  1. baseline     — GLOP only (no 2-opt)
  2. final        — GLOP, then full 2-opt once at the end of the pipeline
  3. per_iter     — GLOP with full 2-opt applied after every revisor iteration
  4. knn_final    — GLOP, then KNN-sparse 2-opt once at the end
  5. knn_per_iter — GLOP with KNN-sparse 2-opt applied after every revisor iter

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
    oom_cls = getattr(torch.cuda, 'OutOfMemoryError', None)
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    # Some CUDA OOMs surface as a plain RuntimeError on older / custom builds.
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if 'cuda out of memory' in msg or 'out of memory' in msg:
            return True
    # CPU OOM (numpy.memmap etc.).
    if isinstance(exc, MemoryError):
        return True
    return False


def _safe_run_mode(label, use_2opt, two_opt_mode, two_opt_kind, two_opt_knn_k,
                   base_opts, revisers):
    """Run one configuration; on OOM, return a sentinel row instead of raising.

    Returns the same dict shape as ``run_mode`` with two extra keys:
        ``skipped`` (bool): True iff this row is an OOM fallback.
        ``error``   (str):  textual error message when skipped.

    Other exception types (FileNotFoundError, user errors, etc.) are
    propagated untouched.
    """
    try:
        return run_mode(label, use_2opt, two_opt_mode, two_opt_kind,
                        two_opt_knn_k, base_opts, revisers)
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
            f'\n[ERROR] OOM in mode={label} '
            f'(kind={two_opt_kind}, knn_k={two_opt_knn_k}); '
            f'skipping this mode. error={exc!r}\n',
            flush=True,
        )
        return {
            'label': label,
            'two_opt_kind': two_opt_kind,
            'two_opt_knn_k': two_opt_knn_k,
            'avg': float('nan'),
            'best': float('nan'),
            'duration': float('nan'),
            'curve': {},
            'skipped': True,
            'error': str(exc),
        }


# Modes to compare: (label, use_2opt, two_opt_mode, two_opt_kind, two_opt_knn_k)
MODES = [
    ('baseline',     False, 'final',    'full', 20),
    ('final',        True,  'final',    'full', 20),
    ('per_iter',     True,  'per_iter', 'full', 20),
    ('knn_final',    True,  'final',    'knn',  20),
    ('knn_per_iter', True,  'per_iter', 'knn',  20),
]
# Each row's `use_2opt` flags whether the pipeline invokes the optional 2-opt
# step; `two_opt_kind` selects full vs. knn in `maybe_two_opt` (post_process.py).


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
    p.add_argument('--two_opt_kind', type=str, default='full',
                   choices=['full', 'knn'],
                   help="2-opt algorithm variant: 'full' (dense) or "
                        "'knn' (k-NN-sparse; uses --two_opt_knn_k)")
    p.add_argument('--two_opt_knn_k', type=int, default=20,
                   help='k for KNN-sparse 2-opt (only used when --two_opt_kind=knn)')
    p.add_argument('--two_opt_debug', action='store_true',
                   help='Print per-sweep 2-opt phase timings to stdout for '
                        'performance investigation.')
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


def run_mode(label, use_2opt, two_opt_mode, two_opt_kind, two_opt_knn_k,
              base_opts, revisers):
    """Run one configuration on a fresh copy of opts with a reset RNG."""
    opts = copy.deepcopy(base_opts)
    opts.use_2opt = use_2opt
    opts.two_opt_mode = two_opt_mode
    opts.two_opt_kind = two_opt_kind
    opts.two_opt_knn_k = two_opt_knn_k
    opts.two_opt_debug = getattr(base_opts, 'two_opt_debug', False)

    # Reset RNG so every mode sees identical warm-start tours / sampling draws.
    torch.manual_seed(base_opts.seed)
    np.random.seed(base_opts.seed)

    print(f'\n===================== running mode: {label} '
          f'(use_2opt={use_2opt}, mode={two_opt_mode}, '
          f'kind={two_opt_kind}, knn_k={two_opt_knn_k}) =====================')
    results, duration, all_stats = _eval_dataset(opts.path, opts, opts.device, revisers)

    costs = final_costs(results)
    return {
        'label': label,
        'two_opt_kind': two_opt_kind,
        'two_opt_knn_k': two_opt_knn_k,
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

    colors = {
        'baseline':     '#7f7f7f',
        'final':        '#1f77b4',
        'per_iter':     '#d62728',
        'knn_final':    '#2ca02c',
        'knn_per_iter': '#9467bd',
    }
    markers = {
        'baseline':     'o',
        'final':        's',
        'per_iter':     '^',
        'knn_final':    'D',
        'knn_per_iter': 'v',
    }

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

    has_knn = any(r['label'].startswith('knn') for r in runs)
    kind_suffix = (
        f", kind={opts.two_opt_kind}" + (f", k={opts.two_opt_knn_k}" if opts.two_opt_kind == 'knn' else "")
        if has_knn else ""
    )
    fig.suptitle(
        f"GLOP + 2-opt comparison — tsp{opts.problem_size}, width={opts.width}, "
        f"val_size={opts.val_size}, lens={opts.revision_lens}, iters={opts.revision_iters}, "
        f"2opt_iters={opts.two_opt_iters}{kind_suffix}", fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    os.makedirs(opts.out_dir, exist_ok=True)
    has_knn = any(r['label'].startswith('knn') for r in runs)
    kind_tag = (
        f"_kind{opts.two_opt_kind}" + (f"_k{opts.two_opt_knn_k}" if opts.two_opt_kind == 'knn' else "")
        if has_knn else ""
    )
    tag = (f"tsp{opts.problem_size}_w{opts.width}"
           f"_lens{'-'.join(map(str, opts.revision_lens))}"
           f"_iters{'-'.join(map(str, opts.revision_iters))}_2opt{opts.two_opt_iters}"
           f"{kind_tag}")
    out_path = os.path.join(opts.out_dir, f'twoopt_compare_{tag}.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def print_table(runs):
    # Pick a non-OOM baseline; fall back to the first run if even baseline OOMed.
    base_avg = None
    for r in runs:
        if r['label'] == 'baseline' and not r.get('skipped'):
            base_avg = r['avg']
            break
    if base_avg is None or (isinstance(base_avg, float) and math.isnan(base_avg)):
        for r in runs:
            if not r.get('skipped') and not math.isnan(r['avg']):
                base_avg = r['avg']
                break
    print('\n================= Final performance =================')
    header = (f"{'mode':<14}{'kind':<6}{'avg cost':>12}{'best cost':>12}"
              f"{'impr% vs base':>16}{'time (s)':>12}")
    print(header)
    print('-' * len(header))
    # Infer kind from the mode label if not already on the record.
    for r in runs:
        if 'two_opt_kind' not in r:
            r['two_opt_kind'] = 'knn' if r['label'].startswith('knn') else 'full'
    for r in runs:
        if r.get('skipped') or math.isnan(r['avg']):
            # OOM rows: render placeholders so the table stays aligned.
            print(f"{r['label']:<14}{r['two_opt_kind']:<6}{'OOM':>12}"
                  f"{'OOM':>12}{'-':>16}{'-':>12}")
            continue
        impr = (100.0 * (base_avg - r['avg']) / base_avg
                if base_avg and not math.isnan(base_avg) else 0.0)
        print(f"{r['label']:<14}{r['two_opt_kind']:<6}{r['avg']:>12.4f}"
              f"{r['best']:>12.4f}{impr:>15.2f}%{r['duration']:>12.2f}")
    print('=====================================================')


def main():
    opts = build_base_opts()
    print('using device:', opts.device)
    print('dataset:', opts.path)
    assert os.path.exists(opts.path), f'dataset not found: {opts.path}'

    revisers = load_revisers(opts)

    t0 = time.time()
    # Use the OOM-safe wrapper so a single mode running out of memory does
    # not abort the whole comparison. Other exception types still propagate.
    runs = [_safe_run_mode(label, use2, mode, kind, knn_k, opts, revisers)
            for (label, use2, mode, kind, knn_k) in MODES]
    skipped = [r['label'] for r in runs if r.get('skipped')]
    if skipped:
        print(f'\n[NOTE] Skipped modes due to OOM: {", ".join(skipped)}',
              flush=True)
    print_table(runs)
    out_path = plot_comparison(runs, opts)
    print(f'\n=== Comparison figure saved to: {out_path} ===')
    print(f'=== Total wall-clock: {time.time() - t0:.2f}s ===')


if __name__ == '__main__':
    main()
