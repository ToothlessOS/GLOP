#!/usr/bin/env python3
"""
Benchmark: compare original GLOP vs GLOP + connection revision.

For each requested test size, runs ``main.py`` twice (with and without
``--use_connection_revision``) with identical settings and reports a
side-by-side comparison of the average cost_revised.

Example:
    python benchmark_connection_revision.py \\
        --test_sizes 1000 10000 \\
        --widths 16 4 \\
        --revision_iters 10 10 5
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_sizes', nargs='+', type=int, default=[1000, 10000],
                        help='TSP test sizes to benchmark.')
    parser.add_argument('--widths', nargs='+', type=int, default=[16, 4],
                        help='Width per test size (one value, or one per test_size).')
    parser.add_argument('--revision_lens', nargs='+', type=int, default=[100, 50, 20],
                        help='Reviser window sizes.')
    parser.add_argument('--revision_iters', nargs='+', type=int, default=[10, 10, 5],
                        help='Revision iterations per scale.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed for reproducible runs.')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='Where to write the JSON results.')
    parser.add_argument('--python', type=str, default='python',
                        help='Python executable to use (e.g., the glop conda env).')
    parser.add_argument('--main_script', type=str, default='main.py',
                        help='Path to main.py (relative to repo root).')
    parser.add_argument('--max_instances', type=int, default=None,
                        help='If set, limit --val_size to this number (for quick smoke tests).')
    parser.add_argument('--no_aug', action='store_true',
                        help='Pass --no_aug to main.py (disables 4x instance augmentation).')
    parser.add_argument('--val_size', type=int, default=None,
                        help='Override val_size for all test sizes.')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Override eval_batch_size for all test sizes.')
    parser.add_argument('--per_size_val_size', type=str, default=None,
                        help='Comma-separated key=value pairs of test_size:val_size '
                             '(e.g. "1000=128,10000=16") for per-size overrides.')
    parser.add_argument('--per_size_eval_batch_size', type=str, default=None,
                        help='Comma-separated key=value pairs of test_size:eval_batch_size '
                             '(e.g. "1000=128,10000=16") for per-size overrides.')
    return parser.parse_args()


def parse_per_size(spec):
    """Parse '1000=128,10000=16' into {1000: 128, 10000: 16}."""
    if spec is None:
        return {}
    out = {}
    for chunk in spec.split(','):
        k, v = chunk.split('=')
        out[int(k.strip())] = int(v.strip())
    return out


def build_cmd(args, test_size, width, use_conn_rev):
    cmd = [
        args.python, args.main_script,
        '--problem_type', 'tsp',
        '--problem_size', str(test_size),
        '--revision_lens', *[str(l) for l in args.revision_lens],
        '--revision_iters', *[str(i) for i in args.revision_iters],
        '--width', str(width),
        '--seed', str(args.seed),
    ]
    if use_conn_rev:
        cmd.append('--use_connection_revision')
    if args.no_aug:
        cmd.append('--no_aug')
    per_val = parse_per_size(args.per_size_val_size)
    per_ebs = parse_per_size(args.per_size_eval_batch_size)
    val_size = per_val.get(test_size, args.val_size)
    ebs = per_ebs.get(test_size, args.eval_batch_size)
    if val_size is not None:
        cmd += ['--val_size', str(val_size)]
    if ebs is not None:
        cmd += ['--eval_batch_size', str(ebs)]
    if args.max_instances is not None:
        cmd += ['--val_size', str(args.max_instances)]
    return cmd


COST_RE = re.compile(r'Average cost_revised:\s*([0-9.eE+\-]+)')
TIME_RE = re.compile(r'Total duration:\s*([0-9.]+)')


def run_main(args, test_size, width, use_conn_rev):
    cmd = build_cmd(args, test_size, width, use_conn_rev)
    label = 'ConnRev' if use_conn_rev else 'Original'
    print(f"  [{label}] running: {' '.join(cmd)}")
    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - start
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"main.py failed for test_size={test_size}, "
                           f"use_conn_rev={use_conn_rev} (exit={proc.returncode})")
    cost_match = COST_RE.search(proc.stdout)
    time_match = TIME_RE.search(proc.stdout)
    if not cost_match:
        print(proc.stdout)
        raise RuntimeError(f"Could not parse 'Average cost_revised' from output")
    avg_cost = float(cost_match.group(1))
    reported_time = float(time_match.group(1)) if time_match else float('nan')
    return avg_cost, reported_time, wall


def main():
    args = parse_args()

    # Expand widths: a single value is broadcast across all test_sizes.
    if len(args.widths) == 1:
        widths = args.widths * len(args.test_sizes)
    elif len(args.widths) != len(args.test_sizes):
        raise ValueError("--widths must have length 1 or match --test_sizes")
    else:
        widths = args.widths

    print(f"Benchmark: original GLOP vs + connection revision")
    print(f"  revision_lens:  {args.revision_lens}")
    print(f"  revision_iters: {args.revision_iters}")
    print(f"  seed:           {args.seed}")
    print(f"  test_sizes:     {args.test_sizes}")
    print(f"  widths:         {widths}")

    results = []
    for test_size, width in zip(args.test_sizes, widths):
        print(f"\n=== TSP-{test_size} (width={width}) ===")
        orig_cost, orig_time, orig_wall = run_main(args, test_size, width, False)
        print(f"  Original:  cost={orig_cost:.4f}  "
              f"reported_time={orig_time:.1f}s  wall={orig_wall:.1f}s")
        conn_cost, conn_time, conn_wall = run_main(args, test_size, width, True)
        print(f"  +ConnRev:  cost={conn_cost:.4f}  "
              f"reported_time={conn_time:.1f}s  wall={conn_wall:.1f}s")
        delta = conn_cost - orig_cost
        delta_pct = (delta / orig_cost) * 100.0 if orig_cost != 0 else 0.0
        time_ratio = (conn_wall / orig_wall) if orig_wall > 0 else float('nan')
        print(f"  Δ:         {delta:+.4f} ({delta_pct:+.3f}%)  "
              f"time_ratio={time_ratio:.2f}x")
        results.append({
            'test_size': test_size,
            'width': width,
            'revision_lens': args.revision_lens,
            'revision_iters': args.revision_iters,
            'seed': args.seed,
            'original_cost': orig_cost,
            'conn_rev_cost': conn_cost,
            'delta': delta,
            'delta_pct': delta_pct,
            'original_time_reported': orig_time,
            'conn_rev_time_reported': conn_time,
            'original_wall': orig_wall,
            'conn_rev_wall': conn_wall,
            'wall_time_ratio': time_ratio,
        })

    # Summary table.
    print("\n=== Summary ===")
    header = (f"{'Test set':<12} {'Width':>5} {'Original':>10} {'+ConnRev':>10} "
              f"{'Δ':>10} {'Δ%':>8} {'Time orig':>10} {'Time new':>10} {'Ratio':>6}")
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"TSP-{r['test_size']:<6} {r['width']:>5} "
              f"{r['original_cost']:>10.4f} {r['conn_rev_cost']:>10.4f} "
              f"{r['delta']:>+10.4f} {r['delta_pct']:>+7.3f}% "
              f"{r['original_wall']:>9.1f}s {r['conn_rev_wall']:>9.1f}s "
              f"{r['wall_time_ratio']:>5.2f}x")

    # Persist JSON.
    out_path = os.path.abspath(args.output)
    with open(out_path, 'w') as f:
        json.dump({
            'config': {
                'revision_lens': args.revision_lens,
                'revision_iters': args.revision_iters,
                'seed': args.seed,
                'test_sizes': args.test_sizes,
                'widths': widths,
            },
            'results': results,
        }, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()