"""Lightweight benchmarking utility for GLOP evaluation runs.

Used by main.py and eval_tsplib.py to record per-stage cost and duration,
plus per-instance before/after cost, and optionally dump the records to
JSON for offline analysis.

The class is purely additive — call sites opt in by passing a BenchLogger
instance; existing call paths in eval_atsp/test_glop.py etc. are not
touched.

Typical usage:

    from utils.bench_utils import BenchLogger
    logger = BenchLogger(results_dir='bench_out/', save_json=True, tag='tsp100')

    # inside a batch loop:
    logger.add_stage('initial', cost_before=..., cost_after=..., duration_s=...)
    logger.add_instance(instance_id=i, cost_before=..., cost_after=...,
                        duration_s=..., breakdown={'cascade_0': ..., 'chunk_2opt_L0': ...})

    # at the end:
    logger.print_summary()
    logger.close()
"""
import json
import os
import re
import time
from typing import Optional


def _natural_key(s):
    """Sort key that orders embedded digits numerically rather than lexically.

    ``'cascade_2' < 'cascade_10'`` (natural) instead of
    ``'cascade_10' < 'cascade_2'`` (lexical). Non-digit segments are compared
    case-insensitively.
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


class BenchLogger:
    """Records per-stage and per-instance cost/duration for an eval run."""

    def __init__(self, results_dir: str = '', save_json: bool = False, tag: str = 'run'):
        self.results_dir = results_dir
        self.save_json = save_json
        self.tag = tag
        self.records = []           # list of dicts (per-stage entries)
        self.per_instance = []      # list of dicts (per-instance entries)
        self.t_start = time.time()

    # ---- writers ------------------------------------------------------------

    def add_stage(self, stage_name: str, *, batch_id=None, cost_before=None,
                  cost_after=None, duration_s=None, extras=None):
        """Append a stage entry (e.g. one cascade level, or one chunk_2opt call)."""
        rec = {'stage': stage_name, 'wall_s': time.time() - self.t_start}
        if batch_id is not None:
            rec['batch_id'] = int(batch_id)
        if cost_before is not None:
            rec['cost_before'] = _to_float_list(cost_before)
        if cost_after is not None:
            rec['cost_after'] = _to_float_list(cost_after)
        if duration_s is not None:
            rec['duration_s'] = float(duration_s)
        if extras:
            for k, v in extras.items():
                rec[k] = _to_jsonable(v)
        self.records.append(rec)

    def add_instance(self, instance_id, *, cost_before, cost_after,
                     duration_s=None, breakdown=None):
        """Append a per-instance entry. `breakdown` is a dict stage->cost_after."""
        rec = {
            'instance_id': int(instance_id),
            'cost_before': _to_float(cost_before),
            'cost_after': _to_float(cost_after),
        }
        if duration_s is not None:
            rec['duration_s'] = float(duration_s)
        if breakdown is not None:
            rec['breakdown'] = {k: _to_float_list(v) for k, v in breakdown.items()}
        self.per_instance.append(rec)

    def merge(self, other: 'BenchLogger'):
        """Concatenate another logger's records. Useful when batched on device."""
        self.records.extend(other.records)
        self.per_instance.extend(other.per_instance)

    # ---- readers ------------------------------------------------------------

    def summary(self) -> dict:
        """Compute aggregate summary statistics."""
        n = len(self.per_instance)
        if n == 0:
            return {'n': 0}
        befores = [r['cost_before'] for r in self.per_instance]
        afters = [r['cost_after'] for r in self.per_instance]
        n_improved = sum(1 for b, a in zip(befores, afters) if a < b - 1e-9)
        mean_before = sum(befores) / n
        mean_after = sum(afters) / n
        return {
            'n': n,
            'mean_cost_before': mean_before,
            'mean_cost_after': mean_after,
            'mean_cost_reduction': mean_before - mean_after,
            'mean_cost_reduction_pct': (100.0 * (mean_before - mean_after) / mean_before
                                        if mean_before > 0 else 0.0),
            'n_improved': n_improved,
            'pct_improved': 100.0 * n_improved / n,
        }

    def print_summary(self):
        """Print a human-readable summary to stdout."""
        s = self.summary()
        if s['n'] == 0:
            print("[BenchLogger] no per-instance records to summarize.")
            return
        print(
            "[BenchLogger] n={n} | mean cost: {b:.4f} -> {a:.4f} "
            "(reduction: {r:.4f}, {rp:.2f}%) | improved {ni}/{n} ({ip:.1f}%)".format(
                n=s['n'],
                b=s['mean_cost_before'],
                a=s['mean_cost_after'],
                r=s['mean_cost_reduction'],
                rp=s['mean_cost_reduction_pct'],
                ni=s['n_improved'],
                ip=s['pct_improved'],
            )
        )
        # Also print per-stage summary.
        if self.records:
            print("[BenchLogger] per-stage breakdown:")
            # Aggregate stages by name.
            from collections import defaultdict
            agg = defaultdict(lambda: {'n': 0, 'duration_s': 0.0,
                                      'cost_before_sum': 0.0, 'cost_after_sum': 0.0,
                                      'cost_count': 0})
            for rec in self.records:
                name = rec['stage']
                agg[name]['n'] += 1
                if 'duration_s' in rec:
                    agg[name]['duration_s'] += rec['duration_s']
                if 'cost_before' in rec and rec['cost_before']:
                    agg[name]['cost_before_sum'] += sum(rec['cost_before'])
                    agg[name]['cost_count'] += len(rec['cost_before'])
                if 'cost_after' in rec and rec['cost_after']:
                    agg[name]['cost_after_sum'] += sum(rec['cost_after'])
            for name in sorted(agg.keys(), key=_natural_key):
                a = agg[name]
                line = "  - {n}: n_calls={nc}".format(n=name, nc=a['n'])
                if a['duration_s'] > 0:
                    line += ", total_duration={d:.3f}s".format(d=a['duration_s'])
                if a['cost_count'] > 0:
                    mean_b = a['cost_before_sum'] / a['cost_count']
                    mean_a = a['cost_after_sum'] / a['cost_count']
                    line += ", mean_cost: {b:.4f} -> {a:.4f}".format(b=mean_b, a=mean_a)
                print(line)

    # ---- output -------------------------------------------------------------

    def dump_json(self, filepath: str):
        """Write a JSON file with all records and a summary."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        payload = {
            'tag': self.tag,
            'summary': self.summary(),
            'records': self.records,
            'per_instance': self.per_instance,
        }
        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=2)

    def close(self):
        """If save_json and results_dir is set, write timestamped JSON file."""
        if not (self.save_json and self.results_dir):
            return
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.results_dir, f'{self.tag}_{ts}.json')
        self.dump_json(path)
        print(f"[BenchLogger] wrote {path}")


def _to_float(x):
    """Coerce a tensor / scalar / list to a Python float."""
    if hasattr(x, 'item'):
        try:
            return float(x.item())
        except (ValueError, RuntimeError):
            pass
    return float(x)


def _to_float_list(x):
    """Coerce a 1-D tensor / list to a Python list of floats."""
    if hasattr(x, 'detach'):
        x = x.detach().cpu().view(-1).tolist()
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return [float(x)]


def _to_jsonable(v):
    """Coerce a value (possibly a tensor) to a JSON-serializable form."""
    if hasattr(v, 'item'):
        try:
            return float(v.item())
        except (ValueError, RuntimeError):
            pass
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v
