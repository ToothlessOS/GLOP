"""Standalone SHPP (Shortest Hamiltonian Path Problem) evaluation.

Compares the GLOP reviser-100 checkpoint against LKH-3 on a test set of
SHPP instances of size 100, drawn from the same `unit` and `scale` point
distributions the revisor was trained on.

Both solvers are given the same open-path problem:
  - the revisor is its native task (open path with anchored endpoints);
  - LKH-3 is run in standard TSP mode with a dummy depot node N+1 and
    FIXED_EDGES_SECTION entries that force the depot to sit between the
    end node N and the start node 1, turning the closed tour back into
    the SHPP we want.

Outputs per-instance results to a JSONL file and an aggregate summary to
a sibling JSON file.  Designed to mirror the `eval_*.py` scripts in the
project root (see `eval_2opt.py` and `eval_cvrp.py` for the closest
templates).
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# Make sibling modules importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import load_model  # noqa: E402


# --- Constants ------------------------------------------------------------

LKH_BIN_REL = "LKH-3.0.14/LKH"            # pre-built binary, relative to GLOP root
REVISER_PATH_REL = "pretrained/Reviser-stage2/reviser_100/epoch-299.pt"
SHPP_SIZE = 100                           # locked by user choice
DUMMY_ID = SHPP_SIZE + 1                  # 101 — internal depot
TSPLIB_SCALE = 10_000_000                 # coords are multiplied by this
                                          # before the rounded Euclidean
                                          # distance is computed, matching
                                          # `utils/lkh.py:write_tsplib`.
HUGE = 1_000_000_000_000                  # sentinel: must dominate any
                                          # real edge after TSPLIB_SCALE
                                          # (max edge ~1.4e7).


# --- Instance generation --------------------------------------------------

def generate_instances(n_unit, n_scale, N, seed):
    """Sample SHPP point sets from `unit` and `scale` distributions.

    Returns
    -------
    coords : np.ndarray, shape (M, N, 2), float64
    dist_labels : list[str] of length M, values in {"unit", "scale"}
    """
    rng = np.random.default_rng(seed)
    parts = []
    labels = []

    if n_unit > 0:
        parts.append(rng.uniform(0.0, 1.0, size=(n_unit, N, 2)))
        labels.extend(["unit"] * n_unit)

    if n_scale > 0:
        # Match `local_construction/generate_data.py:17-28`: per-instance
        # y_max ~ U(0,1) and x, y are uniform in their rectangles.
        scale = np.empty((n_scale, N, 2), dtype=np.float64)
        for i in range(n_scale):
            y_max = rng.uniform()
            scale[i, :, 0] = rng.uniform(0.0, 1.0, size=N)
            scale[i, :, 1] = rng.uniform(0.0, y_max, size=N)
        parts.append(scale)
        labels.extend(["scale"] * n_scale)

    coords = np.concatenate(parts, axis=0)
    # Shuffle so the two distributions are interleaved in the JSONL.
    perm = rng.permutation(coords.shape[0])
    return coords[perm], [labels[i] for i in perm]


# --- SHPP cost ------------------------------------------------------------

def shpp_cost(coords, tour):
    """Open-path L2 cost matching `LOCAL.get_costs` (no closing edge).

    `tour` is a length-N sequence of 0-indexed node IDs.  Orientation is
    enforced so index 0 is the start and index N-1 is the end (matching
    the revisor's anchored output).
    """
    N = coords.shape[0]
    assert len(tour) == N, f"tour length {len(tour)} != N={N}"
    pts = coords[list(tour)]
    return float(np.linalg.norm(pts[1:] - pts[:-1], axis=-1).sum())


# --- LKH-3 writer (TYPE: TSP + dummy depot + FIXED_EDGES_SECTION) ---------

def _rounded_euclidean(p, q):
    """LKH-3 EUC_2D's `round(sqrt(dx^2 + dy^2))`."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return int(round(math.sqrt(dx * dx + dy * dy)))


def write_shpp_lkh_files(directory, instance_id, coords):
    """Emit the .vrp and .par files for a single SHPP instance.

    The .vrp is a standard TSPLIB TSP file with DIMENSION=N+1, an
    EXPLICIT FULL_MATRIX distance matrix, and a FIXED_EDGES_SECTION
    forcing the dummy depot (id N+1) to sit between node N and node 1
    in the closed tour.  Edge weights involving the depot are 0 on
    the two forced edges and HUGE elsewhere, so the closed-tour cost
    equals the SHPP cost exactly.
    """
    N = coords.shape[0]
    name = f"shpp_{instance_id}"
    base = os.path.join(directory, f"{name}.hpp1")
    vrp_path = base + ".vrp"
    par_path = base + ".par"
    tour_path = base + ".tour"
    log_path = base + ".log"

    # --- Distance matrix: (N+1) x (N+1) ---------------------------------
    # 0-indexed: indices 0..N-1 are real nodes, index N is the dummy depot.
    # We want two undirected edges to have cost 0:
    #   depot <-> node 1   (for the closing arc depot -> 1)
    #   depot <-> node N   (for the closing arc N -> depot)
    # All other depot edges are HUGE so the depot only appears in those
    # two forced positions in the tour.
    #
    # Scale coordinates by 1e7 (matching `utils/lkh.py:write_tsplib`) so the
    # rounded Euclidean distances fit in a meaningful integer range.
    D = np.zeros((N + 1, N + 1), dtype=np.int64)
    scaled = np.round(coords * TSPLIB_SCALE).astype(np.int64)
    for i in range(N):
        for j in range(N):
            D[i, j] = _rounded_euclidean(scaled[i], scaled[j])
    # Row/column for the depot: only the two forced neighbours are cheap.
    for k in range(N):
        is_forced_neighbour = (k == 0) or (k == N - 1)
        D[N, k] = 0 if is_forced_neighbour else HUGE   # depot -> real
        D[k, N] = 0 if is_forced_neighbour else HUGE   # real -> depot
    D[N, N] = 0                                         # depot self-loop

    # --- .vrp ------------------------------------------------------------
    with open(vrp_path, "w") as f:
        f.write(f"NAME : {name}\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {N + 1}\n")
        f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")
        # 1-indexed, row-major.
        for i in range(1, N + 2):
            f.write(" ".join(str(int(D[i - 1, j - 1])) for j in range(1, N + 2)))
            f.write("\n")
        # Force the depot into the closing position of the tour so that
        # the resulting closed tour, minus the depot, is the SHPP path.
        f.write("FIXED_EDGES_SECTION\n")
        f.write(f"{DUMMY_ID} 1\n")    # depot -> start (1-indexed)
        f.write(f"{N} {DUMMY_ID}\n")  # end -> depot
        f.write("-1\n")
        f.write("EOF\n")

    # --- .par ------------------------------------------------------------
    with open(par_path, "w") as f:
        f.write(f"PROBLEM_FILE = {vrp_path}\n")
        f.write(f"OUTPUT_TOUR_FILE = {tour_path}\n")
        f.write("RUNS = 1\n")
        f.write("SEED = 1234\n")
        f.write("TRACE_LEVEL = 0\n")
        f.write("MAX_TRIALS = 10000\n")
        f.write("MAX_CANDIDATES = 5\n")

    return vrp_path, par_path, tour_path, log_path


def _lkh_worker(args):
    """Solve a single SHPP instance with LKH-3.

    Returns (instance_id, dist_label, cost, runtime_sec, path_or_None).
    `cost` is the open-path SHPP cost; `path_or_None` is the 0-indexed
    node sequence (length N) excluding the dummy depot.
    """
    instance_id, dist_label, coords, working_dir, lkh_bin = args
    os.makedirs(working_dir, exist_ok=True)
    vrp_path, par_path, tour_path, log_path = write_shpp_lkh_files(
        working_dir, instance_id, coords
    )

    t0 = time.time()
    try:
        with open(log_path, "w") as flog:
            subprocess.check_call(
                [lkh_bin, par_path], stdout=flog, stderr=flog
            )
        runtime = time.time() - t0
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"[lkh_worker] instance {instance_id} LKH-3 failed: {exc}\n"
        )
        return instance_id, dist_label, float("nan"), 0.0, None

    # Read the closed tour (length N+1, 1-indexed).
    tour_1idx = []
    with open(tour_path, "r") as f:
        in_tour = False
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_tour = True
                continue
            if in_tour:
                if line == "-1" or line == "EOF":
                    break
                if line:
                    tour_1idx.append(int(line))

    if len(tour_1idx) != SHPP_SIZE + 1:
        sys.stderr.write(
            f"[lkh_worker] instance {instance_id}: tour length "
            f"{len(tour_1idx)} != {SHPP_SIZE + 1}\n"
        )
        return instance_id, dist_label, float("nan"), runtime, None

    # Strip the depot, leaving the SHPP path 1-indexed, then 0-indexed.
    path_1idx = [t for t in tour_1idx if t != DUMMY_ID]
    if len(path_1idx) != SHPP_SIZE or set(path_1idx) != set(range(1, SHPP_SIZE + 1)):
        sys.stderr.write(
            f"[lkh_worker] instance {instance_id}: depot-stripped path "
            f"is malformed\n"
        )
        return instance_id, dist_label, float("nan"), runtime, None
    path_0idx = [t - 1 for t in path_1idx]

    cost = shpp_cost(coords, path_0idx)
    return instance_id, dist_label, cost, runtime, path_0idx


def run_lkh_parallel(coords, dist_labels, lkh_bin, work_dir, cpus):
    """Run LKH-3 over all instances in parallel via a process pool."""
    tasks = [
        (i, dist_labels[i], coords[i],
         os.path.join(work_dir, f"inst_{i:06d}"), lkh_bin)
        for i in range(coords.shape[0])
    ]
    results = [None] * coords.shape[0]
    with mp.get_context("spawn").Pool(processes=cpus) as pool:
        for r in tqdm(
            pool.imap_unordered(_lkh_worker, tasks),
            total=len(tasks),
            desc="LKH-3",
        ):
            i, dist, cost, rt, path = r
            results[i] = (dist, cost, rt, path)
    return results


# --- Reviser inference ----------------------------------------------------

def run_revisor_inference(model, coords, device, batch_size, decode_strategy="greedy"):
    """Run the reviser over `coords`, returning per-instance cost/path/time.

    The revisor's native `LOCAL.get_costs` already returns the open-path
    L2 cost (no closing edge), so `cost` from the model is the SHPP cost
    directly.
    """
    model.set_decode_type(decode_strategy)
    N = coords.shape[1]
    all_costs = np.empty(coords.shape[0], dtype=np.float64)
    all_paths = [None] * coords.shape[0]

    t0 = time.time()
    with torch.no_grad():
        for start in tqdm(
            range(0, coords.shape[0], batch_size),
            desc="Revisor",
        ):
            end = min(start + batch_size, coords.shape[0])
            batch = torch.from_numpy(coords[start:end]).float().to(device)
            # forward(..., return_pi=True) returns (cost, pi, cost2, pi2_flipped).
            cost, pi, cost2, _pi2 = model(batch, return_pi=True)
            # Use the cheaper of the two decode directions.
            cost_b = torch.stack([cost, cost2], dim=1)             # (B, 2)
            best_idx = cost_b.argmin(dim=1)                        # (B,)
            best_cost = cost_b.gather(1, best_idx.unsqueeze(1)).squeeze(1)
            pi_best = torch.where(
                best_idx.unsqueeze(1).eq(0),
                pi,
                _pi2,
            )                                                     # (B, N)
            for j in range(end - start):
                all_costs[start + j] = float(best_cost[j].item())
                all_paths[start + j] = pi_best[j].cpu().tolist()
    runtime = time.time() - t0

    return all_costs, all_paths, runtime


# --- CLI ------------------------------------------------------------------

def build_opts():
    p = argparse.ArgumentParser(
        description="Compare reviser-100 against LKH-3 on SHPP instances."
    )
    p.add_argument("--val_size", type=int, default=10000,
                   help="Total SHPP instances (split evenly across unit/scale).")
    p.add_argument("--n_unit", type=int, default=-1,
                   help="Override count of unit-distribution instances.")
    p.add_argument("--n_scale", type=int, default=-1,
                   help="Override count of scale-distribution instances.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval_batch_size", type=int, default=256,
                   help="Revisor GPU batch size.")
    p.add_argument("--cpus", type=int, default=max(1, os.cpu_count() or 1),
                   help="Worker processes for parallel LKH-3 calls.")
    p.add_argument("--decode_strategy", choices=["greedy", "sampling"],
                   default="greedy")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--lkh_bin", default=LKH_BIN_REL,
                   help="Path to the LKH-3 executable.")
    p.add_argument("--out_dir", default="results/shpp_eval")
    p.add_argument("--work_dir", default="results/shpp_lkh",
                   help="Per-instance LKH-3 working directories.")
    p.add_argument("--no_progress_bar", action="store_true")
    p.add_argument("--disable_lkh", action="store_true",
                   help="Skip LKH-3 (revisor-only dry run).")
    p.add_argument("--disable_revisor", action="store_true",
                   help="Skip revisor (LKH-3-only dry run).")
    return p.parse_args()


# --- Aggregation ----------------------------------------------------------

def _ci(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    return float(2.0 * np.std(x) / math.sqrt(x.size))


def _summarize(rev_costs, lkh_costs, rev_total, lkh_total):
    rev = np.asarray(rev_costs, dtype=np.float64)
    lkh = np.asarray(lkh_costs, dtype=np.float64)
    finite = np.isfinite(rev) & np.isfinite(lkh) & (lkh > 0)
    gap = np.where(finite, 100.0 * (rev - lkh) / lkh, np.nan)
    return {
        "n": int(rev.size),
        "n_valid": int(finite.sum()),
        "revisor_mean": float(np.nanmean(rev)),
        "revisor_std": float(np.nanstd(rev)),
        "revisor_ci95": _ci(rev[np.isfinite(rev)]),
        "lkh_mean": float(np.nanmean(lkh)),
        "lkh_std": float(np.nanstd(lkh)),
        "lkh_ci95": _ci(lkh[np.isfinite(lkh)]),
        "gap_mean_pct": float(np.nanmean(gap)),
        "gap_std_pct": float(np.nanstd(gap)),
        "win_rate_pct": float(100.0 * np.nanmean(rev < lkh)),
        "revisor_total_sec": float(rev_total),
        "lkh_total_sec": float(lkh_total),
    }


# --- Main -----------------------------------------------------------------

def main():
    opts = build_opts()
    for k, v in vars(opts).items():
        print(f"  {k} = {v}")

    # Resolve relative paths against the project root (script directory).
    proj_root = _HERE
    lkh_bin = opts.lkh_bin
    if not os.path.isabs(lkh_bin):
        lkh_bin = os.path.join(proj_root, lkh_bin)
    reviser_path = REVISER_PATH_REL
    if not os.path.isabs(reviser_path):
        reviser_path = os.path.join(proj_root, reviser_path)

    if not os.path.isfile(lkh_bin) or not os.access(lkh_bin, os.X_OK):
        sys.stderr.write(f"ERROR: LKH-3 binary not found or not executable: {lkh_bin}\n")
        sys.exit(1)
    if not opts.disable_revisor and not os.path.isfile(reviser_path):
        sys.stderr.write(f"ERROR: reviser checkpoint not found: {reviser_path}\n")
        sys.exit(1)

    out_dir = opts.out_dir if os.path.isabs(opts.out_dir) else os.path.join(proj_root, opts.out_dir)
    work_dir = opts.work_dir if os.path.isabs(opts.work_dir) else os.path.join(proj_root, opts.work_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # Resolve n_unit / n_scale.
    n_unit = opts.n_unit if opts.n_unit >= 0 else opts.val_size // 2
    n_scale = opts.n_scale if opts.n_scale >= 0 else opts.val_size - n_unit
    assert n_unit + n_scale == opts.val_size, (
        f"n_unit ({n_unit}) + n_scale ({n_scale}) != val_size ({opts.val_size})"
    )

    coords, dist_labels = generate_instances(n_unit, n_scale, SHPP_SIZE, opts.seed)
    M = coords.shape[0]
    print(f"Generated {M} SHPP instances of size {SHPP_SIZE} "
          f"({n_unit} unit, {n_scale} scale).")

    # --- Phase A: revisor -----------------------------------------------
    rev_costs = np.full(M, np.nan, dtype=np.float64)
    rev_paths = [None] * M
    rev_total = 0.0
    if not opts.disable_revisor:
        device = torch.device(
            f"cuda:{opts.device_id}" if (torch.cuda.is_available() and not opts.no_cuda)
            else "cpu"
        )
        model, _ = load_model(reviser_path, is_local=True)
        model.to(device)
        model.eval()
        rev_costs, rev_paths, rev_total = run_revisor_inference(
            model, coords, device, opts.eval_batch_size, opts.decode_strategy
        )

    # --- Phase B: LKH-3 -------------------------------------------------
    lkh_costs = np.full(M, np.nan, dtype=np.float64)
    lkh_paths = [None] * M
    lkh_runtimes = np.zeros(M, dtype=np.float64)
    lkh_total = 0.0
    if not opts.disable_lkh:
        lkh_results = run_lkh_parallel(
            coords, dist_labels, lkh_bin, work_dir, opts.cpus
        )
        for i, (dist, cost, rt, path) in enumerate(lkh_results):
            lkh_costs[i] = cost
            lkh_runtimes[i] = rt
            lkh_paths[i] = path
        lkh_total = float(lkh_runtimes.sum())

    # --- Per-instance records ------------------------------------------
    tag = f"n{M}_b{opts.eval_batch_size}_dec-{opts.decode_strategy}_seed{opts.seed}"
    jsonl_path = os.path.join(out_dir, f"shpp_{tag}.jsonl")
    summary_path = os.path.join(out_dir, f"shpp_{tag}.summary.json")

    records = []
    for i in range(M):
        rec = {
            "instance_id": i,
            "distribution": dist_labels[i],
            "revisor_cost": float(rev_costs[i]) if np.isfinite(rev_costs[i]) else None,
            "lkh_cost": float(lkh_costs[i]) if np.isfinite(lkh_costs[i]) else None,
            "revisor_time_sec": None,  # not measured per-instance (amortised)
            "lkh_time_sec": float(lkh_runtimes[i]),
            "revisor_path": rev_paths[i],
            "lkh_path": lkh_paths[i],
        }
        if (rec["revisor_cost"] is not None and rec["lkh_cost"] is not None
                and rec["lkh_cost"] > 0):
            rec["gap_pct"] = 100.0 * (rec["revisor_cost"] - rec["lkh_cost"]) / rec["lkh_cost"]
        else:
            rec["gap_pct"] = None
        records.append(rec)

    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote per-instance results to {jsonl_path}")

    # --- Summary --------------------------------------------------------
    overall = _summarize(rev_costs, lkh_costs, rev_total, lkh_total)

    by_dist = {}
    for dist in ("unit", "scale"):
        idx = [i for i, d in enumerate(dist_labels) if d == dist]
        if not idx:
            continue
        by_dist[dist] = _summarize(
            rev_costs[idx], lkh_costs[idx], 0.0, float(lkh_runtimes[idx].sum())
        )

    summary = {
        "config": {
            "shpp_size": SHPP_SIZE,
            "n": M,
            "n_unit": n_unit,
            "n_scale": n_scale,
            "seed": opts.seed,
            "decode_strategy": opts.decode_strategy,
            "eval_batch_size": opts.eval_batch_size,
            "cpus": opts.cpus,
            "lkh_bin": os.path.relpath(lkh_bin, proj_root),
            "reviser_path": os.path.relpath(reviser_path, proj_root),
        },
        "overall": overall,
        "by_distribution": by_dist,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {summary_path}")

    # --- Pretty table ---------------------------------------------------
    print()
    print("=" * 72)
    print(f"SHPP-{SHPP_SIZE}  |  n={M}  |  revisor vs. LKH-3")
    print("=" * 72)
    hdr = (
        f"{'split':<10}  {'revisor':>14}  {'LKH-3':>14}  "
        f"{'gap %':>10}  {'win%':>8}  {'LKH sec':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    def _row(label, s):
        if s["n_valid"] == 0:
            print(f"{label:<10}  (no data)")
            return
        print(
            f"{label:<10}  "
            f"{s['revisor_mean']:>10.4f} ±{s['revisor_ci95']:<4.4f}  "
            f"{s['lkh_mean']:>10.4f} ±{s['lkh_ci95']:<4.4f}  "
            f"{s['gap_mean_pct']:>+10.3f}  "
            f"{s['win_rate_pct']:>8.2f}  "
            f"{s['lkh_total_sec']:>10.2f}"
        )

    _row("overall", overall)
    for dist in ("unit", "scale"):
        if dist in by_dist:
            _row(dist, by_dist[dist])
    print("=" * 72)


if __name__ == "__main__":
    main()
