"""Run LKH-3 on a set of TSP instances and check convex-hull and no-self-intersection
properties of the produced tours using `utils.diagnosis`.

The script:
  1. Compiles the LKH-3 source in `LKH3_eval/LKH-3.0.14/` if the `LKH` binary
     is missing (idempotent — re-runs skip this).
  2. Iterates over one or more `.pkl` TSP datasets (each entry is an (N, 2)
     float array of node coordinates).
  3. For each instance, writes a TSPLIB EUC_2D `.tsp` + matching `.par`,
     invokes the LKH-3 binary, parses the tour, and runs both diagnosis
     checks from `utils.diagnosis` on the permuted coordinates (i.e. the
     coordinates in LKH's visitation order, NOT the original pkl order).
  4. Prints a per-dataset summary plus a grand-total footer and writes a
     `summary.json` sidecar with the per-instance records.

Run from the repo root:
    python LKH3_eval/run_diagnosis.py
    python LKH3_eval/run_diagnosis.py --datasets data/tsp/tsp20_test.pkl --max-instances 3 --verbose
"""

import argparse
import datetime
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

# Make the project root importable when this file is run directly.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.diagnosis import check_convex_hull, check_no_self_intersection  # noqa: E402
from utils.lkh import (  # noqa: E402
    calc_tsp_length,
    read_tsplib,
    write_lkh_par,
    write_tsplib,
)
from problems.tsp.problem_tsp import TSP  # noqa: E402


LKH_DIR = "LKH3_eval/LKH-3.0.14"
LKH_BIN_NAME = "LKH"

DEFAULT_DATASETS = [
    "data/tsp/tsp20_test.pkl",
    "data/tsp/tsp50_test.pkl",
    "data/tsp/tsp100_test.pkl",
    "data/tsp/tsp200_test.pkl",
    "data/tsp/tsp500_test.pkl",
    "data/tsp/tsp1000_test.pkl",
    "data/tsp/tsp10000_test.pkl",
    "data/tsp/tsp100000_test.pkl",
    "data/tsp/tsplib49.pkl",
]

COST_SCALE = 10_000_000  # utils.lkh.write_tsplib scales coords by this factor


# ---------------------------------------------------------------------------
# Build step
# ---------------------------------------------------------------------------

def build_lkh(lkh_dir: str) -> str:
    """Compile LKH-3 if the binary is missing. Returns absolute binary path."""
    bin_path = os.path.abspath(os.path.join(lkh_dir, LKH_BIN_NAME))
    if os.path.isfile(bin_path) and os.access(bin_path, os.X_OK):
        return bin_path
    if not os.path.isdir(lkh_dir):
        raise FileNotFoundError(f"LKH source dir not found: {lkh_dir}")
    makefile = os.path.join(lkh_dir, "Makefile")
    if not os.path.isfile(makefile):
        raise FileNotFoundError(f"LKH Makefile not found: {makefile}")
    print(f"[build] {bin_path} not found; running `make -j` in {lkh_dir}")
    subprocess.check_call(["make", "-j"], cwd=lkh_dir)
    if not (os.path.isfile(bin_path) and os.access(bin_path, os.X_OK)):
        raise RuntimeError(
            f"LKH build did not produce an executable at {bin_path}"
        )
    print(f"[build] OK: {bin_path}")
    return bin_path


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_locs(path: str, max_n: int) -> list:
    """Load up to `max_n` instances from a `.pkl` file. Each instance is an
    (N, 2) float ndarray. Confirmed format: top-level `list` of 2-D lists."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(
            f"{path}: expected a list of (N, 2) instances; got {type(data).__name__}"
        )
    out = []
    for i, row in enumerate(data):
        if i >= max_n:
            break
        arr = np.asarray(row, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"{path}: instance {i} has shape {arr.shape}; expected (N, 2)"
            )
        out.append(arr)
    return out


# ---------------------------------------------------------------------------
# LKH invocation
# ---------------------------------------------------------------------------

def write_problem_pair(
    workdir: str, name: str, loc: np.ndarray, opts: argparse.Namespace
) -> tuple:
    """Write a TSPLIB `.tsp` + matching `.par` into `workdir`. The `.par`
    references `PROBLEM_FILE` and `OUTPUT_TOUR_FILE` by bare basename so the
    LKH binary can find them when invoked with `cwd=workdir`. `opts` controls
    MAX_TRIALS and SEED forwarded to the LKH run."""
    os.makedirs(workdir, exist_ok=True)
    tsp_path = os.path.join(workdir, f"{name}.tsp")
    par_path = os.path.join(workdir, f"{name}.par")
    tour_path = os.path.join(workdir, f"{name}.tour")
    write_tsplib(tsp_path, loc, name=name)
    write_lkh_par(
        par_path,
        {
            "PROBLEM_FILE": f"{name}.tsp",
            "OUTPUT_TOUR_FILE": f"{name}.tour",
            "MAX_TRIALS": opts.max_trials,
            "SEED": opts.seed,
        },
    )
    return tsp_path, par_path, tour_path


def run_lkh_once(
    binary: str,
    workdir: str,
    par_path: str,
    tour_path: str,
    timeout_s: int,
) -> tuple:
    """Run LKH-3 on the given .par file. Returns (tour, lkh_cost, wall_s, log_path).

    tour: 0-indexed Python list of node indices.
    lkh_cost: integer cost parsed from LKH stdout (`Cost.min = <int>`), or None.
    log_path: path to a file containing captured stdout+stderr.
    """
    log_path = os.path.splitext(par_path)[0] + ".log"
    par_basename = os.path.basename(par_path)
    start = time.time()
    try:
        proc = subprocess.run(
            [binary, par_basename],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as e:
        with open(log_path, "w") as f:
            f.write(f"--- TIMEOUT after {timeout_s}s ---\n")
            # When subprocess.run times out, e.stdout/e.stderr are still
            # raw bytes even with text=True (the decoder runs after the
            # timeout is raised). Decode to str so file write succeeds.
            if e.stdout:
                out = (
                    e.stdout.decode("utf-8", errors="replace")
                    if isinstance(e.stdout, bytes)
                    else e.stdout
                )
                f.write("--- STDOUT (partial) ---\n" + out)
            if e.stderr:
                err = (
                    e.stderr.decode("utf-8", errors="replace")
                    if isinstance(e.stderr, bytes)
                    else e.stderr
                )
                f.write("--- STDERR (partial) ---\n" + err)
        raise
    wall = time.time() - start
    # Normalize stdout/stderr to str (text=True should handle this, but
    # be defensive in case of a Python quirk that returns bytes).
    stdout_str = (
        proc.stdout.decode("utf-8", errors="replace")
        if isinstance(proc.stdout, bytes)
        else (proc.stdout or "")
    )
    stderr_str = (
        proc.stderr.decode("utf-8", errors="replace")
        if isinstance(proc.stderr, bytes)
        else (proc.stderr or "")
    )
    with open(log_path, "w") as f:
        f.write("--- STDOUT ---\n" + stdout_str)
        f.write("\n--- STDERR ---\n" + stderr_str)
    if proc.returncode != 0:
        raise RuntimeError(
            f"LKH-3 failed (returncode={proc.returncode}); see {log_path}"
        )
    m = re.search(r"Cost\.min\s*=\s*(\d+)", proc.stdout or "")
    lkh_cost = int(m.group(1)) if m else None
    if not os.path.isfile(tour_path):
        raise RuntimeError(
            f"LKH-3 produced no tour file at {tour_path}; see {log_path}"
        )
    tour = read_tsplib(tour_path)
    return tour, lkh_cost, wall, log_path


# ---------------------------------------------------------------------------
# Per-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_one_instance(
    binary: str,
    workdir: str,
    dataset_basename: str,
    instance_idx: int,
    loc: np.ndarray,
    opts: argparse.Namespace,
) -> dict:
    """Run LKH-3 on one instance, then check convex-hull and no-intersection
    properties. Returns a per-instance record dict."""
    name = f"{dataset_basename}_inst{instance_idx:04d}"
    n = int(loc.shape[0])
    record: dict[str, Any] = {
        "dataset": dataset_basename,
        "instance": instance_idx,
        "N": n,
        "name": name,
    }
    try:
        _, par_path, tour_path = write_problem_pair(workdir, name, loc, opts)
        tour, lkh_cost, wall, log_path = run_lkh_once(
            binary, workdir, par_path, tour_path, opts.timeout
        )
    except Exception as e:
        record.update(
            {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }
        )
        return record
    finally:
        if not opts.keep_tempfiles:
            for ext in (".tsp", ".par", ".tour"):
                p = os.path.join(workdir, f"{name}{ext}")
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            # Keep the .log if LKH failed, so the user can debug.
            lp = os.path.join(workdir, f"{name}.log")
            if "error" in record and os.path.isfile(lp):
                pass  # leave it
            elif os.path.isfile(lp):
                try:
                    os.remove(lp)
                except OSError:
                    pass

    # Verify tour length matches the instance size.
    if len(tour) != n:
        record.update(
            {
                "ok": False,
                "error": f"tour length {len(tour)} != N {n}",
                "lkh_cost": lkh_cost,
                "wall_s": wall,
            }
        )
        return record

    # Cost computations.
    our_cost = float(calc_tsp_length(loc, tour))  # raw coords
    coords_t = torch.tensor(loc, dtype=torch.float32).unsqueeze(0)  # (1, N, 2)
    pi = torch.tensor(tour, dtype=torch.long).view(1, -1)  # (1, N)
    project_cost, _ = TSP.get_costs(coords_t, pi)
    project_cost = float(project_cost.item())
    # LKH-3 cost is in 1e7-scaled integer units; ours is in raw-coord float units.
    lkh_cost_unscaled = lkh_cost / COST_SCALE if lkh_cost is not None else None
    cost_diff_lkh = (
        abs(lkh_cost_unscaled - our_cost) if lkh_cost_unscaled is not None else None
    )
    cost_diff_proj = abs(project_cost - our_cost)
    cost_ok = (
        cost_diff_lkh is not None
        and cost_diff_lkh < n * 1e-5
        and cost_diff_proj < 1e-4
    )

    # Tour-permuted coordinates for the diagnosis functions. The N axis is
    # interpreted as the tour sequence by both functions.
    tour_coords = coords_t.gather(1, pi.unsqueeze(-1).expand_as(coords_t))

    # Convex-hull check. check_convex_hull returns (consistency, tour_seq);
    # we only need the per-instance 0/1 consistency.
    hull_consistency, _ = check_convex_hull(tour_coords)
    hull_ok = bool(hull_consistency.item() == 1.0)

    # No-self-intersection check — O(B*N^2) memory, so skip above threshold.
    si_skipped = n > opts.si_max_n
    if si_skipped:
        has_inter = n_inter = frac = n_pairs = None
        si_ok = None
    else:
        has_inter, n_inter, frac, n_pairs, _ = check_no_self_intersection(
            tour_coords
        )
        has_inter = bool(has_inter.item())
        n_inter = int(n_inter.item())
        frac = float(frac.item())
        n_pairs = int(n_pairs.item())
        si_ok = (not has_inter)

    record.update(
        {
            "ok": True,
            "lkh_cost": lkh_cost,
            "lkh_cost_unscaled": lkh_cost_unscaled,
            "our_cost": our_cost,
            "project_cost": project_cost,
            "cost_diff_lkh": cost_diff_lkh,
            "cost_diff_proj": cost_diff_proj,
            "cost_ok": cost_ok,
            "hull_ok": hull_ok,
            "hull_consistency": float(hull_consistency.item()),
            "si_skipped": si_skipped,
            "si_ok": si_ok,
            "has_inter": has_inter,
            "n_inter": n_inter,
            "si_frac": frac,
            "si_n_pairs": n_pairs,
            "wall_s": wall,
        }
    )
    return record


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    binary: str, pkl_path: str, opts: argparse.Namespace
) -> dict:
    """Run LKH-3 + diagnosis on up to `--max-instances` instances of `pkl_path`."""
    dataset_basename = os.path.splitext(os.path.basename(pkl_path))[0]
    dataset_workdir = os.path.join(opts.workdir, dataset_basename)
    os.makedirs(dataset_workdir, exist_ok=True)
    try:
        instances = load_dataset_locs(pkl_path, opts.max_instances)
        n_total = len(instances)
        print(
            f"\n[dataset] {dataset_basename}: {n_total} instance(s) to evaluate"
        )
        records: list[dict] = []
        for i, loc in enumerate(instances):
            rec = evaluate_one_instance(
                binary, dataset_workdir, dataset_basename, i, loc, opts
            )
            records.append(rec)
            if opts.verbose:
                _print_instance_line(rec)
        agg = _aggregate(records)
        agg["dataset"] = dataset_basename
        agg["pkl_path"] = pkl_path
        agg["records"] = records
        return agg
    finally:
        if not opts.keep_tempfiles and os.path.isdir(dataset_workdir):
            try:
                shutil.rmtree(dataset_workdir)
            except OSError:
                pass


def _aggregate(records: list[dict]) -> dict:
    """Summarize a list of per-instance records into aggregate stats."""
    n_total = len(records)
    n_solved = sum(1 for r in records if r.get("ok"))
    n_failed = n_total - n_solved
    n_hull = sum(1 for r in records if r.get("hull_ok"))
    si_attempted = [r for r in records if r.get("si_ok") is not None]
    si_skipped = [r for r in records if r.get("si_skipped")]
    n_si = sum(1 for r in si_attempted if r.get("si_ok"))
    n_si_skipped = len(si_skipped)
    n_cost_ok = sum(1 for r in records if r.get("cost_ok"))
    our_costs = [r["our_cost"] for r in records if "our_cost" in r]
    lkh_costs = [r["lkh_cost"] for r in records if r.get("lkh_cost") is not None]
    walls = [r["wall_s"] for r in records if "wall_s" in r]
    return {
        "n_total": n_total,
        "n_solved": n_solved,
        "n_failed": n_failed,
        "n_hull": n_hull,
        "n_si": n_si,
        "n_si_attempted": len(si_attempted),
        "n_si_skipped": n_si_skipped,
        "n_cost_ok": n_cost_ok,
        "our_costs": our_costs,
        "lkh_costs": lkh_costs,
        "walls": walls,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_instance_line(rec: dict) -> None:
    name = rec.get("name", "?")
    n = rec.get("N", "?")
    if not rec.get("ok"):
        print(f"  [{name}] N={n}  FAILED: {rec.get('error', '?')}")
        return
    lkh = rec.get("lkh_cost")
    lkh_str = f"{lkh}" if lkh is not None else "?"
    our = rec.get("our_cost", float("nan"))
    proj = rec.get("project_cost", float("nan"))
    hull = "PASS" if rec.get("hull_ok") else "FAIL"
    if rec.get("si_skipped"):
        si = "SKIP"
    elif rec.get("si_ok"):
        si = "PASS"
    else:
        si = "FAIL"
    inter = f"{rec.get('n_inter', '?')}/{rec.get('si_n_pairs', '?')}"
    cost_ok = "OK" if rec.get("cost_ok") else "MISMATCH"
    wall = rec.get("wall_s", 0.0)
    print(
        f"  [{name}] N={n} lkh={lkh_str} our={our:.4f} proj={proj:.4f} "
        f"hull={hull} si={si} inter={inter} cost={cost_ok} wall={wall:.2f}s"
    )


def _fmt_costs(costs: list) -> str:
    if not costs:
        return "n/a"
    arr = np.asarray(costs, dtype=np.float64)
    return f"mean={arr.mean():.6g}, min={arr.min():.6g}, max={arr.max():.6g}"


def _print_dataset_block(agg: dict) -> None:
    name = agg["dataset"]
    n_total = agg["n_total"]
    n_total_str = (
        f"{n_total} (varying N)" if name == "tsplib49" else f"{n_total} instance(s)"
    )
    print(f"\n=== {name} ({n_total_str}) ===")
    print(
        f"  LKH-3 solved:           {agg['n_solved']}/{n_total} OK "
        f"({agg['n_failed']} failed)"
    )
    print(
        f"  Convex hull:            {agg['n_hull']}/{n_total} satisfy "
        f"({100.0 * agg['n_hull'] / max(1, n_total):.1f}%)"
    )
    if agg["n_si_skipped"] == n_total:
        print(f"  No self-intersection:   SKIPPED (all N > --si-max-n)")
    elif agg["n_si_skipped"] > 0:
        print(
            f"  No self-intersection:   {agg['n_si']}/{agg['n_si_attempted']} satisfy "
            f"({agg['n_si_skipped']} skipped due to N > --si-max-n)"
        )
    else:
        print(
            f"  No self-intersection:   {agg['n_si']}/{agg['n_si_attempted']} satisfy "
            f"(100.0%)"
        )
    if agg["lkh_costs"]:
        print(f"  LKH-reported cost:      {_fmt_costs(agg['lkh_costs'])}")
    if agg["our_costs"]:
        print(f"  Our computed cost:      {_fmt_costs(agg['our_costs'])}")
    if agg["walls"]:
        print(
            f"  Wall time (LKH):        total={sum(agg['walls']):.2f}s, "
            f"per-instance mean={np.mean(agg['walls']):.3f}s"
        )
    if agg["n_total"] > 0:
        print(f"  Cost consistency:       {agg['n_cost_ok']}/{n_total} OK")


def _print_grand_total(aggs: list, total_wall: float) -> None:
    n_total = sum(a["n_total"] for a in aggs)
    n_solved = sum(a["n_solved"] for a in aggs)
    n_failed = sum(a["n_failed"] for a in aggs)
    n_hull = sum(a["n_hull"] for a in aggs)
    n_si = sum(a["n_si"] for a in aggs)
    n_si_attempted = sum(a["n_si_attempted"] for a in aggs)
    n_si_skipped = sum(a["n_si_skipped"] for a in aggs)
    n_cost_ok = sum(a["n_cost_ok"] for a in aggs)
    print("\n" + "=" * 64)
    print(
        f"  Overall: {n_solved}/{n_total} solved ({n_failed} failed), "
        f"{n_hull}/{n_total} convex-hull OK,"
    )
    print(
        f"           {n_si}/{n_si_attempted} self-intersection OK "
        f"({n_si_skipped} skipped due to N > --si-max-n),"
    )
    print(
        f"           {n_cost_ok}/{n_total} cost-consistent. "
        f"Total wall: {total_wall:.1f}s."
    )
    print("=" * 64)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: list | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run LKH-3 on a set of TSP .pkl datasets and check convex-hull "
            "and no-self-intersection properties of the produced tours."
        )
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=f"Paths to .pkl TSP datasets (default: all 9 standard datasets).",
    )
    p.add_argument(
        "--max-instances",
        type=int,
        default=10,
        help="Cap on instances solved per dataset (default: 10).",
    )
    p.add_argument(
        "--max-trials",
        type=int,
        default=1,
        help="MAX_TRIALS value passed to LKH-3 via the .par file (default: 1).",
    )
    p.add_argument(
        "--si-max-n",
        type=int,
        default=2000,
        help=(
            "Run check_no_self_intersection only when N <= this; skip above "
            "(memory is O(N^2)). Default: 2000."
        ),
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-instance LKH-3 timeout in seconds (default: 600).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="SEED value passed to LKH-3 (default: 0).",
    )
    p.add_argument(
        "--workdir",
        type=str,
        default=None,
        help=(
            "Scratch dir for LKH-3 .tsp/.par/.tour/.log files. Default: "
            "results/lkh_diagnosis/<UTC-timestamp>/"
        ),
    )
    p.add_argument(
        "--keep-tempfiles",
        action="store_true",
        help="Keep .tsp/.par/.tour/.log files after the run.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print a line for every instance, not just per-dataset summaries.",
    )
    return p.parse_args(argv)


def main(argv: list | None = None) -> int:
    opts = parse_args(argv)
    # 1. Build / verify LKH-3.
    binary = build_lkh(LKH_DIR)
    # 2. Resolve workdir.
    if opts.workdir is None:
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        opts.workdir = os.path.join("results", "lkh_diagnosis", ts)
    os.makedirs(opts.workdir, exist_ok=True)
    print(f"[run] workdir: {opts.workdir}")
    # 3. Per-dataset evaluation.
    overall_start = time.time()
    aggs: list[dict] = []
    for pkl in opts.datasets:
        try:
            agg = evaluate_dataset(binary, pkl, opts)
            aggs.append(agg)
            _print_dataset_block(agg)
        except FileNotFoundError as e:
            print(f"[dataset] SKIP {pkl}: {e}")
        except Exception as e:
            print(f"[dataset] ERROR {pkl}: {type(e).__name__}: {e}")
    total_wall = time.time() - overall_start
    # 4. Grand-total report.
    if aggs:
        _print_grand_total(aggs, total_wall)
    # 5. Sidecar JSON with full per-instance records.
    summary_path = os.path.join(opts.workdir, "summary.json")
    serializable = []
    for agg in aggs:
        records = agg.pop("records", [])
        for r in records:
            serializable.append(r)
    try:
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "workdir": opts.workdir,
                    "total_wall_s": total_wall,
                    "options": {k: getattr(opts, k) for k in vars(opts)},
                    "instances": serializable,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"[run] summary written: {summary_path}")
    except Exception as e:
        print(f"[run] could not write summary: {e}")
    # 6. Exit code: non-zero if any instance failed.
    n_failed = sum(a["n_failed"] for a in aggs)
    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
