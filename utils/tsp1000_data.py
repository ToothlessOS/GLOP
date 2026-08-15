"""Helpers for the TSP1000 datasets shipped under ``data/tsp1000/``.

The TSP1000 instances live in five distribution flavours and a separate
solution directory::

    data/tsp1000/tsp1000_<distribution>.txt         (200 instances, 1000 cities each)
    data/tsp1000_sol/tsp1000_<distribution>         (200 optimal tours + LKH cost/runtime)

GLOP's existing TSP pipeline (``TSPDataset.__init__`` at
``problems/tsp/problem_tsp.py``) only knows how to load ``.pkl`` files of
``(N, 2)`` float arrays. This module bridges that gap:

  * parses the new ``.txt`` instances and solution files;
  * materialises the instances into a ``.pkl`` matching the existing
    contract so the unmodified ``main._eval_dataset`` flow can consume
    them;
  * caches the converted pkl on disk (mtime-aware) for reuse across runs;
  * computes gap-to-optimum percentages so the eval script can report
    LKH-relative performance.

None of the GLOP pipeline files are touched.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from typing import List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Distribution flavours shipped under ``data/tsp1000/`` and
#: ``data/tsp1000_sol/``.
DISTRIBUTIONS: Tuple[str, ...] = (
    "uniform",
    "clustered1",
    "clustered2",
    "explosion",
    "implosion",
)

DEFAULT_DATA_DIR: str = "data/tsp1000"
DEFAULT_SOL_DIR: str = "data/tsp1000_sol"
DEFAULT_CACHE_DIR: str = "data/tsp1000_pkl"

#: Number of TSP instances per ``.txt`` file.
N_INSTANCES_PER_FILE: int = 200

#: Number of cities per TSP instance. The pkl format carries this implicitly
#: in the row shape, but the loader validates it.
N_NODES: int = 1000


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def load_tsp1000_instances(
    path: str,
    num_samples: Optional[int] = None,
    offset: int = 0,
) -> List[torch.Tensor]:
    """Read one ``tsp1000_*.txt`` file.

    Each line of the file is one TSP instance: ``N_NODES`` space-separated
    ``"x,y"`` tokens. Coordinates are in ``[0, 1]`` (already normalised).

    Args:
        path: Path to the ``.txt`` file.
        num_samples: How many instances to read starting at ``offset``.
            ``None`` (default) reads all instances in the file.
        offset: Number of leading instances to skip (matches the
            ``TSPDataset`` semantics).

    Returns:
        List of float tensors, each of shape ``(N_NODES, 2)``. The list
        length is ``num_samples`` (or ``N_INSTANCES_PER_FILE - offset``
        when ``num_samples is None``).
    """
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if num_samples is not None and num_samples < 0:
        raise ValueError(f"num_samples must be >= 0 when set, got {num_samples}")

    stop: Optional[int] = None if num_samples is None else offset + num_samples
    rows: List[torch.Tensor] = []
    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            if line_idx < offset:
                continue
            if stop is not None and line_idx >= stop:
                break
            tokens = line.split()
            if len(tokens) != N_NODES:
                raise ValueError(
                    f"{path}: line {line_idx + 1} has {len(tokens)} tokens, "
                    f"expected {N_NODES}"
                )
            row: List[List[float]] = []
            for tok in tokens:
                x_str, y_str = tok.split(",")
                row.append([float(x_str), float(y_str)])
            rows.append(torch.tensor(row, dtype=torch.float32))
    return rows


def load_tsp1000_optimal_costs(
    path: str,
    num_samples: Optional[int] = None,
    offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read one ``tsp1000_sol/*`` file.

    Each line is ``id0,id1,...,id999 <cost> <runtime>``: ``N_NODES``
    comma-separated 0-indexed tour IDs followed by two whitespace-
    separated floats. The tour IDs are discarded — the gap-to-optimum
    metric only needs the cost.

    The order of the two floats matters: the first (smaller, ~23 for a
    typical uniform TSP1000 instance) is the LKH-3 closed-tour cost;
    the second (~280) is the LKH wall-clock in seconds.

    Args:
        path: Path to the solution file (no extension).
        num_samples: How many lines to read starting at ``offset``.
            ``None`` (default) reads all lines.
        offset: Number of leading lines to skip.

    Returns:
        ``(costs, runtimes)`` — both 1-D ``torch.float32`` tensors of
        length ``num_samples`` (or ``N_INSTANCES_PER_FILE - offset``).
    """
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if num_samples is not None and num_samples < 0:
        raise ValueError(f"num_samples must be >= 0 when set, got {num_samples}")

    stop: Optional[int] = None if num_samples is None else offset + num_samples
    costs: List[float] = []
    runtimes: List[float] = []
    with open(path, "r") as f:
        for line_idx, line in enumerate(f):
            if line_idx < offset:
                continue
            if stop is not None and line_idx >= stop:
                break
            # rsplit(' ', 2) peels the last two whitespace-delimited floats
            # regardless of whether the tour-ID section contains any
            # whitespace artefacts.
            parts = line.rsplit(" ", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"{path}: line {line_idx + 1} could not be split into "
                    f"tour + 2 floats; got {len(parts)} parts"
                )
            cost_str, runtime_str = parts[1], parts[2]
            costs.append(float(cost_str))
            runtimes.append(float(runtime_str))
    return torch.tensor(costs, dtype=torch.float32), torch.tensor(runtimes, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Pkl cache
# ---------------------------------------------------------------------------


def tsp1000_txt_to_pkl_list(
    txt_path: str,
    num_samples: Optional[int] = None,
    offset: int = 0,
) -> List[List[List[float]]]:
    """Materialise the TSP1000 txt as a list-of-lists compatible with the
    GLOP pkl format.

    Each row is a Python ``list`` of ``[x, y]`` pairs — pickle-safe and
    identical in shape to the contents of ``data/tsp/tsp1000_test.pkl``.
    """
    tensors = load_tsp1000_instances(txt_path, num_samples=num_samples, offset=offset)
    return [row.tolist() for row in tensors]


def _pkl_is_fresh(pkl_path: str, txt_path: str) -> bool:
    """Return True iff the pkl exists and is at least as recent as the txt."""
    if not os.path.exists(pkl_path):
        return False
    if not os.path.exists(txt_path):
        # txt missing -> pkl cannot be validated as fresh; let the caller fail
        # downstream when it tries to write/overwrite.
        return False
    return os.path.getmtime(pkl_path) >= os.path.getmtime(txt_path)


def write_tsp1000_pkl_cache(
    txt_path: str,
    pkl_path: str,
    num_samples: Optional[int] = None,
    offset: int = 0,
    overwrite: bool = False,
) -> str:
    """Convert the ``.txt`` instances to a GLOP-compatible ``.pkl``.

    Skips the write if the pkl already exists, is newer than the txt, and
    ``overwrite`` is ``False``. ``mkdir -p`` the parent on demand.

    Returns the absolute ``pkl_path``.
    """
    if not overwrite and _pkl_is_fresh(pkl_path, txt_path):
        return os.path.abspath(pkl_path)

    parent = os.path.dirname(os.path.abspath(pkl_path))
    os.makedirs(parent, exist_ok=True)

    data = tsp1000_txt_to_pkl_list(txt_path, num_samples=num_samples, offset=offset)
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return os.path.abspath(pkl_path)


def _validate_distribution(distribution: str) -> None:
    if distribution not in DISTRIBUTIONS:
        raise ValueError(
            f"unknown distribution {distribution!r}; expected one of {DISTRIBUTIONS}"
        )


def _validate_slice(offset: int, num_samples: Optional[int]) -> int:
    """Return the resolved num_samples; raise if out of range."""
    n = N_INSTANCES_PER_FILE if num_samples is None else num_samples
    if n < 0:
        raise ValueError(f"num_samples must be >= 0, got {n}")
    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if offset + n > N_INSTANCES_PER_FILE:
        raise ValueError(
            f"offset ({offset}) + num_samples ({n}) = {offset + n} exceeds "
            f"N_INSTANCES_PER_FILE ({N_INSTANCES_PER_FILE})"
        )
    return n


def resolve_tsp1000_pkl(
    distribution: str,
    data_dir: str = DEFAULT_DATA_DIR,
    cache_dir: str = DEFAULT_CACHE_DIR,
    num_samples: Optional[int] = None,
    offset: int = 0,
    use_cache: bool = True,
) -> str:
    """Validate inputs and return a ``.pkl`` path ready for ``TSPDataset``.

    The cache filename encodes ``(distribution, offset, num_samples)`` so
    different slices don't collide::

        <cache_dir>/tsp1000_<distribution>_<num_samples>_<offset>.pkl

    With ``use_cache=False`` (or when the cache is absent) the file is
    always (re)written. mtime-aware skip happens only when both files
    exist and ``use_cache`` is True.

    Returns:
        Absolute path to a ``.pkl`` file containing a list of ``(N, 2)``
        float arrays.
    """
    _validate_distribution(distribution)
    n = _validate_slice(offset, num_samples)

    txt_path = os.path.join(data_dir, f"tsp1000_{distribution}.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"tsp1000 instance file not found: {txt_path}")

    pkl_path = os.path.join(cache_dir, f"tsp1000_{distribution}_{n}_{offset}.pkl")
    if not use_cache:
        # When caching is disabled, write into a unique tempfile so the user
        # pays no cost on subsequent runs and the workspace stays clean.
        fd, tmp = tempfile.mkstemp(
            prefix=f"tsp1000_{distribution}_{n}_{offset}_", suffix=".pkl"
        )
        os.close(fd)
        write_tsp1000_pkl_cache(
            txt_path, tmp, num_samples=num_samples, offset=offset, overwrite=True
        )
        return tmp

    return write_tsp1000_pkl_cache(
        txt_path, pkl_path, num_samples=num_samples, offset=offset, overwrite=False
    )


# ---------------------------------------------------------------------------
# Gap-to-optimum helpers
# ---------------------------------------------------------------------------


def gap_to_optimum_pct(
    predicted: torch.Tensor,
    optimal: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-instance gap to optimum in percent: ``100 * (p - o) / o``.

    Both inputs must be on the same device and have the same shape (a 1-D
    tensor of per-instance costs).
    """
    if predicted.shape != optimal.shape:
        raise ValueError(
            f"shape mismatch: predicted {tuple(predicted.shape)} vs "
            f"optimal {tuple(optimal.shape)}"
        )
    opt = optimal.to(predicted.device).to(predicted.dtype)
    # Avoid divide-by-zero on degenerate inputs (empty tensor is fine; only
    # strictly zero costs would crash).
    safe_opt = torch.where(opt.abs() < eps, torch.full_like(opt, eps), opt)
    return 100.0 * (predicted - optimal.to(predicted.device)) / safe_opt


def summarise_distribution_costs(
    costs_pred: torch.Tensor,
    costs_opt: torch.Tensor,
) -> dict:
    """Aggregate mean/median/std/min/max gap% plus mean predicted/optimal cost.

    Inputs are 1-D tensors on the same shape (per-instance). Returns a dict
    with both scalar floats (for printing) and the raw per-instance gap
    tensor (for plotting / further analysis).
    """
    if costs_pred.shape != costs_opt.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(costs_pred.shape)} vs "
            f"opt {tuple(costs_opt.shape)}"
        )
    gap = gap_to_optimum_pct(costs_pred, costs_opt)
    return {
        "gap_mean_pct": gap.mean().item(),
        "gap_median_pct": gap.median().item(),
        "gap_std_pct": gap.std(unbiased=False).item() if gap.numel() > 1 else 0.0,
        "gap_min_pct": gap.min().item(),
        "gap_max_pct": gap.max().item(),
        "mean_pred_cost": costs_pred.mean().item(),
        "mean_opt_cost": costs_opt.mean().item(),
        "n": int(costs_pred.numel()),
        "gap_tensor": gap.detach().cpu(),
    }


__all__ = [
    "DISTRIBUTIONS",
    "DEFAULT_DATA_DIR",
    "DEFAULT_SOL_DIR",
    "DEFAULT_CACHE_DIR",
    "N_INSTANCES_PER_FILE",
    "N_NODES",
    "load_tsp1000_instances",
    "load_tsp1000_optimal_costs",
    "tsp1000_txt_to_pkl_list",
    "write_tsp1000_pkl_cache",
    "resolve_tsp1000_pkl",
    "gap_to_optimum_pct",
    "summarise_distribution_costs",
]