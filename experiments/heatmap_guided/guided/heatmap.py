"""AGFN heatmap loader and inference.

Mirrors `ref/AGFN/tsp/test_tsp.py:33-64` but exposes a clean Python API
that returns a dense (N, N) edge-probability heatmap for a single TSP
instance.

Conventions match the AGFN training distribution:
- coordinates in [0, 1]^2
- topk-by-distance sparse graph (k_sparse = N // 10 by default)
- sigmoid output via AGFN's `par_net_heu` -> reshape to (N, N)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

import torch
from torch import Tensor, nn

# AGFN ships a top-level `tsp/` package with `net.py` and `utils.py`.
# We load these as uniquely-named modules (`agfn_tsp_net`,
# `agfn_tsp_utils`) to avoid colliding with the GLOP `utils` package
# (which would otherwise shadow `utils.py` from `ref/AGFN/tsp/`).
# heatmap.py lives at experiments/heatmap_guided/guided/heatmap.py;
# the repo root is 3 directories up.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_AGFN_TSP = os.path.join(_REPO_ROOT, "ref", "AGFN", "tsp")
_AGFN_NET = os.path.join(_AGFN_TSP, "net.py")
_AGFN_UTILS = os.path.join(_AGFN_TSP, "utils.py")


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_agfn_net = _load_module("agfn_tsp_net", _AGFN_NET)
_agfn_utils = _load_module("agfn_tsp_utils", _AGFN_UTILS)
Net = _agfn_net.Net
gen_pyg_data = _agfn_utils.gen_pyg_data

EPS = 1e-10

# Default sparsity mirrors `ref/AGFN/tsp/test_tsp.py:84`:
# k_sparse defaults to N // 10 when the user does not pass one.
DEFAULT_K_SPARSE_DIVISOR = 10


def load_agfn(
    scale: int,
    device: str = "cpu",
    ckpt_root: Optional[str] = None,
) -> nn.Module:
    """Load a pretrained AGFN TSP-GFN for a given problem size.

    The on-disk `.pt` files are git-LFS pointers (~131 bytes) until
    `git lfs pull` has been run inside `ref/AGFN/`. We detect that case
    and raise a clear error pointing at the fix.
    """
    if ckpt_root is None:
        ckpt_root = os.path.join(_REPO_ROOT, "ref", "AGFN", "pretrained", "tsp")
    ckpt_path = os.path.join(ckpt_root, f"gan_tsp_{scale}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"AGFN checkpoint not found at {ckpt_path}. "
            f"Available scales: 100, 200, 500, 1000."
        )
    # Detect LFS pointer: a real .pt is hundreds of KB to tens of MB;
    # an LFS pointer textfile is ~131 bytes.
    if os.path.getsize(ckpt_path) < 1024:
        with open(ckpt_path, "rb") as f:
            head = f.read(64)
        if b"git-lfs" in head or head.startswith(b"version https://git-lfs"):
            raise RuntimeError(
                f"AGFN checkpoint at {ckpt_path} is a git-LFS pointer "
                f"(~131 bytes on disk). Run `cd ref/AGFN && git lfs pull` "
                f"to download the real weights, then retry."
            )

    net = Net(gfn=True, Z_out_dim=1, start_node=None).to(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


@torch.no_grad()
def infer_heatmap(
    model: nn.Module,
    coords: Tensor,
    k_sparse: Optional[int] = None,
) -> Tensor:
    """Run AGFN and return a dense (N, N) edge-probability heatmap.

    Args:
        model: a Net returned by `load_agfn`.
        coords: (N, 2) float tensor of node coordinates in [0, 1]^2.
        k_sparse: number of nearest neighbours per node. Defaults to
            N // 10 to match AGFN's own inference script.
    Returns:
        (N, N) tensor with H[i, j] in (0, 1] for the k_sparse nearest
        neighbours of i and H[i, j] = 0 elsewhere. Diagonal entries are
        also 0 (the sparse graph never has self-loops).
    """
    model.eval()
    n = coords.shape[0]
    if k_sparse is None:
        k_sparse = max(1, n // DEFAULT_K_SPARSE_DIVISOR)
    # `gen_pyg_data` is AGFN's helper. It returns (Data, distances).
    pyg, _ = gen_pyg_data(coords, k_sparse=k_sparse, start_node=None)
    her = model(pyg)  # (n * k_sparse,) heuristic vector over the sparse edges
    mat = Net.reshape(pyg, her) + EPS  # (N, N) dense, +EPS guarantees > 0
    return mat
