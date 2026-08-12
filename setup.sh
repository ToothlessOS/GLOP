#!/usr/bin/env bash
# Install Python dependencies for GLOP.
# Assumes an active virtualenv (e.g. conda). Python-only — does not install CUDA
# drivers, system packages, or build LKH-3. See README.md "Dependencies" for context.

set -euo pipefail

PYTORCH_INDEX="https://download.pytorch.org/whl/cu117"
PYG_INDEX="https://data.pyg.org/whl/torch-1.13.0+cu117.html"

# Core CUDA-11.7 torch stack (must come from the PyTorch wheel index).
pip install --index-url "$PYTORCH_INDEX" torch==1.13.0+cu117

# PyG companion wheels must match torch + CUDA version.
pip install --extra-index-url "$PYG_INDEX" \
    pyg-lib==0.4.0+pt113cu117 \
    torch-scatter==2.1.1+pt113cu117 \
    torch-sparse==0.6.17+pt113cu117

# Everything else (torch_geometric, scipy, random-insertion, tensorboard-logger, …).
pip install -r requirements.txt