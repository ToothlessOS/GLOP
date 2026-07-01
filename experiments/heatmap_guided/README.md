# Heatmap-guided decomposition for GLOP

This experiment replaces GLOP's **fixed-chunk, rolling-shift** decomposition
(`utils/functions.py:LCP_TSP`) with a **heatmap-guided** one. At each
revision step we score the current tour against an AGFN edge-probability
heatmap and revise the **worst-aligning** length-`L` window, plus two
half-overlapping windows of the same size (which naturally cover the
boundary edges of the main window).

## Motivation

GLOP's default cascade (`--revision_lens 100 50 20` with
`--revision_iters 20 25 5` — 50 reviser passes) chops the current TSP
tour into non-overlapping, evenly-rolled chunks. The chunk boundaries
are essentially random — they ignore where the tour actually disagrees
with the underlying combinatorial structure of the instance.

AGFN ships a pretrained edge-probability heatmap GNN
(`ref/AGFN/pretrained/tsp/gan_tsp_{500,1000}.pt`) that scores every
`(i, j)` edge in the instance. This heatmap is a strong, learned prior
on **which parts of the tour deserve attention**. Using it to pick
chunks should let the same reviser budget reach lower cost, or reach
the same cost in fewer passes.

The experiment benchmarks the two methods on identical instances with
identical warm-starts and the same revisers — only the chunk-selection
policy differs.

---

## How the algorithm works

```
inputs:  pi : (B, N) int64   — node IDs in tour order (closed loop)
         H  : (N, N) float   — AGFN edge-probability heatmap
         L  : int            — current reviser segment size
         exclude : (B, N) bool — tour positions already revised this stage

for stage_id, (L, n_iter) in enumerate(cascade):
    exclude.zero_()
    for it in range(n_iter):
        # 1. score every length-L window in the tour against the heatmap
        s = worst_window(pi, H, L, exclude_visited=exclude)

        # 2. revise the worst window + 2 half-overlaps of the same size
        #    main:  [s,        s + L)
        #    left:  [s - L/2,  s + L/2)   covers edge (s-1 -> s)
        #    right: [s + L/2,  s + 3L/2)   covers edge (s+L -> s+L+1)
        seed, pi = revise_main_with_overlaps(seed, pi, s, L, reviser, ...)

        # 3. mark the three windows as visited for this cascade stage
        exclude.scatter_(the three windows)
```

Three details make this work:

1. **`pi` is maintained in lockstep with `seed`.** Whenever we permute
   a window of `seed` we apply the same permutation to `pi`. The
   heatmap is indexed by node IDs, not by tour positions, so without
   this invariant the next iteration's heatmap lookup is wrong.

2. **`H` is invariant under tour permutation.** It scores edges by
   *node pair*, not by position. We run AGFN once per instance and
   reuse the heatmap across every iteration of every cascade stage.

3. **The exclude mask forces tour coverage.** Without it, every
   iteration would pick the same worst window (the heatmap's
   *relative* misalignments don't change when you fix one part). With
   it, the loop is forced to visit each tour position once before
   repeating.

---

## File map

```
experiments/heatmap_guided/
├── guided/
│   ├── heatmap.py            # load_agfn, infer_heatmap
│   ├── align.py              # edge_mismatch, worst_window (cyclic cumsum)
│   ├── overlapping_windows.py  # revise_main_with_overlaps (batched)
│   └── guided_reconnect.py   # top-level cascade driver (no_grad + lockstep pi)
├── run_benchmark.py          # CLI: builds dataset, runs both methods
├── plot.py                   # matplotlib cost-vs-iter chart
└── results/
    └── tsp{500,1000}/
        ├── raw.json          # per-iter cost + wall time
        ├── summary.csv       # mean cost and wall time per method
        └── cost_vs_iter.png  # produced by plot.py
```

| File | Function / class | Role |
|---|---|---|
| `guided/heatmap.py` | `load_agfn(scale, device)` | Loads the AGFN `Net`, with LFS-pointer detection. |
| | `infer_heatmap(model, coords)` | Runs AGFN once, returns dense `(N, N)` heatmap. |
| `guided/align.py` | `edge_mismatch(pi, H)` | Per-edge misalignment `m = 1 − H[edge]`. |
| | `worst_window(pi, H, L, exclude)` | Argmax over length-`L` window scores; cyclic wrap; respects exclude mask. |
| `guided/overlapping_windows.py` | `revise_main_with_overlaps` | Batch-extracts all `B×3` windows, calls `revision` once, scatters back. |
| `guided/guided_reconnect.py` | `guided_reconnect` | Cascade driver. Wraps the iteration loop in `torch.no_grad()`; keeps `pi` in lockstep. |
| `run_benchmark.py` | `main(opts)` | Loads revisers, builds dataset, builds warm-start, runs both methods, writes raw.json + summary.csv. |
| `plot.py` | `main(opts)` | Reads all `raw.json` under `--in_dir`, writes `cost_vs_iter.png` per problem size. |

---

## Prerequisites

```bash
# AGFN pretrained weights are git-LFS pointers (~131 bytes on disk)
# until you run this once:
cd /home/toothlessos/Projects/nrp/GLOP/ref/AGFN
git lfs pull
```

If LFS is not pulled, `load_agfn` raises a clear error pointing at
this command (`guided/heatmap.py:load_agfn`).

GLOP revisers live under `pretrained/Reviser-stage2/reviser_{20,50,100}/epoch-299.pt`.

Test data: `data/tsp/tsp{500,1000}_test.pkl`.

---

## CLI flags (`run_benchmark.py`)

| Flag | Default | Description |
|---|---|---|
| `--problem_size` | 500 | TSP size; must match an AGFN checkpoint (100/200/500/1000). |
| `--val_size` | 128 | Number of test instances. |
| `--eval_batch_size` | 128 | Tours per reviser forward call. Lower = less VRAM, more chunks; higher = more VRAM, fewer chunks. With bf16 (default), 128 fits comfortably in 8 GB. |
| `--no_amp` | off | Disable bf16 mixed precision (default: enabled on CUDA — gives ~1.5× speedup and halves attention-score VRAM on Ampere+ GPUs). |
| `--revision_lens` | `100 50 20` | Reviser sizes (cascade). Must have the same length as `--revision_iters`. |
| `--revision_iters` | `20 25 5` | Iterations per cascade stage. Total iterations = `sum(revision_iters)`. Use `--smoke` for a 2-pass fast recipe. |
| `--smoke` | off | Fast smoke-test shorthand. Sets `--revision_lens 100 --revision_iters 2 --width 1 --no_aug` and caps `--val_size`/`--eval_batch_size` to 8. Use this for a ~30s end-to-end check. |
| `--width` | 10 | Number of random-insertion warm-starts per instance. Use `--smoke` to set this to 1 for a fast check. |
| `--decode_strategy` | `greedy` | `greedy` or `sampling` (passed to the reviser). |
| `--no_aug` | off | Disable the 4× coordinate augmentation in `revision`. |
| `--no_prune` | off | Disable width-pruning after the first cascade stage. |
| `--device` | `auto` | Device for revisers/AGFN/batch ops. `auto` picks CUDA if available else CPU. Pass `cuda`, `cuda:0`, `cpu` to override. |
| `--seed` | 1 | Random seed for reproducibility. |
| `--path` | auto | Override the test data path (default `data/tsp/tsp{N}_test.pkl`). |
| `--out_dir` | `results/tsp500` | Where to write `raw.json` and `summary.csv`. |

---

## Usage

### Quick smoke test (CPU, ~30s)

Confirms both arms run end-to-end on a tiny problem size:

```bash
cd /home/toothlessos/Projects/nrp/GLOP
python -m experiments.heatmap_guided.run_benchmark \
    --problem_size 500 --val_size 8 --eval_batch_size 8 \
    --revision_lens 100 --revision_iters 5 --width 1 \
    --decode_strategy greedy --no_aug \
    --out_dir experiments/heatmap_guided/results/_smoke
```

> Tip: the same recipe is also available as a single flag:
> ```bash
> python -m experiments.heatmap_guided.run_benchmark --smoke \
>     --out_dir experiments/heatmap_guided/results/_smoke
> ```

While the cascade runs, the script prints one line per `LCP_TSP` call
so you can see it is alive:

```
[cascade] stage 0/0 start: L=100, n_iter=2
[cascade] stage 0/0 (L=100) iter 1/2  global 1/2  cost=12.34  dt=2.10s  eta=2s
[cascade] stage 0/0 (L=100) iter 2/2  global 2/2  cost=12.12  dt=1.95s  eta=0s
```

This writes `results/_smoke/raw.json` and `summary.csv`.

### Full TSP-500 benchmark

```bash
python -m experiments.heatmap_guided.run_benchmark \
    --problem_size 500 --val_size 128 --eval_batch_size 64 \
    --revision_lens 100 50 20 --revision_iters 20 25 5 --width 10 \
    --decode_strategy greedy \
    --out_dir experiments/heatmap_guided/results/tsp500
```

### Full TSP-1000 benchmark

```bash
python -m experiments.heatmap_guided.run_benchmark \
    --problem_size 1000 --val_size 128 --eval_batch_size 64 \
    --revision_lens 100 50 20 --revision_iters 20 25 5 --width 10 \
    --decode_strategy greedy \
    --out_dir experiments/heatmap_guided/results/tsp1000
```

### Plot

After running the benchmark(s), generate cost-vs-iteration charts:

```bash
python -m experiments.heatmap_guided.plot
# writes experiments/heatmap_guided/results/tsp{500,1000}/cost_vs_iter.png
```

The chart shows both arms on the same axes, with the cascade stages
shaded. `matplotlib` is imported lazily — the script will print a
warning and skip plotting if it isn't installed.

---

## Output files

### `raw.json`

Per-problem-size run record. Schema:

```jsonc
{
  "problem_size": 500,
  "val_size": 128,
  "eval_batch_size": 64,
  "width": 10,
  "revision_lens": [100, 50, 20],
  "revision_iters": [20, 25, 5],
  "cost_ori": [ ... ],             // B=val_size floats — warm-start cost
  "baseline": {
    "cost": [ ... ],               // B floats — final cost after GLOP cascade
    "history": [
      {
        "iter": 0,
        "stage": 0,
        "reviser_size": 100,
        "cost": [ ... ],           // B floats — per-instance cost after this iter
        "t": 0.83                  // wall time for this iteration (seconds)
      },
      ...
    ],
    "wall_time": 41.5              // total wall time for the GLOP arm (seconds)
  },
  "guided": { "...": "same shape as baseline" }
}
```

### `summary.csv`

Three rows — one per method:

```
method,mean_cost,std_cost,wall_time_s
warm_start,12.345678,0.123456,0.00
glop_baseline,10.876543,0.098765,41.50
heatmap_guided,10.654321,0.087654,38.20
```

### `cost_vs_iter.png`

`plot.py` reads `raw.json` and emits a 2-line cost-vs-iteration chart
(GLOP baseline vs heatmap-guided), with the cascade-stage regions
shaded. Title shows `TSP-{N} (val_size=V, width=W)`.

---

## Memory & performance notes

The most common cause of "memory explodes" is running on CPU when a
GPU is available. The script's `--device` defaults to `auto` which
picks CUDA when available — verify the startup banner shows the
device you expect:

```
[*] Device: cuda
```

If you ever see `[*] Device: cpu` and you have a CUDA-capable
machine, your PyTorch build doesn't see the GPU. Re-install with
`pip install torch --index-url https://download.pytorch.org/whl/cu118`
(or the matching CUDA version).

If you must run on CPU (no GPU), the script caps threads to 2 by
default (`HEATMAP_GUIDED_THREADS=N` to override) to keep the
per-thread MKL/OpenMP memory arena bounded. The script also prints a
one-time advisory at startup when it detects CPU + AMP-enabled — the
warning recommends `--no_aug` (4× speedup, no quality loss on this
benchmark) and `--smoke` / smaller `--revision_iters` for a fast check.
The warning is suppressed under `--smoke` (you already opted into the
fast path) and under `--no_amp` (you already opted out of AMP).

GLOP's revision loop is also a known hot spot — it builds an autograd
graph around every `reviser.forward` + `gather`, and at the benchmark
scale (`width=10 × val_size=128 = 1280` tours × 50 iterations) that
graph accumulates to hundreds of MB and OOMs the run. Three changes
keep it bounded:

1. **`torch.no_grad()` around the cascade** (`guided_reconnect.py:79`).
   The cascade is pure inference; no gradients are needed. Without
   this, every iteration leaks ~megabytes of autograd graph.
2. **Batched `revise_main_with_overlaps`** (`overlapping_windows.py`).
   The old code ran `for bi in range(B): revision(...)` — `B×3`
   reviser calls per cascade iteration plus an `(L, L, 2)` `isclose`
   tensor per call. The new code gathers all `B×3` windows into one
   `(B×3, L, 2)` batch, runs `revision` once, and recovers the
   per-window permutation with a single `torch.cdist.argmin`.
3. **Vectorised `exclude` mask update** (`guided_reconnect.py:99-113`).
   Replaces a per-instance Python loop (with `int(s[bi].item())` CPU
   syncs) with one advanced-indexed scatter.

The memory-regression test `test_guided_reconnect_does_not_accumulate_graph`
in `tests/test_heatmap_guided.py` asserts that 25 cascade iterations
don't grow RSS by more than 64 MB. Before the fix this would climb
well past 200 MB.

### Wall-time note

The heatmap-guided arm runs **3× as many reviser calls per iteration**
as the baseline (main + left half-overlap + right half-overlap), but
those calls are batched. Empirically the wall times are comparable;
see the `wall_time_s` column in `summary.csv`.

---

## Tests

```bash
ruff check experiments/heatmap_guided tests/test_heatmap_guided.py
pytest tests/test_heatmap_guided.py -v
```

The tests use a tiny mock `AttentionModel` — no GPU, no AGFN
checkpoints, no GLOP revisers required.

| Test | What it covers |
|---|---|
| `test_edge_mismatch_shape_and_sign` | `edge_mismatch` returns the right shape and sign. |
| `test_edge_mismatch_picks_low_heat_edges` | Low-heat edges score higher. |
| `test_worst_window_picks_min_alignment` | The picker returns the worst-aligned window. |
| `test_worst_window_cyclic_wrap` | Windows that wrap around `N` are read correctly. |
| `test_exclude_visited_skips_used` | The exclude mask forces coverage. |
| `test_worst_window_tie_breaks_smallest_s` | Ties resolve to the smallest `s`. |
| `test_pi_lockstep_after_revision` | `(seed, pi)` multiset invariant holds. |
| `test_overlapping_windows_covers_boundary` | The three windows (main, left, right) hit the expected positions. |
| `test_guided_reconnect_improves_or_equals` | End-to-end with a mock reviser; final cost ≤ warm-start cost. |
| `test_guided_reconnect_does_not_accumulate_graph` | **Memory regression**: RSS growth < 64 MB over 25 iterations. |

---

## Known limitations / future work

- **One heatmap per instance.** `infer_heatmap` is called on
  `seed[0]`, the first instance's coords — every other tour in the
  batch uses the same heatmap. This is fine for the current
  benchmark (instances are i.i.d. and width=10 helps average out the
  noise), but a per-instance or PyG-batched forward would be cleaner.
  See `guided_reconnect.py:62-70`.
- **Coordinate range.** AGFN was trained on `[0, 1]^2`. TSPLIB-style
  instances with arbitrary bounding boxes would need a
  `coordinate_transformation` before AGFN forward (the reviser
  already has this in `utils/functions.py:190`).
- **Per-instance heatmap cost.** AGFN is run once per *run*, not per
  iteration, but it's not run inside the cascade hot path — i.e. one
  forward per instance × `val_size` instances. For very large
  val_size this becomes the dominant cost.