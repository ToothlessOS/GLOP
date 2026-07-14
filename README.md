# [AAAI 2024] GLOP: Learning Global Partition and Local Construction for Solving Large-scale Routing Problems in Real-time

**Welcome!** This repository contains the code implementation of paper [*GLOP: Learning Global Partition and Local Construction for Solving Large-scale Routing Problems in Real-time*](https://arxiv.org/abs/2312.08224). GLOP is a unified hierarchical framework that efficiently scales toward large-scale routing problems. It partitions large routing problems into Travelling Salesman Problems (TSPs) and TSPs into Shortest Hamiltonian Path Problems (SHPPs). We hybridize non-autoregressive neural heuristics for coarse-grained problem partitions and autoregressive neural heuristics for fine-grained route constructions.

![diagram](./assets/diagram.png)

---

## News
🚀 Oct 2024: We released a [Python library](https://github.com/Furffico/random-insertion) for performing fast random insertion on TSP and SHPP instances

🐛 Jul 2024: Thanks to [Wenzheng Pan](https://github.com/wzever), we detected a [bug](https://github.com/henry-yeh/GLOP/pull/3) in the insertion for ATSP and fixed it. After the bug fix, GLOP achieves better performance on ATSP; see the below table for the updated results. Results on other problems are unaffected.

Version|ATSP150|ATSP250|ATSP1000
-|-|-|-
Before| 1.89 (6.4s) | 2.10 (9.6s) | 2.79 (39s)
After| 1.89 (8.2s) | 2.04 (9.3s) | 2.33 (15s)

---

## Highlights

- Hybridizing **non-autoregressive** solvers for problem partitions and **autoregressive** solvers for solution constructions.
- Competitive performance across large-scale **TSP, ATSP, CVRP, and PCTSP**.
- State-of-the-art scalability and efficiency: reasonable solutions for **TSP100K**, etc.

---

## Dependencies

- Python>=3.8
- NumPy 1.23
- PyTorch 1.13.0
- [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) 2.0.7
- [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) 0.6.9
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 2.0.4
- SciPy
- tqdm
- [random-insertion](https://github.com/Furffico/random-insertion)>=0.3.0

---

## How to Use

### Resources

- Download checkpoints from [checkpoints-downloading-link](https://drive.google.com/file/d/1u9-GVTMRux3rWGcbipSqyTyBx_V8pm9G/view?usp=sharing) and place them in `./pretrained`.

- Download test datasets from [test-datasets-downloading-link](https://drive.google.com/file/d/1WuICJGKRsiTjVTq7_ivh29wWShv8BRBO/view?usp=sharing) and place them in `./data`.

### Evaluation 
To evaluate our method on your own datasets, add `--path PATH_OF_YOUR_DATASET`.

#### For TSP
```bash
# For TSP500:
python main.py --problem_size 500 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 10 --eval_batch_size 64 --val_size 128 --decode_strategy greedy

# For TSP1000:
python main.py --problem_size 1000 --revision_iters 20 25 5 --revision_lens 100 50 20 --width 10 --eval_batch_size 32 --val_size 128 --decode_strategy greedy

# For TSP10k:
python main.py --problem_size 10000 --revision_iters 50 25 5 --revision_lens 100 50 20 --width 1 --eval_batch_size 16 --val_size 16 --decode_strategy greedy

# For TSP100k:
python main.py --problem_size 100000 --revision_iters 50 25 5 --revision_lens 100 50 20 --width 1 --eval_batch_size 1 --val_size 1 --decode_strategy greedy

# To conduct cross-distribution evaluation, e.g.:
python main.py --problem_size 100 --revision_lens 100 50 20 10 --revision_iters 20 10 10 5 --width 140 --eval_batch_size 100 --val_size 10000 --decode_strategy sampling --path data/tsp/tsp_uniform100_10000.pkl --no_aug --no_prune

# To reproduce the results of 49 TSPLib instances:
python eval_tsplib.py --eval_batch_size 1 --val_size 49 --path data/tsp/tsplib49.pkl --width 128 --decode_strategy greedy --no_prune
```


To reduce the inference duration, try:
```bash
# set
--width 1
# add
--no_aug
# less revisions, e.g.,
--revision_iters 5 5 5
```

#### For ATSP

Please refer to `./eval_atsp/`


#### For CVRP

```bash
# For CVRP1K using LKH-3 as sub-solver: 
python eval_cvrp.py --cpus 12 --problem_size 1000

# For CVRP1K using neural sub-TSP solver
python main.py --problem_type cvrp --problem_size 1000 --revision_lens 20 --revision_iters 5

# For CVRP2K using LKH-3 as sub-solver: 
python eval_cvrp.py --cpus 12 --problem_size 2000

# For CVRP2K using neural sub-TSP solver
python main.py --problem_type cvrp --problem_size 2000 --revision_lens 50 20 --revision_iters 5 5

# For CVRP5K using LKH-3 as sub-solver
python eval_cvrp.py --cpus 12 --problem_size 5000 --ckpt_path pretrained/Partitioner/cvrp/cvrp-2000.pt

# For CVRP5K using neural sub-TSP solver
python main.py --problem_type cvrp --problem_size 5000 --ckpt_path pretrained/Partitioner/cvrp/cvrp-2000.pt --revision_lens 20 --revision_iters 5

# For CVRP7K using LKH-3 as sub-solver
python eval_cvrp.py --cpus 12 --problem_size 7000 --ckpt_path pretrained/Partitioner/cvrp/cvrp-2000.pt

# For CVRP7K using neural sub-TSP solver
python main.py --problem_type cvrp --problem_size 7000 --ckpt_path pretrained/Partitioner/cvrp/cvrp-2000.pt --revision_lens 20 --revision_iters 5

# For CVRPLIB using LKH-3 as sub-solver
python eval_cvrplib.py

# For CVRPLIB using neural sub-TSP solver
python eval_cvrplib_neural.py
```


#### For PCTSP

```bash
# e.g., for PCTSP500
python main.py --problem_type pctsp --problem_size 500 --n_subset 10 --eval_batch_size 50 --val_size 100 --revision_iters 10 10 5 --revision_lens 100 50 20

# set n_subset = 1 for greedy mode
--n_subset 1
```


### Training

Please refer to READMEs in `./local_construction/` and `./heatmap/*/`.

---

## Per-iteration cost logging

`LCP_TSP` (the GLOP sub-TSP revisor cascade in `utils/functions.py`) writes
one JSON object per iter per revisor layer to a JSONL sidecar. Each record
carries run-level metadata + per-iter cost, so the same file works for both
programmatic analysis and the standalone visualization script below.

### Enable the log

Pass `--iter_cost_log <path>` to `main.py` (or any other caller of
`utils.functions.LCP_TSP`). Omitting the flag preserves the previous
behavior — no file is written and nothing else changes.

```bash
# Example: TSP50, 3 iters, log to results/iter_costs_tsp50.jsonl
python main.py --problem_size 50 --revision_lens 20 --revision_iters 3 \
               --width 4 --eval_batch_size 16 --val_size 16 \
               --path data/tsp/tsp50_test.pkl \
               --iter_cost_log results/iter_costs_tsp50.jsonl \
               --no_progress_bar
```

The same flag is honored by the post-revision pass (`--post_revision_lens
/ --post_revision_iters`) so the log covers the full pipeline.

### JSONL record schema

One JSON object per line; any single record is self-describing because the
run-level metadata is embedded on every line. The plot script only reads a
subset of these fields.

```json
{
  "ts": "2026-07-09T03:00:00.000Z",     // ISO-8601 UTC timestamp of this iter
  "layer_id": 0,                          // revisor-layer index (0-indexed)
  "iter_id": 3,                           // 0-indexed iter within the layer
  "revision_len": 100,                    // revisor window size for this layer
  "revision_iter": 20,                    // total iter count for this layer
  "do_block_2opt": true,                  // whether _block_swap_two_opt ran
  "best": 16.421,                         // post-2-opt (or direct revisor) best cost
  "avg":  16.587,                         // post-2-opt (or direct revisor) avg cost
  "count": 128,                           // # tour instances aggregated
  "cost_before_2opt_best": 16.430,        // only when do_block_2opt=true
  "cost_before_2opt_avg":  16.601,        // only when do_block_2opt=true
  "iter_elapsed_s": 2.34,                 // wall-clock seconds spent in this iter
  "total_elapsed_s": 7.10,                // wall-clock seconds since LCP_TSP started
  // run-level metadata (embedded on every line):
  "problem_type": "tsp",
  "problem_size": 500,
  "val_size": 128,
  "width": 10,
  "revision_lens":  [100, 50, 20],
  "revision_iters": [20, 25, 5],
  "decode_strategy": "sampling",
  "seed": 1,
  "dataset_path": "data/tsp/tsp500_test.pkl",
  "tag": "tsp500_w10_lens100-50-20_iters20-25-5",
  "run_started_utc": "2026-07-09T02:59:55.000Z"
}
```

`best` and `avg` are post-2-opt values when `do_block_2opt=true`, and the
direct revisor output (equal to `cost_before_2opt_*`, which is omitted
from the record) when 2-opt is off. `total_elapsed_s` is monotonically
non-decreasing within a single LCP_TSP call.

### Render the convergence plot

```bash
python scripts/plot_solver_curve.py results/iter_costs_tsp50.jsonl \
                                   --out_dir results/
# → results/solver_curve_tsp50.png
```

The script groups records by `layer_id`, orders layers by `revision_len`
ascending (matching the coarse-to-fine revisor cascade), and produces a
single PNG with one subplot per layer. Each subplot draws the `best`
(solid, square markers) and `avg` (dashed, circle markers) curves
reduced across `--width`, with a viridis color ramp and first/last-value
annotations. The suptitle embeds the run-level metadata.

The script can be re-run on any saved JSONL without re-running the model —
useful for regenerating plots after a presentation change. See
`python scripts/plot_solver_curve.py --help` for the full CLI.

### Compatibility

The `--iter_cost_log` flag is opt-in. Existing commands and snapshot
artifacts (`meta.json` from `--save_tours`, `summary.csv`, etc.) are
unchanged. The in-memory `stats_list` accumulation that was previously
discarded is still produced; the JSONL is an additional, persistent
output channel.

---

## Purity-Guided Decomposition

GLOP's revisor cascade decomposes a tour into `revision_len`-sized chunks
and shifts the chunk window by `shift_len` per iteration. By default the
window is aligned to index 0 of the input tour. The `--purity_guided_decomp`
flag instead aligns the window so that the **worst-purity edge** (the edge
with the highest purity-order score K_p) lands at the **middle of the first
revisor chunk** — i.e., index `revision_len // 2` of chunk 0 on iteration 0.
This places the most problematic region in the middle of a subproblem so
the revisor's first pass attacks it directly.

### How it works

`check_purity_order` ([utils/diagnosis.py](utils/diagnosis.py)) returns a
`(B, N)` per-edge purity score K_p — higher means more interior cities lie
inside the edge's diameter circle, which empirically correlates with worse
local revisor performance. For each revisor layer (and post-revision pass):

1. Compute `scores = check_purity_order(seed).sum(dim=0)` — an `(N,)`
   tensor of batch-aggregated purity scores.
2. Take `loc = scores.argmax()` — the index of the worst edge.
3. Compute the rotation
   `initial_pos = (loc − shift_len − revision_len // 2) mod N`.
4. Apply the rotation once at the top of `LCP_TSP` (per layer), then the
   existing `shift_len` sweep operates on the rotated tour.

See [docs/DIAGNOSIS.md](docs/DIAGNOSIS.md) for the K_p definition, summary
statistics, and visualization. See [`utils/functions.py:LCP_TSP`](utils/functions.py)
for the rotation application.

### Usage

```bash
# Default behavior (unchanged): chunk window starts at index 0 of the input tour.
python main.py --problem_size 100 --revision_lens 5 10 --revision_iters 4 2 \
    --decode_strategy greedy --seed 0 --tag base

# Opt in: rotate so the worst-purity edge lands at the middle of chunk 0.
python main.py --problem_size 100 --revision_lens 5 10 --revision_iters 4 2 \
    --decode_strategy greedy --seed 0 --tag purity \
    --purity_guided_decomp --iter_cost_log runs/purity.jsonl
```

### Notes

- **Default off**: the flag is opt-in. With the flag absent, `initial_pos=0`
  and the rotation is skipped entirely (the tensor is not copied) — bit-identical
  to pre-change behavior. Existing snapshots and baseline runs are unaffected.
- **Per-layer constant**: `initial_pos` is computed once when entering each
  revisor layer (in both the primary cascade and the post-revision pass).
  Subsequent iterations within a layer still sweep with `shift_len`, so the
  bad edge is covered near the middle of every chunk window across the sweep.
- **Combined with `--diagnose` / `--do_block_2opt`**: independent flags; all
  three can be on simultaneously without conflict.
- **Cost**: `check_purity_order` is O(B·N²) per layer; negligible at
  problem sizes up to ~1000.

---

## Citation

🤩 If you encounter any difficulty using our code, please do not hesitate to submit an issue or directly contact us!

😍 If you do find our work helpful (or if you would be so kind as to offer us some encouragement), please consider kindly giving a star, and citing our paper.

```bibtex
@inproceedings{ye2024glop,
  title={GLOP: Learning Global Partition and Local Construction for Solving Large-scale Routing Problems in Real-time},
  author={Ye, Haoran and Wang, Jiarui and Liang, Helan and Cao, Zhiguang and Li, Yong and Li, Fanzhang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024},
}
```


## Acknowledgements

* [Attention, learn to solve routing problems!](https://github.com/wouterkool/attention-learn-to-route)
* [Learning Collaborative Policies to Solve NP-hard Routing Problems](https://github.com/alstn12088/LCP)
* [Generalize a small pre-trained model to arbitrarily large TSP instances](https://github.com/Spider-scnu/TSP)
* [Learning generalizable models for vehicle routing problems via knowledge distillation](https://github.com/jieyibi/AMDKD)
* [Matrix Encoding Networks for Neural Combinatorial Optimization](https://github.com/yd-kwon/MatNet)
