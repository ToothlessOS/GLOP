# TSP Diagnosis: Self-Intersection, Convex-Hull, and Purity-Order Checks

This document maps the **diagnosis** stage of the GLOP / LKH-3 comparison
onto the actual code in this repository. After a TSP solver (GLOP or LKH-3)
returns a tour, three geometric properties are checked:

1. **No self-intersection** — the tour should not cross itself.
2. **Convex-hull consistency** — the tour should visit the convex hull
   vertices in cyclic order.
3. **Purity order** — for each edge, count how many other cities lie
   inside the edge's diameter circle.

The first two are classical heuristics for "is this a reasonable TSP
tour?". The third (purity order) is a finer-grained scalar metric that
measures how locally optimal each edge is, in the diameter-circle sense.

---

## TL;DR

| Concern                          | File                          | Function / Lines             |
|----------------------------------|-------------------------------|------------------------------|
| No-self-intersection check       | `utils/diagnosis.py:17`       | `check_no_self_intersection` |
| Convex-hull consistency          | `utils/diagnosis.py:236`      | `check_convex_hull`          |
| Purity-order computation         | `utils/diagnosis.py:287`      | `check_purity_order`         |
| Per-instance metric aggregation  | `utils/diagnosis.py:347`      | `_summarize_purity_order`    |
| Problematic-tour plot            | `utils/diagnosis.py:402`      | `plot_tsp_tours`             |
| Purity-colored tour plot         | `utils/diagnosis.py:618`      | `plot_tsp_tours_purity`      |
| Direct crossing-fix 2-opt        | `utils/diagnosis.py:95`       | `fix_intersections_via_2opt` |
| GLOP output driver               | `main.py:705-794`             | TSP diagnosis block          |
| LKH-3 driver                     | `LKH3_eval/run_diagnosis.py`  | `main` / `evaluate_one_instance` |
| Unit tests                       | `tests/test_diagnosis.py`     | `_test_check_purity_order_*` etc. |

---

## 1. Purity Order — Definition

For each edge `e_{ij} = (x_i, x_j)` in a closed-loop TSP tour, treat the
edge as a diameter of a circle and count how many other cities lie
inside that circle:

```
N_c(e_{ij}) = { x ∈ X \ {x_i, x_j} : (x_i - x)^T (x_j - x) < 0 }
K_p(e_{ij}) = |N_c(e_{ij})|
```

The two endpoints of each edge are excluded by the strict `<` test (the
dot product is exactly zero, not strictly negative). So `K_p` is always
a count of *third* cities lying inside the diameter circle.

**Range:** `K_p ∈ [0, N - 2]` per edge, where `N` is the number of cities
in the tour. The maximum is reached when every other city lies inside
the edge's diameter circle — typically a sign of a very long edge.

**Interpretation:**

- **`K_p = 0` ("pure" edge):** no other city lies inside the diameter
  circle. The edge is locally optimal in the diameter sense. A high
  fraction of pure edges means the tour is locally tight.
- **High `K_p`:** many cities are tucked inside the diameter circle.
  This usually corresponds to a long edge in a "spread-out" part of
  the tour, or a region with many interior cities.

The purity order is a **per-edge** scalar. The output of
`check_purity_order(seeds)` has shape `(B, N)`: for each batch instance
`b` and each edge index `k`, `purity_order[b, k]` is the purity of edge
`(seeds[b, k], seeds[b, (k+1) % N])`. A closed tour of `N` nodes has
exactly `N` edges, so the second axis indexes the per-edge scalars.

---

## 2. The Three Purity Summaries

`_summarize_purity_order(purity_order)` aggregates the `(B, N)` per-edge
tensor over the full batch and returns three scalar metrics:

| Key                              | Definition                                       | Edge case |
|----------------------------------|--------------------------------------------------|-----------|
| `mean_purity_order`              | mean K_p over all `B * N` edges                  | `0.0` if empty |
| `fraction_pure`                  | proportion of edges with `K_p == 0` (in `[0, 1]`)| `0.0` if empty |
| `mean_purity_order_nonpure`      | mean K_p restricted to edges with `K_p > 0`      | `NaN` if `fraction_pure == 1.0` |

These three summaries are reported in the GLOP run output and in the
LKH-3 dataset / grand-total blocks. They are also written to the
`summary.json` sidecar in the LKH-3 workdir for every instance.

**Why three and not one?** `mean_purity_order` alone is dominated by
the bulk of pure edges; `fraction_pure` alone ignores how impure the
non-pure edges are. `mean_purity_order_nonpure` isolates the "hard"
edges by ignoring trivially-pure ones, so a high value here means the
non-pure edges are concentrated in difficult regions.

---

## 3. Running the GLOP Pipeline

The GLOP main script runs the local-construction pipeline, dumps
tours, and runs the diagnosis block in the same process.

### Command

```bash
python main.py \
    --problem tsp \
    --problem_size 50 \
    --width 1 \
    --path data/tsp/tsp50_test.pkl \
    --save_tours
```

For full coverage of the test set, run for each `problem_size` in
`{20, 50, 100, 200, 500, 1000, 10000, 100000}` — there is no built-in
multi-dataset driver, so the loop is external.

### Expected stdout additions

```
=== Purity-order diagnostics ===
  Mean purity order:                ...
  Fraction of 0-order pure edges:   ...
  Mean purity order (K_p > 0 only): ...
=== Purity-colored visualization saved to: results/tour_purity_tsp50_w1.png ===
```

The block also prints the existing intersection / convex-hull summaries
and saves `results/tour_diag_tsp50_w1.png` if any tour is problematic.

### Output files

- `results/tour_diag_<tag>.png` — problematic tours only (intersection
  or hull violation), red edges on top of a faint base tour.
- `results/tour_purity_<tag>.png` — ALL tours in the batch (capped at
  16), edges colored by their scalar purity via a viridis gradient,
  colorbar on the right.
- `glop_dump_<tag>/<tag>_raw.pt` and `<tag>_postfix.pt` — full batch
  tensors (if `--save_tours` is set).
- `glop_dump_<tag>/meta.json` — index of saved tensors.

---

## 4. Running the LKH-3 Pipeline

The LKH-3 driver runs LKH-3 on a set of `.pkl` TSP datasets, then
applies all three diagnosis checks (intersection, hull, purity) on the
LKH-3 tours.

### Command

```bash
python LKH3_eval/run_diagnosis.py
```

This iterates all 9 default datasets (`tsp20_test` through
`tsp100000_test` and `tsplib49`) with `--max-instances 10` per dataset.
For full coverage, pass a higher cap:

```bash
python LKH3_eval/run_diagnosis.py --max-instances 10000
```

For a single dataset:

```bash
python LKH3_eval/run_diagnosis.py \
    --datasets data/tsp/tsp50_test.pkl \
    --max-instances 3 \
    --verbose
```

### Output layout

```
results/lkh_diagnosis/<UTC-timestamp>/<dataset>/
    <dataset>_inst<NNNN>.{par,tsp,log,tour}     # LKH temp files (--keep-tempfiles)
    tour_purity_<dataset>_inst<NNNN>.png         # NEW: per-instance purity plot
    summary.json                                 # per-instance records
```

The `summary.json` sidecar is rewritten for every dataset with the
per-instance records, including the new keys:

- `purity_mean` (float)
- `purity_frac_pure` (float)
- `purity_mean_nonpure` (float or `null` if all pure)
- `purity_png` (string path or `null` if plotting failed)

### Expected stdout additions

Per-dataset block adds a `Purity order:` line:

```
=== tsp50_test (3 instance(s)) ===
  LKH-3 solved:           3/3 OK (0 failed)
  Convex hull:            3/3 satisfy (100.0%)
  No self-intersection:   3/3 satisfy (100.0%)
  ...
  Purity order:           mean=1.2345, frac_pure=0.4000, mean_nonpure=2.0556
```

Grand-total footer adds a final `purity order:` line aggregating
across every instance of every dataset.

---

## 5. What the Plots Show

### `tour_diag_<tag>.png` — Problematic tours (red overlay)

Only renders tours that have at least one self-intersection or a
hull-visitation violation. The base tour is drawn in light blue, then
red overlays mark the edges that participate in any crossing. The
convex hull is a dotted gray border; hull vertices are red on failure
and blue on success. Skips clean tours entirely.

### `tour_purity_<tag>.png` — All tours (discrete 3-category color)

Renders every tour in the batch (capped at 16). Each edge is colored
by its purity *category* using a **discrete 3-color palette** (not a
continuous gradient). Purity order is an integer with a small
effective range, so three bins are sufficient and visually cleaner
than a continuous colormap:

| K_p  | Category     | Color        |
|------|--------------|--------------|
| `0`  | pure         | **C0** (blue)   |
| `1`  | one interior | **C1** (orange) |
| `2+` | two or more  | **C3** (red)    |

These are matplotlib's default `C0`/`C1`/`C3` (blue/orange/red) — the
same palette the project already uses in `plot_glop_lkh_diff` (blue =
GLOP-base, orange = LKH-base, red = "problematic"). The legend on
the right of the figure shows the three category swatches and labels.

A faint gray base loop is drawn first so the underlying tour structure
is visible when many edges overlap. Per-panel title shows the
per-category count: `K_p=0:N  K_p=1:N  K_p≥2:N/N`.

This plot does not skip clean tours — purity is most informative when
the tour is otherwise well-formed, since a heavily-intersecting tour
will have most edges inside each other's circles by construction.

---

## 6. Interpretation Guide

| Metric | High value means | Low value means |
|---|---|---|
| `mean_purity_order` | Many cities in diameter circles overall; tour has interior structure | Most edges are pure; tour is locally tight |
| `fraction_pure` | Most edges are diameter-locally-optimal | Many edges have interior cities — either long edges or "concave" geometry |
| `mean_purity_order_nonpure` | The non-pure edges are concentrated in difficult regions | Even the non-pure edges have only a few interior cities; tour is generally good |

**Cross-solver comparison:** the same three metrics on GLOP and LKH-3
tours on the same instance let you quantify the local-quality gap. A
GLOP tour with `fraction_pure = 0.6` and an LKH-3 tour with
`fraction_pure = 0.85` for the same instance is a strong signal that
GLOP picked a less-diameter-tight tour in that region.

**Cross-problem-size trend:** on uniformly random instances, expect
`mean_purity_order` to grow roughly like `O(√N)` because longer tours
have more diameter circles that happen to contain cities by chance.
`fraction_pure` should stay near 1 for very small N (no third city
to be inside) and decay slowly toward a constant for large N.

---

## 7. Pipeline End-to-End

```
GLOP:  main.py --problem tsp --problem_size N --width 1 --path <pkl>
        │
        └─ eval_dataset
              │
              └─ TSP diagnosis block (main.py:705-794)
                    ├─ check_no_self_intersection
                    ├─ check_convex_hull
                    ├─ check_purity_order              ← NEW
                    ├─ _summarize_purity_order         ← NEW (print 3 metrics)
                    ├─ plot_tsp_tours                  (problematic tours only)
                    ├─ plot_tsp_tours_purity           ← NEW (all tours, viridis)
                    ├─ fix_intersections_via_2opt
                    └─ [optional] run_post_revision

LKH-3: python LKH3_eval/run_diagnosis.py
        │
        └─ main()
              └─ for each dataset:
                    └─ evaluate_dataset
                          └─ for each instance:
                                ├─ run_lkh_once
                                ├─ check_convex_hull
                                ├─ check_no_self_intersection
                                ├─ check_purity_order         ← NEW
                                ├─ _summarize_purity_order    ← NEW
                                ├─ plot_tsp_tours_purity      ← NEW (per-instance PNG)
                                └─ record.update({...})       ← NEW: 4 keys
                    └─ _aggregate (NEW: 3 lists)
                    └─ _print_dataset_block (NEW: "Purity order:" line)
              └─ _print_grand_total (NEW: aggregate purity line)
```

---

## 8. Cross-Reference

- For the local-construction stage that *produces* these tours, see
  `docs/LOCAL_CONSTRUCTION.md`.
- For the GLOP-vs-LKH edge-diff comparison (which uses the same
  per-tour plotting primitives), see `utils/compare_solvers.py`.
- For **purity-guided decomposition** — using `check_purity_order` to
  align the revisor's first chunk window with the worst-purity edge —
  see `--purity_guided_decomp` in `main.py` and the "Purity-Guided
  Decomposition" section of `README.md`. The flag is opt-in (default
  off, bit-identical to pre-change behavior when absent).
