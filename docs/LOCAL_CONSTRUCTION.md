# GLOP Local Construction: Where the Code Lives

This document maps the *Local Construction* stage of GLOP (AAAI 2024) onto
the actual code in this repository. The core idea: an intermediate TSP
solution is broken into sub-**open-loop** TSP problems (Shortest Hamiltonian
Path Problems, **SHPPs**), each is revised by a neural solver, and the
revised chunks are stitched back into the full tour. The loop is then
re-entered with a different chunk offset, optionally cascading to smaller
revisers.

---

## TL;DR

| Concern                          | File                       | Function / Lines          |
|----------------------------------|----------------------------|---------------------------|
| Break TSP into sub-SHPPs         | `utils/functions.py`       | `decomposition` (176)     |
| Solve each sub-SHPP              | `utils/functions.py`       | `revision` (215)          |
| Iterate decompose ↔ revise       | `utils/functions.py`       | `LCP_TSP` (260)           |
| Cascade multiple revisers        | `utils/functions.py`       | `reconnect` (296)         |
| Top-level driver (TSP/CVRP/...)  | `main.py`                  | `eval_dataset` / `_eval_dataset` |
| Open-loop path cost              | `problems/local/problem_local.py` | `LOCAL.get_costs` (14) |
| Open-loop state                  | `problems/local/state_local.py`   | `StateLOCAL` (6)     |
| Neural reviser model             | `nets/attention_local.py`  | `AttentionModel` (43)     |
| Initial tour (random insertion)  | `main.py`                  | line 69                   |
| Closed-loop cost for full tour   | `main.py`                  | line 124                  |
| Training data decomposition      | `local_construction/generate_data_RI.py`, `generate_data_RG.py` | — |
| ATSP variant of the same idea    | `eval_atsp/test_glop.py`   | `revision` (121)          |

---

## End-to-End Trace

```text
main.py
  └─ utils.functions.reconnect      (line 296)
       └─ utils.functions.LCP_TSP  (line 260)
            ├─ utils.functions.decomposition   (line 176)   ← break into sub-SHPPs
            └─ utils.functions.revision         (line 215)   ← conquer each sub-SHPP
                 ├─ coordinate_transformation    (line 190)
                 ├─ 4× dihedral augmentation
                 ├─ nets.attention_local.AttentionModel.forward  (line 187)
                 │     ├─ _inner (line 277) — bidirectional decode
                 │     └─ _get_log_p (line 417) — open-loop mask
                 └─ gather sub-tour back into the seed (line 250)
```

The reshape in `LCP_TSP` (lines 290–292) merges revised chunks back into a
single sequence. A different `shift_len` on the next pass offsets the chunk
boundaries so a different set of sub-problems gets revised. After all
revisers in `opts.revision_lens` have been applied (e.g. `100 → 50 → 20`),
`main.py` adds the closing edge once at line 124 to compute the full
closed-loop tour cost.

---

## 1. Decomposition — break intermediate TSP into sub-SHPPs

**File:** `utils/functions.py` (lines 176–188)

```python
def decomposition(seeds, coordinate_dim, revision_len, offset, shift_len = 1):
    # change decomposition point
    seeds = torch.cat([seeds[:, shift_len:], seeds[:, :shift_len]], 1)

    if offset != 0:
        decomposed_seeds = seeds[:, :-offset]
        offset_seeds = seeds[:, -offset:]
    else:
        decomposed_seeds = seeds
        offset_seeds = None
    # decompose original seeds
    decomposed_seeds = decomposed_seeds.reshape(-1, revision_len, coordinate_dim)
    return decomposed_seeds, offset_seeds
```

What it does, line by line:

1. `shift_len` — rolls the tour so the next pass's chunk boundaries differ.
2. Slices off the trailing `offset = num_nodes % revision_len` nodes (not a
   full chunk).
3. Reshapes the remaining sequence into non-overlapping chunks of length
   `revision_len`. **This is the moment the long tour becomes a batch of
   `revision_len`-node open-loop sub-instances (SHPPs).**

---

## 2. Revision — conquer each sub-SHPP with the neural reviser

**File:** `utils/functions.py` (lines 215–258)

Key steps:

1. **Initial path-length cost** (line 219):
   `init_cost = revision_cost_func(decomposed_seeds, original_subtour)`.
   The cost function is `TSP.get_costs(..., return_local=True)` or
   `LOCAL.get_costs` — both return path length only, no closing edge.
2. **Coordinate transformation** (line 222): normalises each chunk to a
   unit-bounding-box for the reviser (shift / scale / optional swap so the
   longer axis is x).
3. **Dihedral augmentation** (lines 224–228): when `--no_aug` is not set,
   builds 4 rotations/reflections of each chunk and stacks them so the
   reviser can pick the best.
4. **Bidirectional decode** (lines 233–237):
   ```python
   cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(
       augmented_seeds, return_pi=True
   )
   ```
   The two outputs correspond to forward and reverse decoding — see
   `nets/attention_local.py:313` (`reverse=True`) and line 192
   (`torch.flip(pi2, dims=(-1,))`).
5. **Best-of-N selection** (lines 240–244): picks the best of
   `4 augments × 2 directions = 8` candidates per chunk.
6. **Accept only improvements** (line 249):
   ```python
   sub_tour[reduced_cost < 0] = original_subtour
   ```
7. **Merge back** (line 250):
   ```python
   decomposed_seeds = decomposed_seeds.gather(
       1, sub_tour.unsqueeze(-1).expand_as(decomposed_seeds)
   )
   ```

---

## 3. Iterate — decompose ↔ revise across the full tour

**File:** `utils/functions.py` (lines 260–293)

```python
def LCP_TSP(seeds, cost_func, reviser, revision_len, revision_iter, opts, shift_len):
    batch_size, num_nodes, coordinate_dim = seeds.shape
    offset = num_nodes % revision_len
    embeddings = None  # only when problem_size == revision_len
    for i in range(revision_iter):
        decomposed_seeds, offset_seed = decomposition(
            seeds, coordinate_dim, revision_len, offset, shift_len
        )
        original_subtour = torch.arange(0, revision_len, dtype=torch.long).to(...)

        if revision_len == num_nodes:
            decomposed_seeds_revised, embeddings = revision(
                ..., iter=i, embeddings=embeddings
            )
            embeddings = torch.cat(
                [embeddings[:, shift_len:], embeddings[:, :shift_len]], 1
            )  # roll the embeddings
        else:
            decomposed_seeds_revised, _ = revision(...)

        seeds = decomposed_seeds_revised.reshape(batch_size, -1, coordinate_dim)
        if offset_seed is not None:
            seeds = torch.cat([seeds, offset_seed], dim=1)
    return seeds
```

What to notice:

- **Outer loop** — runs `revision_iter` passes; each pass calls
  `decomposition` then `revision`.
- **Re-merge** — lines 290–292 flatten the revised chunks back into a single
  sequence and re-append the `offset_seed` tail that was sliced off earlier.
- **Different chunk boundaries each pass** — the `shift_len` arg (set by
  `reconnect` to `revision_len // revision_iter`) rolls the tour so each pass
  revises a different set of sub-problems.
- **Embedding reuse** — when `revision_len == num_nodes` (i.e. reviser size
  matches the whole instance), the encoder embeddings are reused across
  iterations instead of being recomputed.

---

## 4. Cascade revisers and prune

**File:** `utils/functions.py` (lines 296–332)

`reconnect` is the public entry point. It:

- Iterates over each reviser size in `opts.revision_lens`
  (e.g. `100 → 50 → 20`) and calls `LCP_TSP` for `opts.revision_iters[i]`
  passes per size.
- Optionally prunes to the best `width` candidates per instance (lines
  323–325) unless `--no_prune`.
- Returns the final tour batch and revised cost.

`main.py` calls it like this:

```python
tours, costs_revised = reconnect(
    get_cost_func=get_cost_func,
    batch=seed,
    opts=opts,
    revisers=revisers,
)
```

---

## 5. What makes the sub-problems *open-loop* (SHPP, not TSP)

### Open-loop cost

**File:** `problems/local/problem_local.py` (lines 14–22)

```python
@staticmethod
def get_costs(dataset, pi):
    # Gather dataset in order of tour
    d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
    # Length is distance (L2-norm of difference) from each next location
    # from its prev and of last from first
    return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1), None
```

Note the absence of a `+ (d[:, -1] - d[:, 0])` term — that is what makes it
an open path rather than a closed tour. The same logic is available for the
plain TSP problem via `TSP.get_costs(..., return_local=True)`.

### Open-loop state

**File:** `problems/local/state_local.py`

`StateLOCAL` (line 6) carries `first_a` and `last_a` — the two free
endpoints of the open path. The model uses both during decoding (see
`_get_parallel_step_context` in `nets/attention_local.py`, where
`state.last_a` is used as additional context for the TSP reviser).

### Open-loop masking

**File:** `nets/attention_local.py` — `AttentionModel._get_log_p`
(lines 417–456)

The mask at step 0 unmasks the first node and keeps the last masked; the
mask at the final step unmasks the last node and keeps the first masked.
So the autoregressive decoder produces a *path* with free endpoints, not a
closed loop.

---

## 6. The neural reviser

**File:** `nets/attention_local.py`

- `AttentionModel` (line 43) — encoder + autoregressive decoder.
- `forward` (lines 187–197) — returns `cost, pi, cost2, flipped_pi2`
  (bidirectional decoding).
- `_inner` (lines 277–344) — runs forward and reverse passes with shared
  encoder embeddings; reverse direction uses `reverse=True` at line 313.
- `_get_log_p` (lines 417–456) — open-loop masking.
- `_get_parallel_step_context` (lines 458–513) — uses `state.last_a` so
  the model is aware of both endpoints of the SHPP.

Checkpoints live in `pretrained/Reviser-stage2/reviser_{size}/epoch-299.pt`
and are loaded in `main.py` lines 24–27 via
`utils.functions.load_model(..., is_local=True)`.

---

## 7. Training data also uses the same decomposition

- `local_construction/generate_data_RI.py` — calls
  `utils.functions.decomposition` to carve random-inserted 500-node TSPs
  into `revision_len`-sized SHPPs for Reviser-stage2 training. The script
  loops `shift` over `revision_iter` so each training instance sees every
  chunk offset.
- `local_construction/generate_data_RG.py` — same idea, applied to
  FI-revised data to produce Reviser-50 training sets.

Both reuse the same `decomposition` helper that the runtime uses, so the
training distribution matches inference.

---

## 8. ATSP variant (matrix-form, not Euclidean)

ATSP uses a different solver family (MatNet-style), so the decompose /
revise logic is reimplemented inline rather than calling
`utils.functions.revision`.

**File:** `eval_atsp/test_glop.py`

`revision(tour, inst, tester)` (lines 121–148) reshapes a tour into
`N_REVISER`-sized chunks, slices the sub-distance matrices, scales them,
runs the SHPP solver, and gathers the revised solution back. This is the
ATSP analogue of `utils.functions.revision`.

Supporting files:

- `eval_atsp/ASHPPEnv.py` — `ASHPPEnv` (Augmented SHPP environment with
  fixed start = node 0 and fixed end = last node).
- `eval_atsp/ASHPPModel.py` — MatNet-style encoder/decoder for matrix
  SHPPs.
- `eval_atsp/ASHPPTrainer.py`, `train_glop.py` — training loop and entry.
- `eval_atsp/ATSPTester_glop.py` — tester that uses the SHPP solver as the
  inner solver for ATSP.

---

## 9. Why each part exists (paper → code)

| Paper concept                        | Code                                     |
|--------------------------------------|------------------------------------------|
| Sub-open-loop TSP (SHPP)             | `LOCAL.get_costs` (no closing edge)      |
| Decompose intermediate solution      | `utils.functions.decomposition`          |
| Solve each sub-problem with a policy | `utils.functions.revision` +             |
|                                      | `nets.attention_local.AttentionModel`    |
| Stitch chunks back together          | `LCP_TSP` reshape (lines 290–292)        |
| Multi-iteration refinement           | `LCP_TSP` outer loop                     |
| Cascade multiple reviser sizes       | `reconnect` (cascades over               |
|                                      | `opts.revision_lens`)                    |
| Initial solution (random insertion)  | `main.py:69` →                           |
|                                      | `utils.insertion.random_insertion_parallel` |
| Closing the loop for final cost      | `main.py:124` — single closing edge      |

---

## 10. Running the pipeline end-to-end

After downloading checkpoints and test data per the top-level `README.md`:

```bash
python main.py --problem_size 500 --revision_iters 20 25 5 \
               --revision_lens 100 50 20 --width 10 \
               --eval_batch_size 64 --val_size 128 \
               --decode_strategy greedy
```

Trace path during execution:

```
main.py → utils.functions.reconnect (296)
        → LCP_TSP (260)
            → decomposition (176)         # break into sub-SHPPs
            → revision (215)              # conquer each sub-SHPP
                → AttentionModel.forward (nets/attention_local.py:187)
        reshape + offset re-append (290–292)  # merge back
```

To debug, set a breakpoint at `utils/functions.py:215` (entering
`revision`) — that's where each sub-SHPP is handed to the neural solver.

