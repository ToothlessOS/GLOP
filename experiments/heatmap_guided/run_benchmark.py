"""Benchmark: GLOP cascade baseline vs heatmap-guided cascade.

Both arms run on the *same* set of instances, starting from the *same*
random-insertion warm-start, with the *same* revisers and the *same*
revision budget. The only difference is how the next chunk is selected:

  - baseline: `reconnect` (non-overlapping chunks with rolling shift)
  - guided:   `guided_reconnect` (worst-aligning window + half-overlaps)

Per-iteration cost and wall time are recorded for both, then written to
`{out_dir}/raw.json` and aggregated into `summary.csv`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Tuple

import numpy as np
import torch

from utils.functions import LCP_TSP, load_model, load_problem
from utils.insertion import random_insertion_parallel

from .guided.guided_reconnect import guided_reconnect
from .guided.heatmap import load_agfn

# ---------------------------------------------------------------------------
# GLOP baseline with per-iteration history
# ---------------------------------------------------------------------------


def closed_loop_cost(seed: torch.Tensor) -> torch.Tensor:
    seg = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1)
    closing = (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
    return seg + closing


def _autocast_ctx(opts):
    """Return a `torch.amp.autocast` context wrapping the reviser
    forward with bf16 mixed precision, if enabled and on CUDA.

    bf16 has the same range as fp32 (no scaling needed) and is
    natively fast on Ampere+ GPUs (RTX 30/40, A100). It halves
    attention-score VRAM (1.6 GB → 820 MB per encoder layer at
    L=100 / B=1280) and typically gives 1.5–2× speedup over fp32.

    Falls back to a no-op if AMP is disabled or running on CPU.
    """
    if not getattr(opts, "use_amp", True):
        import contextlib

        return contextlib.nullcontext()
    if not str(opts.device).startswith("cuda"):
        import contextlib

        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def _apply_smoke_overrides(opts) -> None:
    """Apply --smoke overrides. No-op when opts.smoke is False.

    A fast end-to-end recipe that still exercises both arms: a single
    cascade stage with 2 iterations, width=1, no augmentation, and
    val/eval capped at 8. This is shorthand for the recipe documented
    in `experiments/heatmap_guided/README.md`'s "Smoke test" example.
    """
    if not getattr(opts, "smoke", False):
        return
    print("[*] --smoke: overriding to fast recipe "
          "(L=100, n_iter=2, width=1, --no_aug, val/eval<=8)")
    opts.revision_lens = [100]
    opts.revision_iters = [2]
    opts.width = 1
    opts.no_aug = True
    if opts.val_size > 8:
        opts.val_size = 8
    if opts.eval_batch_size > 8:
        opts.eval_batch_size = 8


def _print_cpu_amp_warning(opts) -> None:
    """Print a one-time advisory when the cascade is about to run on
    CPU with bf16 AMP enabled. AMP is a CUDA-only optimisation, so
    the run will be ~2x slower than the equivalent CUDA run. The two
    largest CPU levers are --no_aug (~4x speedup, no quality loss on
    this benchmark) and shrinking --revision_iters / using --smoke.

    Guards:
      - is_cpu: only warn for the CPU case; CUDA users with a stalled
        run have a different problem (likely autograd accumulation,
        which the existing torch.no_grad() fix already addresses).
      - not no_amp: don't warn if the user already opted out of AMP.
      - not smoke: don't warn when the user already opted into the
        fast path via --smoke.
    """
    is_cpu = str(opts.device).startswith("cpu")
    if not is_cpu:
        return
    if opts.no_amp:
        return
    if getattr(opts, "smoke", False):
        return
    print(
        "[!] Running on CPU with bf16 AMP disabled. Expect the GLOP "
        "cascade to be ~2x slower than the same run on CUDA. "
        "Recommended: pass --no_aug (4x speedup, no quality loss) "
        "and consider --smoke or smaller --revision_iters "
        "(e.g. 2 2 1) for a fast check.",
        flush=True,
    )


def _cascade_chunk_size(B: int, opts) -> int:
    """Compute the chunk size for the cascade reviser forward.

    The cascade processes `B = width × val_size` tours per LCP_TSP
    call. We split that into `num_chunks` (default 2) chunks so each
    reviser forward sees `ceil(B / num_chunks)` tours. With the
    default `B=1280` and `num_chunks=2`, the chunk size is 640.

    `--eval_batch_size` (default 128) is the per-call inference batch
    size — the chunk size the model is comfortable with at its
    training-time footprint. It is kept for compatibility with the
    README's CLI examples and the original `main.py` recipe. The
    cascade chunking itself is controlled by `--num_chunks`; the
    chunk size is whatever the chunks dictate.

    Pass `--num_chunks 1` to disable chunking (one reviser forward
    per LCP_TSP call, equivalent to the original `main.py` cascade
    driver).
    """
    n = max(1, int(getattr(opts, "num_chunks", 1)))
    return max(1, (B + n - 1) // n)


def _chunked_LCP_TSP(
    seed: torch.Tensor,
    get_cost_func,
    reviser,
    revision_len: int,
    revision_iter: int,
    opts,
    shift_len: int,
    chunk_size: int,
) -> torch.Tensor:
    """Run `utils.functions.LCP_TSP` on each chunk of `chunk_size` tours
    independently, then concatenate.

    `LCP_TSP` reshapes the input to `(B * N / L, L, 2)` — for
    `B = width × val_size = 1280` and `L = 100` that's 6400 SHPPs in
    one reviser forward, ~13 GB of attention scores after 4× coord
    augmentation. We chunk over B to keep each forward within the
    reviser's training-time footprint. The chunks are independent
    (the rolling shift is a no-op across chunks since each chunk
    sees a closed loop of its own tours) so the result is
    equivalent to a single full-batch call, just VRAM-bounded.

    The default `chunk_size=640` (set via `opts.eval_batch_size` in
    the caller) gives 2 chunks for the benchmark's `B=1280`; smaller
    chunk sizes are supported for users with tighter VRAM.
    """
    B = seed.shape[0]
    out_chunks = []
    with _autocast_ctx(opts):
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = seed[start:end]
            chunk = LCP_TSP(
                chunk,
                get_cost_func,
                reviser,
                revision_len,
                revision_iter,
                opts=opts,
                shift_len=shift_len,
            )
            out_chunks.append(chunk)
    return torch.cat(out_chunks, dim=0)


def glop_reconnect_with_history(
    get_cost_func,
    batch: torch.Tensor,
    opts,
    revisers: list,
) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """A per-iteration-instrumented version of `utils.functions.reconnect`.

    Records `{iter, stage, reviser_size, cost, t}` after every LCP_TSP
    call, so the cost-vs-iteration chart has a data point per pass.

    The reviser forward is chunked over the batch dimension
    (`_chunked_LCP_TSP`) so peak VRAM stays bounded.
    """
    seed = batch
    history: List[dict] = []

    if len(revisers) == 0:
        cost = closed_loop_cost(seed)
        return seed, cost, history

    for stage_id, reviser_size in enumerate(opts.revision_lens):
        assert reviser_size <= seed.size(1)
        n_iter = opts.revision_iters[stage_id]
        shift_len = max(reviser_size // n_iter, 1)
        # The whole cascade is pure inference — no gradients needed.
        # Disabling autograd is the dominant memory fix: every reviser
        # forward + gather would otherwise stay attached to the autograd
        # graph and accumulate across the ~50 iterations × ~10 chunks,
        # OOM'ing the run on CUDA. Mirrors `guided_reconnect.py:79` and
        # `main.py:102`.
        print(
            f"    [cascade] stage {stage_id}/{len(opts.revision_lens)-1} "
            f"start: L={reviser_size}, n_iter={n_iter}",
            flush=True,
        )
        with torch.no_grad():
            for it in range(n_iter):
                t0 = time.time()
                seed = _chunked_LCP_TSP(
                    seed,
                    get_cost_func,
                    revisers[stage_id],
                    reviser_size,
                    n_iter,
                    opts=opts,
                    shift_len=shift_len,
                    chunk_size=_cascade_chunk_size(seed.size(0), opts),
                )
                cost = closed_loop_cost(seed)
                history.append(
                    {
                        "iter": len(history),
                        "stage": stage_id,
                        "reviser_size": reviser_size,
                        "cost": cost.detach().cpu().tolist(),
                        "t": time.time() - t0,
                    }
                )
                # Per-iter progress so the user sees the cascade is
                # alive. Matches the existing "    ->" indentation.
                # `flush=True` matters when stdout is piped/redirected
                # (e.g. nohup, tee) — without it the prints buffer and
                # the run looks hung.
                done = len(history)
                total = sum(opts.revision_iters)
                # `cost_ori` reports best-of-width per instance (averaged),
                # but the cascade runs on all (width*val_size) tours so a
                # naive `cost.mean()` mixes bad warm-starts with good ones.
                # Show both: `cost` (best, comparable to `cost_ori`) and
                # `mean` (true mean over all warm-starts).
                mean_cost = cost.mean().item()
                if opts.width > 1:
                    n_inst = cost.size(0) // opts.width
                    cost_best = (
                        cost.reshape(n_inst, opts.width).min(1).values.mean().item()
                    )
                else:
                    cost_best = mean_cost
                dt = history[-1]["t"]
                # Running-mean dt for this stage so ETA smooths the
                # first iteration.
                avg_dt = sum(
                    h["t"] for h in history if h["stage"] == stage_id
                ) / (it + 1)
                eta = avg_dt * (n_iter - (it + 1))
                print(
                    f"    [cascade] stage {stage_id}/{len(opts.revision_lens)-1} "
                    f"(L={reviser_size}) iter {it+1}/{n_iter}  "
                    f"global {done}/{total}  "
                    f"cost={cost_best:.4f}  mean={mean_cost:.4f}  "
                    f"dt={dt:.2f}s  eta={eta:.0f}s",
                    flush=True,
                )
        # Mirror the guided arm: free intermediate references between
        # stages. CPU tensors don't have a CUDA caching allocator to
        # drain, but `gc.collect()` releases intermediates that the
        # reviser forwards leave behind via local closures.
        import gc  # local import: gc is only needed here

        gc.collect()

    # Final width-prune: keep the best warm-start per instance.
    # Each instance has `width` warm-starts, so reshape (B,) =
    # (width * val_size,) into (val_size, width) and min over width.
    # The earlier code used `eval_batch_size` as the row size, which
    # only works when `eval_batch_size == val_size` (the original
    # main.py DataLoader-chunked context). In run_benchmark.py the
    # full B is processed in one cascade pass, so the row size must
    # be `width` for the grouping to be correct. With `width=1` the
    # prune is a no-op (each "row" is a single element); we skip it
    # to avoid an unnecessary reshape.
    if not opts.no_prune and opts.width > 1:
        n_instances = cost.size(0) // opts.width
        cost, minidx = cost.reshape(n_instances, opts.width).min(1)
        # `minidx` indexes the `width` axis (dim 1 of the reshape);
        # arange(n_instances) indexes the `n_instances` axis (dim 0).
        seed = seed.reshape(n_instances, opts.width, seed.size(-2), 2)[
            torch.arange(n_instances), minidx
        ]
    return seed, cost, history


# ---------------------------------------------------------------------------
# Initial tour construction (mirrors main.py:60-69)
# ---------------------------------------------------------------------------


def build_warm_start(
    dataset, opts, device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate `width` random-insertion tours per instance.

    Returns:
        seed: (B*width, N, 2) — coords in tour order.
        pi:   (B*width, N)    — node IDs in tour order, parallel to seed.
        cost_ori: (B,)        — best random-insertion cost per instance.
    """
    # The `random_insertion` Cython extension requires a 3-D tensor
    # with `.shape` (see `utils/insertion.py:21`). LOCALDataset /
    # TSPDataset are torch Dataset objects, not tensors, so we
    # materialise the (V, N, 2) tensor up front.
    coords = torch.stack([dataset[i] for i in range(len(dataset))])

    orders = [torch.randperm(opts.problem_size) for _ in range(opts.width)]
    pi_all = [random_insertion_parallel(coords, order) for order in orders]
    pi_all = torch.tensor(np.array(pi_all).astype(np.int64)).reshape(
        len(orders), opts.val_size, opts.problem_size
    )  # (width, V, N)

    # Build batched coordinates and pi. Mirror `main.py:107`:
    # `batch.repeat(width, 1, 1)` -> (width*V, N, 2), then gather.
    pi_b = pi_all.reshape(-1, opts.problem_size)  # (width*V, N)
    coords_b = coords.repeat(opts.width, 1, 1)  # (width*V, N, 2)
    seed_b = coords_b.gather(1, pi_b.unsqueeze(-1).expand(-1, -1, 2))
    seed_b = seed_b.to(device)
    pi_b = pi_b.to(device)

    cost_ori = closed_loop_cost(seed_b).reshape(opts.width, opts.val_size).min(0).values
    return seed_b, pi_b, cost_ori


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(opts) -> None:
    # Apply --smoke overrides (if any) first so the rest of main sees
    # the recipe that --smoke implies. Must run before revisers are
    # loaded, the dataset is sized, or the CPU-thread cap is decided.
    _apply_smoke_overrides(opts)

    # One-time CPU/AMP advisory. Suppressed under --smoke (the user
    # already opted into the fast path) and --no_amp (the user
    # already opted out of AMP).
    _print_cpu_amp_warning(opts)

    # ---------------------------------------------------------------------------
    # Thread limits
    # ---------------------------------------------------------------------------
    # PyTorch defaults to using all available CPU cores via MKL/OpenMP. Each
    # thread owns its own memory arena, so for a CPU run with N cores the
    # peak working set is ~N× larger than the single-threaded working set.
    # Capping the threads also keeps the process predictable on shared
    # machines. Skip on GPU: intra-op threads only matter for CPU ops.
    if str(opts.device).startswith("cpu"):
        n_threads = max(1, int(os.environ.get("HEATMAP_GUIDED_THREADS", "2")))
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)
        print(f"[*] Capping CPU threads to {n_threads}")

    print(f"[*] Loading revisers for sizes {opts.revision_lens}")
    revisers = []
    for reviser_size in opts.revision_lens:
        reviser_path = f"pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt"
        reviser, _ = load_model(reviser_path, is_local=True)
        reviser.to(opts.device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)
        revisers.append(reviser)

    print(f"[*] Loading dataset {opts.path}")
    problem = load_problem("tsp")
    dataset = revisers[0].problem.make_dataset(
        filename=opts.path, num_samples=opts.val_size, offset=0
    )

    print(f"[*] Generating warm-start (width={opts.width})")
    seed, pi, cost_ori = build_warm_start(dataset, opts, opts.device)
    B, N, _ = seed.shape
    print(
        f"    -> seed shape {tuple(seed.shape)}, mean cost_ori {cost_ori.mean().item():.4f}"
    )

    def get_cost_func(input, pi):
        return problem.get_costs(input, pi, return_local=True)

    os.makedirs(opts.out_dir, exist_ok=True)

    # ---------------- Baseline (GLOP cascade) ----------------
    # The baseline does not need `pi` — `glop_reconnect_with_history`
    # operates on coords only. `pi` is only required by the guided arm.
    cost_baseline = None
    t_baseline = 0.0
    hist_baseline = []
    if not opts.no_baseline:
        print("[*] Running GLOP baseline (reconnect)")
        seed_baseline = seed.clone()
        t0 = time.time()
        seed_b_final, _, hist_baseline = glop_reconnect_with_history(
            get_cost_func, seed_baseline, opts, revisers
        )
        t_baseline = time.time() - t0
        cost_baseline = closed_loop_cost(seed_b_final)
        print(
            f"    -> baseline cost {cost_baseline.mean().item():.4f} in {t_baseline:.1f}s"
        )
    else:
        print("[*] Skipping GLOP baseline (--no_baseline)")

    # ---------------- Heatmap-guided ----------------
    print(f"[*] Loading AGFN (scale={opts.problem_size})")
    agfn = load_agfn(opts.problem_size, device=str(opts.device))

    print("[*] Running heatmap-guided cascade")
    seed_guided = seed.clone()
    pi_guided = pi.clone()
    t0 = time.time()
    seed_g_final, _, cost_g_final, hist_guided = guided_reconnect(
        get_cost_func,
        seed_guided,
        pi_guided,
        opts,
        revisers,
        agfn,
    )
    t_guided = time.time() - t0
    print(f"    -> guided cost {cost_g_final.mean().item():.4f} in {t_guided:.1f}s")

    # ---------------- Dump raw results ----------------
    raw = {
        "problem_size": opts.problem_size,
        "val_size": opts.val_size,
        "eval_batch_size": opts.eval_batch_size,
        "width": opts.width,
        "revision_lens": opts.revision_lens,
        "revision_iters": opts.revision_iters,
        "cost_ori": cost_ori.detach().cpu().tolist(),
    }
    if not opts.no_baseline:
        raw["baseline"] = {
            "cost": cost_baseline.detach().cpu().tolist(),
            "history": hist_baseline,
            "wall_time": t_baseline,
        }
    raw["guided"] = {
        "cost": cost_g_final.detach().cpu().tolist(),
        "history": hist_guided,
        "wall_time": t_guided,
    }
    raw_path = os.path.join(opts.out_dir, "raw.json")
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"[*] Wrote {raw_path}")

    # ---------------- Summary ----------------
    summary_lines = [
        "method,mean_cost,std_cost,wall_time_s",
        f"warm_start,{cost_ori.mean().item():.6f},{cost_ori.std().item():.6f},0.0",
    ]
    if not opts.no_baseline:
        summary_lines.append(
            f"glop_baseline,{cost_baseline.mean().item():.6f},"
            f"{cost_baseline.std().item():.6f},{t_baseline:.2f}"
        )
    summary_lines.append(
        f"heatmap_guided,{cost_g_final.mean().item():.6f},"
        f"{cost_g_final.std().item():.6f},{t_guided:.2f}"
    )
    summary_path = os.path.join(opts.out_dir, "summary.csv")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"[*] Wrote {summary_path}")

    print()
    print("=" * 60)
    print(f"TSP-{opts.problem_size} (val_size={opts.val_size}, width={opts.width})")
    print(f"  warm_start : {cost_ori.mean().item():.4f}")
    if not opts.no_baseline:
        print(f"  glop       : {cost_baseline.mean().item():.4f}   [{t_baseline:.1f}s]")
    else:
        print("  glop       : (skipped --no_baseline)")
    print(f"  guided     : {cost_g_final.mean().item():.4f}   [{t_guided:.1f}s]")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=500)
    parser.add_argument("--val_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=2,
        help="Number of chunks to split the cascade batch into. "
        "chunk_size is computed as ceil(B / num_chunks). Default 2 "
        "gives 2 chunks for the default B=1280 (640 tours each).",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable bf16 mixed precision (default: enabled on CUDA).",
    )
    parser.add_argument("--revision_lens", nargs="+", default=[100, 50, 20], type=int)
    parser.add_argument("--revision_iters", nargs="+", default=[20, 25, 5], type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast smoke-test mode: sets --revision_lens 100 "
        "--revision_iters 2 --width 1 --no_aug and caps "
        "--val_size/--eval_batch_size to 8. ~30s end-to-end check.",
    )
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--no_prune", action="store_true")
    parser.add_argument(
        "--no_baseline",
        action="store_true",
        help="Skip the GLOP baseline cascade. The raw.json will omit "
        "the 'baseline' key; only the heatmap-guided arm runs.",
    )
    parser.add_argument("--decode_strategy", type=str, default="greedy")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for revisers/AGFN/batch ops. 'auto' picks CUDA "
        "if available else CPU; pass 'cuda', 'cuda:0', 'cpu' to override.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="results/tsp500")
    opts = parser.parse_args()

    # Resolve --device auto to the actual best available device. The
    # previous default was "cpu", which is what made the benchmark
    # chew through system RAM — the user has a CUDA-capable machine
    # in most cases and just never had to opt in.
    if opts.device == "auto":
        opts.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {opts.device}")

    if opts.path == "":
        opts.path = f"data/tsp/tsp{opts.problem_size}_test.pkl"
    opts.use_amp = not opts.no_amp
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    main(opts)
