import math
import torch
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from utils import load_model
from torch.utils.data import DataLoader
import time
from utils.functions import reconnect
from utils.functions import load_problem
import pprint as pp
from utils.insertion import random_insertion_parallel
from heatmap.cvrp.infer import load_partitioner
from heatmap.cvrp.inst import sum_cost
from utils.diagnosis import (
    check_no_self_intersection,
    check_convex_hull,
    check_purity_order,
    _summarize_purity_order,
    plot_tsp_tours,
    plot_tsp_tours_purity,
    fix_intersections_via_2opt,
)
from utils.functions import run_post_revision, LCP_TSP, load_problem


def eval_dataset(dataset_path, opts, run_metadata=None):
    pp.pprint(vars(opts))

    revisers = []
    revision_lens = opts.revision_lens

    for reviser_size in revision_lens:
        reviser_path = f"pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt"
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)

    for reviser in revisers:
        reviser.to(opts.device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)

    # NEW: forward --iter_cost_log + run_metadata to the per-batch helper so
    # ``LCP_TSP`` (via ``reconnect``) can append the per-iter JSONL sidecar.
    iter_log_path = getattr(opts, "iter_cost_log", "")
    results, duration, all_stats, coords_for_dump = _eval_dataset(
        dataset_path,
        opts,
        opts.device,
        revisers,
        iter_log_path=iter_log_path,
        run_metadata=run_metadata,
    )

    costs, costs_revised, costs_revised_with_penalty, costs_warm_avg, tours = zip(
        *results
    )
    costs = torch.tensor(costs)  # best-of-width warm start per instance (existing)
    costs_warm_avg = torch.tensor(
        costs_warm_avg
    )  # avg-of-width warm start per instance (new)
    if opts.problem_type in ["cvrp", "cvrplib"]:
        costs_revised = torch.stack(costs_revised)
    else:
        costs_revised = torch.cat(costs_revised, dim=0)

    if opts.problem_type == "pctsp":
        costs_revised_with_penalty = torch.cat(costs_revised_with_penalty, dim=0)

    def _ci(t):
        return (2 * torch.std(t) / math.sqrt(len(t))).item()

    print("=== Warm start (random insertion) ===")
    print(
        "  avg  = {:.4f} +- {:.4f}".format(
            costs_warm_avg.mean().item(), _ci(costs_warm_avg)
        )
    )
    print("  best = {:.4f} +- {:.4f}".format(costs.mean().item(), _ci(costs)))
    print("=== Final (after LCP revision) ===")
    print(
        "  avg  = {:.4f} +- {:.4f}".format(
            costs_revised.mean().item(), _ci(costs_revised)
        )
    )
    print("  best = {:.4f}".format(costs_revised.min().item()))
    if opts.problem_type == "pctsp":
        print("=== Final with penalty ===")
        print(
            "  avg  = {:.4f} +- {:.4f}".format(
                costs_revised_with_penalty.mean().item(),
                _ci(costs_revised_with_penalty),
            )
        )
        print("  best = {:.4f}".format(costs_revised_with_penalty.min().item()))
    print("=== Total duration: {:.2f}s ===".format(duration))

    # Aggregate per-iteration stats across batches (weighted by count) and plot.
    """
    if all_stats:
        plot_solver_curve(all_stats, opts)
    """

    if opts.problem_type != "cvrp":
        tours = torch.cat(tours, dim=0)

    # For non-TSP types there is no canonical `(val_size, N, 2)` coords
    # tensor that maps cleanly to the GLOP tour positions; let the caller
    # treat that as 'snapshot not applicable'. For CVRP/PCTSP, downstream
    # comparison tooling targets tsp only.
    if opts.problem_type != "tsp":
        coords_for_dump = None

    return tours, coords_for_dump


def _save_tours_snapshot(out_dir, tag, stage, coords, tours, opts):
    """Persist a per-stage GLOP-tour snapshot to disk for head-to-head use.

    Writes ``<out_dir>/<tag>_<stage>.pt`` containing the original city
    coordinates (in canonical .pkl order) and the tours reordered into
    GLOP's traversal order, and updates ``<out_dir>/meta.json`` with one
    entry per stage (cost statistics + which file holds the snapshot).

    Args:
        out_dir: target directory; created if missing.
        tag: short identifier shared across stages — encodes the CLI
            configuration (problem type/size, width, lens, iters).
        stage: one of {"raw", "postfix", "postrev"}; used as the suffix
            on the .pt file and as a key in meta.json.
        coords: (M, N, 2) tensor — original city coordinates in .pkl
            canonical order. May be None for non-TSP runs.
        tours: (M, N, 2) tensor — GLOP tours at the current pipeline
            stage; closed-loop costs are recomputed from this for the
            meta.json summary statistics.
        opts: argparse Namespace; persisted in meta.json so the
            downstream compare_solvers.py knows exactly what produced
            this dump.
    """
    import datetime
    import json
    import os

    os.makedirs(out_dir, exist_ok=True)

    snap_path = os.path.join(out_dir, f"{tag}_{stage}.pt")
    snap_payload = {
        "stage": stage,
        "coords": None if coords is None else coords.detach().cpu(),
        "tours": tours.detach().cpu(),
    }
    torch.save(snap_payload, snap_path)

    # Recompute closed-loop cost from the saved `tours` to keep
    # meta.json consistent with what's on disk.
    cost = (tours[:, 1:] - tours[:, :-1]).norm(p=2, dim=2).sum(1) + (
        tours[:, 0] - tours[:, -1]
    ).norm(p=2, dim=1)

    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        meta = {
            "tag": tag,
            "problem_type": getattr(opts, "problem_type", "tsp"),
            "problem_size": getattr(opts, "problem_size", None),
            "val_size": getattr(opts, "val_size", None),
            "width": getattr(opts, "width", None),
            "revision_lens": list(getattr(opts, "revision_lens", [])),
            "revision_iters": list(getattr(opts, "revision_iters", [])),
            "decode_strategy": getattr(opts, "decode_strategy", "sampling"),
            "seed": getattr(opts, "seed", None),
            "dataset_path": getattr(opts, "path", ""),
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "stages": {},
        }

    meta["stages"][stage] = {
        "file": os.path.basename(snap_path),
        "cost_mean": float(cost.mean().item()),
        "cost_std": float(cost.std().item()),
        "cost_min": float(cost.min().item()),
        "cost_max": float(cost.max().item()),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"[save_tours] {stage}: wrote {snap_path} "
        f"(cost mean={cost.mean().item():.4f}, "
        f"min={cost.min().item():.4f})"
    )


def _eval_dataset(
    dataset_path, opts, device, revisers, iter_log_path=None, run_metadata=None
):
    start = time.time()
    if opts.problem_type == "tsp":
        dataset = revisers[0].problem.make_dataset(
            filename=dataset_path, num_samples=opts.val_size, offset=0
        )
        if opts.problem_size <= 100:
            if opts.width >= 4:
                opts.width //= 4
                opts.tsp_aug = True
            else:
                opts.tsp_aug = False
        orders = [torch.randperm(opts.problem_size) for i in range(opts.width)]
        pi_all = [
            random_insertion_parallel(dataset, order) for order in orders
        ]  # instance: (p_size, 2)
        pi_all = torch.tensor(np.array(pi_all).astype(np.int64)).reshape(
            len(orders), opts.val_size, opts.problem_size
        )  # width, val_size, p_size
    elif (
        opts.problem_type == "pctsp"
    ):  # dataset (n_cons*val_size, p_size, 2), pi_all (width, n_cons*val_size, p_size), penalty (n_cons, val_size)
        from problems.pctsp import init

        opts.eval_batch_size = opts.eval_batch_size * opts.n_subset
        dataset, penalty = init(
            dataset_path, opts
        )  # (n_val*n_subset, max_seq_len, 2), (val_size*n_subset, )
        dataset = dataset.cpu()
        max_seq_len = dataset.size(1)
        order = torch.arange(max_seq_len)  # width=1 by default for pctsp
        pi_all = random_insertion_parallel(
            dataset, order
        )  # (n_val*n_subset, max_seq_len)
        pi_all = torch.tensor(pi_all.astype(np.int64)).unsqueeze(
            0
        )  # (1, n_val*n_subset, max_seq_len)
        assert pi_all.shape == (1, opts.val_size * opts.n_subset, max_seq_len)
    elif opts.problem_type == "cvrp":
        from problems.cvrp import init

        dataset, n_tsps_per_route_lst = init(dataset_path, opts)
        opts.eval_batch_size = 1
    elif opts.problem_type == "cvrplib":
        from problems.cvrp import init

        ckpt_path = (
            "./pretrained/Partitioner/cvrp/cvrp-2000-cvrplib.pt"
            if opts.ckpt_path == ""
            else opts.ckpt_path
        )
        partitioner = load_partitioner(2000, opts.device, ckpt_path, 300, 6)
        dataset, n_tsps_per_route_lst = init(dataset_path, opts, partitioner)
        opts.eval_batch_size = 1

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    # Canonical, un-permuted city coordinates for downstream
    # head-to-head comparison (utils/compare_solvers.py recovers the
    # GLOP permutation by exact-match lookup against this tensor).
    coords_for_dump = None
    if opts.problem_type == "tsp":
        coords_for_dump = torch.stack(
            list(dataset.data)[: opts.val_size]
        )  # (val_size, N, 2)

    problem = load_problem("tsp")
    get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)

    results = []
    all_stats = []  # accumulated per-iter (best, avg, count) across batches
    for batch_id, batch in tqdm(enumerate(dataloader), disable=opts.no_progress_bar):
        # tsp batch shape: (bs, problem size, 2)
        avg_cost = 0
        with torch.no_grad():
            if opts.problem_type in ["tsp", "pctsp"]:
                p_size = batch.size(1)
                batch = batch.repeat(opts.width, 1, 1)  # (1,1,1) for pctsp
                pi_batch = pi_all[
                    :,
                    batch_id
                    * opts.eval_batch_size : (batch_id + 1)
                    * opts.eval_batch_size,
                    :,
                ].reshape(-1, p_size)
                seed = batch.gather(1, pi_batch.unsqueeze(-1).repeat(1, 1, 2))
            elif opts.problem_type in ["cvrp", "cvrplib"]:
                batch = batch.squeeze()  # (n_subTSPs_for_width_routes, max_seq_len, 2)
                n_subTSPs, max_seq_len, _ = batch.shape
                n_tsps_per_route = n_tsps_per_route_lst[batch_id]
                assert sum(n_tsps_per_route) == n_subTSPs
                opts.eval_batch_size = n_subTSPs
                order = torch.arange(max_seq_len)
                pi_batch = random_insertion_parallel(batch, order)
                pi_batch = torch.tensor(pi_batch.astype(np.int64))
                assert pi_batch.shape == (n_subTSPs, max_seq_len)
                seed = batch.gather(1, pi_batch.unsqueeze(-1).repeat(1, 1, 2))
                assert seed.shape == (n_subTSPs, max_seq_len, 2)
            else:
                raise NotImplementedError

            seed = seed.to(device)
            cost_ori = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (
                seed[:, 0] - seed[:, -1]
            ).norm(p=2, dim=1)
            if opts.problem_type in ["tsp", "pctsp"]:
                cost_ori_grouped = cost_ori.reshape(
                    -1, opts.eval_batch_size
                )  # (width, eval_batch_size)
                cost_ori_best, _ = cost_ori_grouped.min(0)  # (eval_batch_size,)
                cost_ori_avg = cost_ori_grouped.mean(0)  # (eval_batch_size,)
                avg_cost = cost_ori_best.mean().item()
                avg_cost_warm = cost_ori_avg.mean().item()
            elif opts.problem_type in ["cvrp", "cvrplib"]:
                avg_cost = sum_cost(cost_ori, n_tsps_per_route).min()
                avg_cost_warm = float(
                    cost_ori.mean().item()
                )  # width=1 forced for CVRP (main.py enforces)
            else:
                raise NotImplementedError

            if opts.problem_size <= 100 and opts.problem_type == "tsp" and opts.tsp_aug:
                seed2 = torch.cat((1 - seed[:, :, [0]], seed[:, :, [1]]), dim=2)
                seed3 = torch.cat((seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed4 = torch.cat((1 - seed[:, :, [0]], 1 - seed[:, :, [1]]), dim=2)
                seed = torch.cat((seed, seed2, seed3, seed4), dim=0)

            tours, costs_revised = reconnect(
                get_cost_func=get_cost_func,
                batch=seed,
                opts=opts,
                revisers=revisers,
                stats_list=all_stats,
                iter_log_path=iter_log_path,  # NEW: forwarded to LCP_TSP
                run_metadata=run_metadata,  # NEW: forwarded to LCP_TSP
            )

            # === POST-PROCESSING: fix self-intersections via direct 2-opt ===
            if (
                getattr(opts, "post_fix_intersections", False)
                and opts.problem_type == "tsp"
            ):
                from utils.diagnosis import fix_intersections_via_2opt

                cost_before = (tours[:, 1:] - tours[:, :-1]).norm(p=2, dim=2).sum(1) + (
                    tours[:, 0] - tours[:, -1]
                ).norm(p=2, dim=1)
                tours, fix_info = fix_intersections_via_2opt(tours)
                cost_after = (tours[:, 1:] - tours[:, :-1]).norm(p=2, dim=2).sum(1) + (
                    tours[:, 0] - tours[:, -1]
                ).norm(p=2, dim=1)
                pct = 100.0 * (
                    cost_after.mean().item() / cost_before.mean().item() - 1.0
                )
                print(
                    f"[POST] intersection-fix: iters={fix_info['iters_used']}, "
                    f"intersections {fix_info['initial_intersections']} -> "
                    f"{fix_info['final_intersections']}, "
                    f"cost {cost_before.mean().item():.4f} -> "
                    f"{cost_after.mean().item():.4f} ({pct:+.2f}%)"
                )
                costs_revised = cost_after
            # === END POST-PROCESSING ===

        if opts.problem_type == "pctsp":
            costs_revised_with_penalty, costs_revised_minidx = (
                costs_revised.reshape(-1, opts.n_subset)
                + penalty[
                    batch_id
                    * opts.eval_batch_size : (batch_id + 1)
                    * opts.eval_batch_size
                ].reshape(-1, opts.n_subset)
            ).min(1)
            costs_revised, _ = costs_revised.reshape(-1, opts.n_subset).min(1)
            tours = tours.reshape(-1, opts.n_subset, max_seq_len, 2)[
                torch.arange(opts.eval_batch_size // opts.n_subset),
                costs_revised_minidx,
                :,
                :,
            ]
            assert (
                costs_revised.size(0)
                == costs_revised_with_penalty.size(0)
                == tours.size(0)
                == opts.eval_batch_size // opts.n_subset
            )
        elif opts.problem_type in ["cvrp", "cvrplib"]:
            assert costs_revised.shape == (n_subTSPs,)
            costs_revised, best_partition_idx = sum_cost(
                costs_revised, n_tsps_per_route
            ).min(dim=0)
            subtour_start = sum(n_tsps_per_route[:best_partition_idx])
            tours = tours[
                subtour_start : subtour_start + n_tsps_per_route[best_partition_idx]
            ]
            assert tours.shape == (n_tsps_per_route[best_partition_idx], max_seq_len, 2)
            tours = tours.reshape(-1, 2)

        if opts.problem_type == "pctsp":
            results.append(
                (
                    avg_cost,
                    costs_revised,
                    costs_revised_with_penalty,
                    avg_cost_warm,
                    tours,
                )
            )
        elif opts.problem_type in ["tsp", "cvrp", "cvrplib"]:
            results.append((avg_cost, costs_revised, None, avg_cost_warm, tours))
        else:
            raise NotImplementedError

    duration = time.time() - start

    return results, duration, all_stats, coords_for_dump


def _aggregate_solver_curve(all_stats):
    """Aggregate per-batch per-iter stats into per-(layer, iter) means.

    Each input entry has keys {layer_id, iter_id, sum_best, sum_avg, count}.
    We sum across batches and divide by the total count for each (layer, iter).
    Returns a dict: {layer_id: {'iters': [...], 'best': [...], 'avg': [...]}}.
    """
    agg = {}
    for s in all_stats:
        key = (s["layer_id"], s["iter_id"])
        if key not in agg:
            agg[key] = {"sum_best": 0.0, "sum_avg": 0.0, "count": 0}
        agg[key]["sum_best"] += s["sum_best"]
        agg[key]["sum_avg"] += s["sum_avg"]
        agg[key]["count"] += s["count"]
    layers = sorted({k[0] for k in agg.keys()})
    out = {}
    for lid in layers:
        iters = sorted(k[1] for k in agg.keys() if k[0] == lid)
        best, avg = [], []
        for it in iters:
            entry = agg[(lid, it)]
            best.append(entry["sum_best"] / max(entry["count"], 1))
            avg.append(entry["sum_avg"] / max(entry["count"], 1))
        out[lid] = {"iters": iters, "best": best, "avg": avg}
    return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=200)
    parser.add_argument("--problem_type", type=str, default="tsp")
    parser.add_argument(
        "--val_size",
        type=int,
        default=128,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128,
        help="Batch size to use during (baseline) evaluation",
    )
    parser.add_argument(
        "--revision_lens",
        nargs="+",
        default=[20],
        type=int,
        help="The sizes of revisers",
    )
    parser.add_argument(
        "--revision_iters",
        nargs="+",
        default=[
            10,
        ],
        type=int,
        help="Revision iterations (I_n)",
    )
    parser.add_argument(
        "--decode_strategy",
        type=str,
        default="sampling",
        help="decode strategy of the model",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1,
        help="The initial solutions for a TSP instance generated with diversified insertion",
    )
    parser.add_argument(
        "--no_aug", action="store_true", help="Disable instance augmentation"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="",
        help="The test dataset path for cross-distribution evaluation",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--n_subset",
        type=int,
        default=1,
        help="The number of stochastically constructed PCTSP node subsets",
    )
    parser.add_argument(
        "--n_partition",
        type=int,
        default=1,
        help="The number of stochastically constructed CVRP partitions",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default="", help="Checkpoint path for CVRP eval"
    )
    parser.add_argument(
        "--no_prune",
        action="store_true",
        help="Do not prune the unpromising tours after the first round of revisions",
    )
    parser.add_argument(
        "--do_block_2opt",
        dest="do_block_2opt",
        action="store_true",
        default=True,
        help="Apply block-level (hypernode) 2-opt after each revision iteration (default: True)",
    )
    parser.add_argument(
        "--no_block_2opt",
        dest="do_block_2opt",
        action="store_false",
        help="Disable block-level (hypernode) 2-opt",
    )
    parser.add_argument(
        "--block_swap_max_iter",
        type=int,
        default=10,
        help="Maximum passes of best-of-pass block-swap search per revision iteration",
    )
    parser.add_argument(
        "--post_fix_intersections",
        action="store_true",
        default=False,
        help="After LCP_TSP, fix self-intersections via direct crossing-fix "
        "2-opt and report the cost reduction. (Re-warm-start with "
        "additional revisors is a separate manual step; see "
        "utils.diagnosis.fix_intersections_via_2opt and "
        "utils.functions.run_post_revision.)",
    )
    parser.add_argument(
        "--post_revision_lens",
        nargs="+",
        default=[],
        type=int,
        help="Revisor sizes for the re-warm-start pass after the 2-opt "
        "fix (e.g. 20 10). If empty, no re-warm-start is performed "
        "even if --post_fix_intersections is set.",
    )
    parser.add_argument(
        "--post_revision_iters",
        nargs="+",
        default=[],
        type=int,
        help="Iterations per layer for the re-warm-start pass. Must have "
        "the same length as --post_revision_lens.",
    )
    parser.add_argument(
        "--post_revision_batch_size",
        type=int,
        default=None,
        help="Chunk size for the post-revision pass. If unset, the full "
        "eval batch is processed in one shot. Set this to cap peak VRAM "
        "when the post-revisor stack is memory-heavy.",
    )
    parser.add_argument(
        "--diagnose",
        dest="diagnose",
        action="store_true",
        default=True,
        help="Run convex-hull / self-intersection diagnostics on each sub-TSP before revision (default: True)",
    )
    parser.add_argument(
        "--no_diagnose",
        dest="diagnose",
        action="store_false",
        help="Disable sub-TSP heuristic-validity diagnostics",
    )
    parser.add_argument(
        "--save_tours",
        type=str,
        default="",
        help="If set (a directory path), save GLOP final tours at every "
        "available stage (raw, postfix, postrev) to this directory. "
        "Each stage writes `<tag>_<stage>.pt` with `coords` and `tours` "
        "tensors plus a `meta.json` index, for head-to-head comparison "
        "with LKH-3 via utils/compare_solvers.py.",
    )
    parser.add_argument(
        "--iter_cost_log",
        type=str,
        default="",
        help="Path to a JSONL sidecar that records per-iter closed-loop "
        "tour cost from LCP_TSP. One JSON object per line. Each record "
        "contains per-iter fields (layer_id, iter_id, revision_len, "
        "best, avg, do_block_2opt, cost_before_2opt_best/avg when "
        "applicable, iter_elapsed_s, total_elapsed_s) plus run-level "
        "metadata (problem_type, problem_size, val_size, width, "
        "revision_lens, revision_iters, decode_strategy, seed, "
        "dataset_path, tag, run_started_utc). Visualize with "
        "`python scripts/plot_solver_curve.py <path>`. "
        "Empty (default) disables logging.",
    )
    parser.add_argument(
        "--purity_guided_decomp",
        dest="purity_guided_decomp",
        action="store_true",
        default=False,
        help="Rotate the input tour so the edge with the highest purity score "
        "(see utils/diagnosis.py:check_purity_order) lands at index "
        "revision_len // 2 of the first revisor chunk. This places the "
        "most problematic region in the middle of a subproblem so the "
        "revisor's first pass attacks it directly. Computed once per "
        "revisor layer; default: off (no rotation, original behavior).",
    )
    opts = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device_id = opts.device_id
    device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")
    opts.device = device
    print("using device:", device)

    if opts.path == "":
        if opts.problem_type == "tsp":
            opts.path = f"data/tsp/tsp{opts.problem_size}_test.pkl"
        elif opts.problem_type == "cvrp":
            opts.path = f"data/vrp/vrp{opts.problem_size}_test_seed1234.pkl"
        elif opts.problem_type == "pctsp":
            opts.path = f"data/pctsp/pctsp{opts.problem_size}_test_seed1234.pkl"
        else:
            raise NotImplementedError

    if opts.problem_type == "cvrp":
        if opts.eval_batch_size != 1:
            opts.eval_batch_size = 1
            warnings.warn("Set eval_batch_size to 1 for CVRP!")
        if opts.width != 1:
            opts.width = 1
            warnings.warn("Set width to 1 for CVRP!")
        if opts.n_partition != 1:
            opts.n_partition = 1
            warnings.warn("Set n_partition to 1 for CVRP!")
    if opts.problem_type == "pctsp":
        if opts.width != 1:
            opts.width = 1
            warnings.warn("Set width to 1 for PCTSP!")

    torch.manual_seed(opts.seed)

    # NEW: build the run-level metadata block that is embedded on every
    # line of the --iter_cost_log JSONL sidecar. The schema is a superset
    # of the meta.json fields written by _save_tours_snapshot (problem_type,
    # problem_size, val_size, width, revision_lens, revision_iters,
    # decode_strategy, seed, dataset_path) plus the run tag and UTC
    # start timestamp. Constructed once in __main__ so that both the
    # pre-revision and post-revision LCP_TSP calls see the same values.
    import datetime as _dt

    snapshot_tag = (
        f"{opts.problem_type}{opts.problem_size}_w{opts.width}"
        f"_lens{'-'.join(str(x) for x in opts.revision_lens)}"
        f"_iters{'-'.join(str(x) for x in opts.revision_iters)}"
    )
    run_metadata = {
        "problem_type": opts.problem_type,
        "problem_size": opts.problem_size,
        "val_size": opts.val_size,
        "width": opts.width,
        "revision_lens": list(opts.revision_lens),
        "revision_iters": list(opts.revision_iters),
        "decode_strategy": opts.decode_strategy,
        "seed": opts.seed,
        "dataset_path": opts.path,
        "tag": snapshot_tag,
        "run_started_utc": _dt.datetime.now(_dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        # NEW: epoch seconds at run start. LCP_TSP uses this to compute
        # ``total_elapsed_s`` (cumulative wall-clock time since the GLOP
        # run began, not since the current LCP_TSP call started) so the
        # per-iter JSONL records and the plot's suptitle can report a
        # single, monotonically increasing "total time" across all
        # revisor layers and the optional post-revision pass.
        "run_start_epoch_seconds": time.time(),
    }

    tours, coords_for_dump = eval_dataset(opts.path, opts, run_metadata=run_metadata)

    # Save tours at every available stage for head-to-head comparison
    # with LKH-3 via utils/compare_solvers.py. The tag encodes the CLI
    # configuration so multiple runs into the same --save_tours dir
    # don't clobber each other.
    save_dir = getattr(opts, "save_tours", "")
    if save_dir and opts.problem_type == "tsp":
        _save_tours_snapshot(
            save_dir, snapshot_tag, "raw", coords_for_dump, tours, opts
        )

    # NEW: Diagnosis on outputs (TSP only)
    if opts.problem_type == "tsp":
        print("TSP specific diagnostics on final tours:")
        print("=== Final tours shape: {} ===".format(tours.shape))
        # Run intersection and convex hull diagnostics
        (
            has_intersection,
            num_intersections,
            fraction_intersections,
            num_valid_pairs,
            intersection_results,
        ) = check_no_self_intersection(tours)
        consistency, convex_hull_results = check_convex_hull(tours)
        print("=== Intersection diagnostics ===")
        print("Has intersection: {}".format(has_intersection.sum().item()))
        print("Number of intersections: {}".format(num_intersections.sum().item()))
        print(
            "Fraction of intersections: {}".format(fraction_intersections.mean().item())
        )
        print("=== Convex hull diagnostics ===")
        print("Consistent with convex hull: {}".format(consistency.sum().item()))

        print(intersection_results.shape)
        print(intersection_results[0])
        print(convex_hull_results.shape)
        print(convex_hull_results)

        # Purity-order diagnostics on the raw GLOP output batch.
        # `tours` has shape (B, N, 2) in traversal order; check_purity_order
        # returns a (B, N) per-edge tensor. The three summaries are
        # aggregated across all B*N edges of the batch.
        purity_order = check_purity_order(tours)
        purity_summary = _summarize_purity_order(purity_order)
        print("=== Purity-order diagnostics ===")
        print(
            f"  Mean purity order:                {purity_summary['mean_purity_order']:.4f}"
        )
        print(
            f"  Fraction of 0-order pure edges:   {purity_summary['fraction_pure']:.4f}"
        )
        mpn = purity_summary["mean_purity_order_nonpure"]
        mpn_str = f"{mpn:.4f}" if not math.isnan(mpn) else "n/a"
        print(f"  Mean purity order (K_p > 0 only): {mpn_str}")

        # Purity-colored visualization. Plots ALL tours (capped at 16)
        # with each edge colored by its scalar purity via a viridis
        # gradient, so locally-optimal (low-purity) edges stand out
        # against "interior" (high-purity) edges.
        try:
            out_path_purity = plot_tsp_tours_purity(
                tours,
                purity_order=purity_order,
                has_intersection=has_intersection,
                num_intersections=num_intersections,
                intersection_results=intersection_results,
                consistency=consistency,
                out_dir="results",
                tag=f"tsp{opts.problem_size}_w{opts.width}",
            )
            if out_path_purity is not None:
                print(
                    f"=== Purity-colored visualization saved to: {out_path_purity} ==="
                )
        except Exception as e:
            print(
                f"[plot_tsp_tours_purity] skipped due to error: {type(e).__name__}: {e}"
            )

        # Visualize problematic tours (self-intersection and/or hull violation).
        try:
            out_path = plot_tsp_tours(
                tours,
                has_intersection=has_intersection,
                num_intersections=num_intersections,
                intersection_results=intersection_results,
                consistency=consistency,
                out_dir="results",
                tag=f"tsp{opts.problem_size}_w{opts.width}",
            )
            if out_path is not None:
                print(f"=== Problematic-tour visualization saved to: {out_path} ===")
        except Exception as e:
            print(f"[plot_tsp_tours] skipped due to error: {type(e).__name__}: {e}")

        # 2-opt is pure inference (no parameters to update) and does
        # in-place mutation of `tours`. Wrap in no_grad so the autograd
        # graph is not retained through up to max_iter=20 outer
        # iterations × ~2 check_no_self_intersection calls each.
        with torch.no_grad():
            tours, fix_info = fix_intersections_via_2opt(tours, max_iter=20)

        print("""[POST] 2-opt finished: {iters_used} iterations, "
            "{initial_intersections} initial crossings, "
            "{final_intersections} final crossings.""".format(**fix_info))
        print(
            "[POST] Initial cost: {:.4f}, final cost: {:.4f}".format(
                fix_info["initial_cost"].mean().item(),
                fix_info["final_cost"].mean().item(),
            )
        )

        # Snapshot `postfix` — tours after fix_intersections_via_2opt.
        if save_dir and snapshot_tag:
            _save_tours_snapshot(
                save_dir, snapshot_tag, "postfix", coords_for_dump, tours, opts
            )

        # === POST-PROCESSING: revisor re-warm-start on the fixed tours ===
        # Uses the standalone run_post_revision helper to drive additional
        # LCP_TSP passes with configurable revisor levels, treating the
        # fixed tours as a new warm-start seed. Gated by
        # --post_revision_lens / --post_revision_iters.
        if getattr(opts, "post_revision_lens", []) and getattr(
            opts, "post_revision_iters", []
        ):
            if len(opts.post_revision_lens) != len(opts.post_revision_iters):
                raise ValueError(
                    "--post_revision_lens and --post_revision_iters must "
                    "have the same length."
                )

            print(
                "[INFO] Current VRAM usage: {:.2f} GB".format(
                    torch.cuda.memory_allocated() / 1e9
                )
            )

            print(
                f"[POST] loading {len(opts.post_revision_lens)} post-revisors: "
                f"sizes={opts.post_revision_lens}, iters={opts.post_revision_iters}"
            )
            # Release the cached GPU memory held by the original revisors
            # (already out of scope but PyTorch's allocator holds onto it)
            # so we don't OOM when loading the post-revisors below.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            post_revisers = []
            for post_size in opts.post_revision_lens:
                post_path = (
                    f"pretrained/Reviser-stage2/reviser_{post_size}/epoch-299.pt"
                )
                post_rev, _ = load_model(post_path, is_local=True)
                post_rev.to(opts.device)
                post_rev.eval()
                post_rev.set_decode_type(opts.decode_strategy)
                post_revisers.append(post_rev)

            print(
                "[INFO] Current VRAM usage: {:.2f} GB".format(
                    torch.cuda.memory_allocated() / 1e9
                )
            )

            problem = load_problem("tsp")
            get_cost_func = lambda input, pi: problem.get_costs(
                input, pi, return_local=True
            )

            cost_before = (tours[:, 1:] - tours[:, :-1]).norm(p=2, dim=2).sum(1) + (
                tours[:, 0] - tours[:, -1]
            ).norm(p=2, dim=1)

            # Wrap the post-revision pass in torch.no_grad() to mirror the
            # original ``reconnect`` pass — without it, every revisor
            # forward pass builds the autograd graph and roughly doubles
            # peak VRAM during inference.
            with torch.no_grad():
                tours, _ = run_post_revision(
                    tours=tours,
                    revisers=post_revisers,
                    get_cost_func=get_cost_func,
                    opts=opts,
                    post_revision_lens=opts.post_revision_lens,
                    post_revision_iters=opts.post_revision_iters,
                    batch_size=getattr(opts, "post_revision_batch_size", None),
                    iter_log_path=getattr(opts, "iter_cost_log", ""),
                    run_metadata=run_metadata,
                )

            cost_after = (tours[:, 1:] - tours[:, :-1]).norm(p=2, dim=2).sum(1) + (
                tours[:, 0] - tours[:, -1]
            ).norm(p=2, dim=1)
            pct = 100.0 * (cost_after.mean().item() / cost_before.mean().item() - 1.0)
            print(
                f"[POST] revisor re-warm-start (layers="
                f"{len(opts.post_revision_lens)}): "
                f"cost {cost_before.mean().item():.4f} -> "
                f"{cost_after.mean().item():.4f} ({pct:+.2f}%)"
            )

            pct = 100.0 * (
                cost_after.mean().item() / fix_info["initial_cost"].mean().item() - 1.0
            )
            print(
                f"[POST] total perf gain: "
                f"cost {fix_info['initial_cost'].mean().item():.4f} -> "
                f"{cost_after.mean().item():.4f} ({pct:+.2f}%)"
            )

            # Snapshot `postrev` — tours after the optional revisor
            # re-warm-start pass. The `postrev` file is intentionally
            # distinct from `postfix` so downstream comparison can tell
            # whether revisor re-warm-start helped or hurt the gap.
            if save_dir and snapshot_tag:
                _save_tours_snapshot(
                    save_dir, snapshot_tag, "postrev", coords_for_dump, tours, opts
                )

            # Release post-revisor weights and allocator blocks now that
            # the post-revision pass is done, so subsequent steps
            # (visualization, plotting) start from a clean slate.
            del post_revisers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
