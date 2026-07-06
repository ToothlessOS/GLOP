import warnings
import torch
import numpy as np
import os
import json
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import torch.nn.functional as F
import math
import time


def load_problem(name):
    from problems import TSP, LOCAL

    problem = {
        "local": LOCAL,
        "tsp": TSP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def torch_load_cpu(load_path):
    return torch.load(
        load_path, map_location=lambda storage, loc: storage
    )  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print("  [*] Loading model from {}".format(load_path))

    load_data = torch.load(
        os.path.join(os.getcwd(), load_path), map_location=lambda storage, loc: storage
    )

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get("optimizer", None)
        load_model_state_dict = load_data.get("model", load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, "r") as f:
        args = json.load(f)

    # Backwards compatibility
    if "data_distribution" not in args:
        args["data_distribution"] = None
        probl, *dist = args["problem"].split("_")
        if probl == "op":
            args["problem"] = probl
            args["data_distribution"] = dist[0]
    return args


def load_model(path, epoch=None, is_local=True):

    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == ".pt"
            )
        model_filename = os.path.join(path, "epoch-{}.pt".format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)

    args = load_args(os.path.join(path, "args.json"))

    if is_local:

        from nets.attention_local import AttentionModel

        model = AttentionModel(
            args["embedding_dim"],
            args["hidden_dim"],
            load_problem("local"),
            n_encode_layers=args["n_encode_layers"],
            mask_inner=True,
            mask_logits=True,
            normalization=args["normalization"],
            tanh_clipping=args["tanh_clipping"],
            checkpoint_encoder=args.get("checkpoint_encoder", False),
            shrink_size=args.get("shrink_size", None),
        )

    else:
        raise NotImplementedError

    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get("model", {})})

    model, *_ = _load_model_file(model_filename, model)

    model.eval()  # Put in eval mode

    return model, args


def parse_softmax_temperature(raw_temp):
    # Load from file
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    assert opts.cpus is not None
    num_cpus = opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, "offset", None)
    if offset is None:
        offset = 0
    ds = dataset[offset : (offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = Pool if use_multiprocessing and num_cpus > 1 else ThreadPool
    with pool_cls(num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(
                    func,
                    [
                        (directory, str(i + offset).zfill(w), *problem)
                        for i, problem in enumerate(ds)
                    ],
                ),
                total=len(ds),
                mininterval=opts.progress_bar_mininterval,
            )
        )

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def do_batch_rep(v, n):
    if isinstance(v, dict):
        return {k: do_batch_rep(v_, n) for k, v_ in v.items()}
    elif isinstance(v, list):
        return [do_batch_rep(v_, n) for v_ in v]
    elif isinstance(v, tuple):
        return tuple(do_batch_rep(v_, n) for v_ in v)

    return v[None, ...].expand(n, *v.size()).contiguous().view(-1, *v.size()[1:])


######## LCP-TSP ##########
def decomposition(seeds, coordinate_dim, revision_len, offset, shift_len=1):
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


def _summarize_cost_across_width(cost, opts):
    """Return (best, avg, count) for the --width random restarts.

    `cost` has shape (width * eval_batch_size,) before pruning, or
    (eval_batch_size,) after pruning. Reshape to (width, eval_batch_size),
    reduce along dim=0 to get best/avg per instance, then take the mean
    across instances for a scalar summary. Falls back to (mean, mean, n)
    when the width axis is absent (width <= 1, or the tensor has already
    been pruned to a single restart per element).
    """
    width = getattr(opts, "width", 1)
    eval_batch_size = getattr(opts, "eval_batch_size", 1)
    # Width axis is present only when the tensor really holds
    # width * eval_batch_size instances (i.e. not pruned yet).
    has_width_axis = (
        width is not None and width > 1 and cost.numel() == width * eval_batch_size
    )
    if not has_width_axis:
        # Single restart, or already pruned — best == avg == mean.
        mean = cost.mean().item()
        return mean, mean, cost.numel()
    grouped = cost.reshape(width, eval_batch_size)
    best = grouped.min(dim=0).values.mean().item()
    avg = grouped.mean(dim=0).mean().item()
    return best, avg, grouped.numel()


def coordinate_transformation(x):
    input = x.clone()
    max_x, indices_max_x = input[:, :, 0].max(dim=1)
    max_y, indices_max_y = input[:, :, 1].max(dim=1)
    min_x, indices_min_x = input[:, :, 0].min(dim=1)
    min_y, indices_min_y = input[:, :, 1].min(dim=1)
    # shapes: (batch_size, ); (batch_size, )

    diff_x = max_x - min_x
    diff_y = max_y - min_y
    xy_exchanged = diff_y > diff_x

    # shift to zero
    input[:, :, 0] -= (min_x).unsqueeze(-1)
    input[:, :, 1] -= (min_y).unsqueeze(-1)

    # exchange coordinates for those diff_y > diff_x
    input[xy_exchanged, :, 0], input[xy_exchanged, :, 1] = (
        input[xy_exchanged, :, 1],
        input[xy_exchanged, :, 0],
    )

    # scale to (0, 1)
    scale_degree = torch.max(diff_x, diff_y)
    scale_degree = scale_degree.view(input.shape[0], 1, 1)
    input /= scale_degree + 1e-10
    return input


def revision(
    opts,
    revision_cost_func,
    reviser,
    decomposed_seeds,
    original_subtour,
    iter=None,
    embeddings=None,
):

    # tour length of segment TSPs
    reviser_size = original_subtour.shape[0]
    init_cost = revision_cost_func(decomposed_seeds, original_subtour)

    # coordinate transformation
    transformed_seeds = coordinate_transformation(decomposed_seeds)
    # augmentation
    if not opts.no_aug:
        seed2 = torch.cat(
            (1 - transformed_seeds[:, :, [0]], transformed_seeds[:, :, [1]]), dim=2
        )
        seed3 = torch.cat(
            (transformed_seeds[:, :, [0]], 1 - transformed_seeds[:, :, [1]]), dim=2
        )
        seed4 = torch.cat(
            (1 - transformed_seeds[:, :, [0]], 1 - transformed_seeds[:, :, [1]]), dim=2
        )
        augmented_seeds = torch.cat((transformed_seeds, seed2, seed3, seed4), dim=0)
    else:
        augmented_seeds = transformed_seeds

    if iter is None:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(
            augmented_seeds, return_pi=True
        )
    elif iter == 0:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2, embeddings = reviser(
            augmented_seeds, return_pi=True, return_embedding=True
        )
    else:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(
            augmented_seeds, return_pi=True, embeddings=embeddings
        )

    if not opts.no_aug:
        _, better_tour_idx = (
            torch.cat([cost_revised1, cost_revised2], dim=0).reshape(8, -1).min(dim=0)
        )
        sub_tour = torch.cat([sub_tour1, sub_tour2], dim=0).reshape(
            8, -1, reviser_size
        )[better_tour_idx, torch.arange(sub_tour1.shape[0] // 4), :]
    else:
        _, better_tour_idx = torch.stack((cost_revised1, cost_revised2)).min(dim=0)
        sub_tour = torch.stack((sub_tour1, sub_tour2))[
            better_tour_idx, torch.arange(sub_tour1.shape[0])
        ]

    cost_revised, _ = reviser.problem.get_costs(decomposed_seeds, sub_tour)
    reduced_cost = init_cost - cost_revised

    sub_tour[reduced_cost < 0] = original_subtour
    decomposed_seeds = decomposed_seeds.gather(
        1, sub_tour.unsqueeze(-1).expand_as(decomposed_seeds)
    )

    if embeddings is not None:
        if not opts.no_aug:
            embeddings = embeddings.gather(
                1, sub_tour.repeat(4, 1).unsqueeze(-1).expand_as(embeddings)
            )
        else:
            embeddings = embeddings.gather(
                1, sub_tour.unsqueeze(-1).expand_as(embeddings)
            )

    return decomposed_seeds, embeddings


# ---------------------------------------------------------------------------
# Hypernode (block-level) 2-opt
# ---------------------------------------------------------------------------
# Treat each revision_len-aligned subproblem as a "hypernode". Repeatedly
# search for a pair of non-adjacent blocks whose position swap strictly
# lowers the closed-loop tour cost; this only touches the four boundary
# edges around the two swapped blocks.
# ---------------------------------------------------------------------------

_BLOCK_SWAP_CACHE = {}


def _block_pair_indices(sub_counts, device):
    """Return (j_idx, k_idx) listing every non-adjacent block pair (j, k)
    with j+1 < k < sub_counts. Cached by (sub_counts, device).

    The wraparound pair (j=0, k=sub_counts-1) is excluded because it
    merges two of the four boundary edges into the same physical
    edge (the wraparound edge), which would break the 4-edge delta
    formula. That swap can still be reached indirectly via two
    intermediate swaps.
    """
    key = (sub_counts, str(device))
    cached = _BLOCK_SWAP_CACHE.get(key)
    if cached is not None:
        return cached
    j_list, k_list = [], []
    for j in range(sub_counts):
        for k in range(j + 2, sub_counts):
            if j == 0 and k == sub_counts - 1:
                continue
            j_list.append(j)
            k_list.append(k)
    j_idx = torch.tensor(j_list, device=device, dtype=torch.long)
    k_idx = torch.tensor(k_list, device=device, dtype=torch.long)
    _BLOCK_SWAP_CACHE[key] = (j_idx, k_idx)
    return j_idx, k_idx


def _block_swap_delta(seeds_blocks, j_idx, k_idx):
    """For every (batch, pair), compute the tour-cost delta of swapping
    blocks j<->k. Negative = improving swap.

    seeds_blocks: (B, S, L, D)
    j_idx, k_idx: (P,) long tensors with j_idx[p] + 1 < k_idx[p] < S
    Returns:     (B, P) delta tensor.
    """
    first = seeds_blocks[:, :, 0, :]  # (B, S, D)
    last = seeds_blocks[:, :, -1, :]  # (B, S, D)
    S = seeds_blocks.shape[1]

    # Boundary edge costs: outof[j] = || last[j] - first[(j+1) % S] ||
    nxt_first = torch.roll(first, shifts=-1, dims=1)
    outof = (last - nxt_first).norm(p=2, dim=-1)  # (B, S)

    # OLD sum of the four boundary edges that change:
    #   outof[j-1] + outof[j] + outof[k-1] + outof[k]
    prev_outof = torch.roll(outof, shifts=1, dims=1)
    old = (
        prev_outof[:, j_idx] + outof[:, j_idx] + prev_outof[:, k_idx] + outof[:, k_idx]
    )  # (B, P)

    # NEW edges after swap:
    #   (j-1 -> k):   || last[j-1]   - first[k]   ||
    #   (k   -> j+1): || last[k]     - first[(j+1) % S] ||
    #   (k-1 -> j):   || last[k-1]   - first[j]   ||
    #   (j   -> k+1): || last[j]     - first[(k+1) % S] ||
    prev_last = torch.roll(
        last, shifts=1, dims=1
    )  # (B, S, D); prev_last[..., j] = last[..., (j-1) % S]

    # torch.roll does not accept tensor shifts, so use advanced indexing.
    j_plus_1 = (j_idx + 1) % S  # (P,)
    k_plus_1 = (k_idx + 1) % S  # (P,)
    nxt_first_j = first[:, j_plus_1, :]  # (B, P, D); first[(j+1) % S]
    nxt_first_k = first[:, k_plus_1, :]  # (B, P, D); first[(k+1) % S]

    new = (
        (prev_last[:, j_idx, :] - first[:, k_idx, :]).norm(p=2, dim=-1)
        + (last[:, k_idx, :] - nxt_first_j).norm(p=2, dim=-1)
        + (prev_last[:, k_idx, :] - first[:, j_idx, :]).norm(p=2, dim=-1)
        + (last[:, j_idx, :] - nxt_first_k).norm(p=2, dim=-1)
    )  # (B, P)

    return new - old  # (B, P); negative = good


def _block_swap_two_opt(seeds, revision_len, offset, max_iter=10, eps=-1e-9):
    """Block-level (hypernode) 2-opt.

    Treat each revision_len-aligned subproblem as a hypernode. Repeatedly
    pick the best single block-pair swap that strictly lowers the
    closed-loop tour cost; stop when no improving swap exists or
    max_iter is reached. The offset tail participates as its own
    subproblem (padded to revision_len for uniform indexing, unpadded
    at the end).
    """
    B, N, D = seeds.shape
    n_aligned = N - offset
    assert n_aligned % revision_len == 0
    S = n_aligned // revision_len + (1 if offset > 0 else 0)
    if S < 3:
        return seeds

    # --- pad the tail so all blocks have uniform size L ---
    if offset > 0:
        tail = seeds[:, -offset:, :]  # (B, offset, D)
        pad = tail[:, -1:, :].repeat(1, revision_len - offset, 1)
        tail_padded = torch.cat([tail, pad], dim=1)
        aligned = torch.cat([seeds[:, :-offset, :], tail_padded], dim=1)  # (B, S*L, D)
    else:
        aligned = seeds

    aligned = aligned.reshape(B, S, revision_len, D)

    # --- index pairs ---
    j_idx, k_idx = _block_pair_indices(S, seeds.device)

    # aligned holds the ORIGINAL blocks: aligned[b, j] is the original
    # block j (j = 0..S-1, where S-1 is the padded tail). We never mutate
    # aligned; instead we track the current ordering in `perm` where
    # perm[b, i] is the original block index at current position i. The
    # delta is computed against the permuted order built on-the-fly.
    perm = torch.arange(S, device=seeds.device).expand(B, S).clone()
    rows = torch.arange(B, device=seeds.device)
    for it in range(max_iter):
        if j_idx.numel() == 0:
            break
        # Build the current permuted view of aligned (without overwriting it).
        aligned_perm = aligned.gather(
            1, perm.view(B, S, 1, 1).expand(-1, -1, revision_len, D)
        )
        delta = _block_swap_delta(aligned_perm, j_idx, k_idx)  # (B, P)
        best_delta, best_pair = delta.min(dim=1)  # (B,), (B,)
        improving = best_delta < eps
        if not improving.any():
            break

        # j_sel / k_sel are POSITIONS in the current permuted view. Swap
        # perm entries at those positions: this is the canonical swap move
        # on the hypernode-level tour.
        j_sel = j_idx[best_pair]  # (B,)
        k_sel = k_idx[best_pair]  # (B,)
        old_j_perm = perm[rows, j_sel].clone()
        old_k_perm = perm[rows, k_sel].clone()
        perm[rows, j_sel] = torch.where(improving, old_k_perm, perm[rows, j_sel])
        perm[rows, k_sel] = torch.where(improving, old_j_perm, perm[rows, k_sel])

    # --- flatten back and unpad the tail ---
    if offset > 0:
        # Reconstruct the output (B, N, D) by walking `perm` and pulling
        # each block's content from `aligned` (originals) or `tail`.
        # aligned still holds the original blocks; perm[b, i] gives the
        # original block index at output position i.
        out = torch.empty(
            B, n_aligned + offset, D, device=seeds.device, dtype=seeds.dtype
        )
        for b in range(B):
            pos = 0
            for i in range(S):
                orig_idx = int(perm[b, i].item())
                if orig_idx == S - 1:
                    block = tail[b]  # (offset, D)
                else:
                    block = aligned[b, orig_idx]  # (L, D)
                out[b, pos : pos + block.shape[0], :] = block
                pos += block.shape[0]
        return out
    else:
        # No offset: same as the offset branch but using aligned for every
        # block (all blocks are size revision_len).
        out = torch.empty(B, n_aligned, D, device=seeds.device, dtype=seeds.dtype)
        for b in range(B):
            pos = 0
            for i in range(S):
                block = aligned[b, int(perm[b, i].item())]  # (L, D)
                out[b, pos : pos + block.shape[0], :] = block
                pos += block.shape[0]
        return out


def _run_diagnostics(seeds, revision_len):
    """Run heuristic-validity diagnostics on the sub-TSP `seeds` and print a
    single summary line matching the existing `[Revisor L=...]` format.

    Returns a dict so callers can stash the per-batch numbers in `stats_list`.
    Lazy-imports `utils.diagnosis` to keep the import surface minimal when
    diagnostics are disabled.
    """
    from utils.diagnosis import check_no_self_intersection, check_convex_hull

    has_inter, _, frac_inter, n_pairs, _ = check_no_self_intersection(seeds)
    consistency, _ = check_convex_hull(seeds)

    B = seeds.shape[0]
    n_intersecting = int(has_inter.sum().item())
    n_consistent = int((consistency > 0.5).sum().item())
    mean_frac = float(frac_inter.mean().item()) if frac_inter.numel() else 0.0
    mean_pairs = float(n_pairs.float().mean().item()) if n_pairs.numel() else 0.0

    print(
        "[Revisor L={:>4}] diag: self-intersect={}/{} ({:5.1f}%), "
        "convex-hull-OK={}/{} ({:5.1f}%), "
        "mean-frac-intersect={:.4f}, mean-valid-pairs={:.0f}".format(
            revision_len,
            n_intersecting,
            B,
            100.0 * n_intersecting / max(B, 1),
            n_consistent,
            B,
            100.0 * n_consistent / max(B, 1),
            mean_frac,
            mean_pairs,
        )
    )
    return {
        "n_intersecting": n_intersecting,
        "n_consistent": n_consistent,
        "total": B,
        "mean_frac_intersections": mean_frac,
        "mean_valid_pairs": mean_pairs,
    }


def LCP_TSP(
    seeds,
    cost_func,
    reviser,
    revision_len,
    revision_iter,
    opts,
    shift_len,
    layer_id=None,
    stats_list=None,
):

    batch_size, num_nodes, coordinate_dim = seeds.shape
    offset = num_nodes % revision_len
    embeddings = None  # used only in case problem_size == revision_len for efficiency

    # NEW: heuristic-validity diagnostics on the input sub-TSP. Gated by
    # --diagnose / --no_diagnose. Skipped when `seeds` is a single-segment
    # subproblem (num_nodes <= revision_len would mean it is already a single
    # revision target; we still run if num_nodes > revision_len since that's
    # the case where decompose-on-edge becomes interesting).
    if getattr(opts, "diagnose", False):
        diag = _run_diagnostics(seeds, revision_len)
        if stats_list is not None:
            stats_list.append({"iter_id": -1, **diag})

    layer_start = time.time()  # NEW: per-layer wall-clock start
    for i in range(revision_iter):

        decomposed_seeds, offset_seed = decomposition(
            seeds, coordinate_dim, revision_len, offset, shift_len
        )

        original_subtour = torch.arange(0, revision_len, dtype=torch.long).to(
            decomposed_seeds.device
        )

        if revision_len == num_nodes:
            decomposed_seeds_revised, embeddings = revision(
                opts,
                cost_func,
                reviser,
                decomposed_seeds,
                original_subtour,
                iter=i,
                embeddings=embeddings,
            )
            embeddings = torch.cat(
                [embeddings[:, shift_len:], embeddings[:, :shift_len]], 1
            )  # roll the embeddings
        else:
            decomposed_seeds_revised, _ = revision(
                opts, cost_func, reviser, decomposed_seeds, original_subtour
            )

        # decomposed_seeds_revised: (batch_size * num_segments, revision_len, coordinate_dim)
        seeds = decomposed_seeds_revised.reshape(batch_size, -1, coordinate_dim)
        if offset_seed is not None:
            seeds = torch.cat(
                [seeds, offset_seed], dim=1  # Append the tail segment
            )  # seeds: (batch_size, num_nodes, coordinate_dim)

        # Hypernode (block-level) 2-opt: treat each revision_len-aligned
        # subproblem as a hypernode and search pairwise block swaps that
        # lower the closed-loop tour cost. The offset tail participates
        # as its own subproblem. Gated by --do_block_2opt / --no_block_2opt.
        cost_before_2opt = (seeds[:, 1:] - seeds[:, :-1]).norm(p=2, dim=2).sum(1) + (
            seeds[:, 0] - seeds[:, -1]
        ).norm(p=2, dim=1)
        do_block_2opt = getattr(opts, "do_block_2opt", True)
        block_swap_max_iter = getattr(opts, "block_swap_max_iter", 10)
        if do_block_2opt:
            seeds = _block_swap_two_opt(
                seeds, revision_len, offset, max_iter=block_swap_max_iter
            )
        cost_after_2opt = (seeds[:, 1:] - seeds[:, :-1]).norm(p=2, dim=2).sum(1) + (
            seeds[:, 0] - seeds[:, -1]
        ).norm(p=2, dim=1)
        best_before, avg_before, _ = _summarize_cost_across_width(
            cost_before_2opt, opts
        )
        best_after, avg_after, count_after = _summarize_cost_across_width(
            cost_after_2opt, opts
        )

        # NEW: per-iteration logging — closed-loop tour cost across all --width restarts
        iter_elapsed = time.time() - layer_start
        if do_block_2opt:
            print(
                "[Revisor L={:>4}] iter {:>2}/{:>2}: best={:.4f}, avg={:.4f}, 2-opt Δavg={:+.4f}, 2-opt Δbest={:+.4f}, elapsed={:6.2f}s".format(
                    revision_len,
                    i + 1,
                    revision_iter,
                    best_after,
                    avg_after,
                    avg_after - avg_before,
                    best_after - best_before,
                    iter_elapsed,
                )
            )
        else:
            print(
                "[Revisor L={:>4}] iter {:>2}/{:>2}: best={:.4f}, avg={:.4f}, elapsed={:6.2f}s".format(
                    revision_len,
                    i + 1,
                    revision_iter,
                    best_after,
                    avg_after,
                    iter_elapsed,
                )
            )

        # NEW: heuristic-validity diagnostics on the input sub-TSP. Gated by
        # --diagnose / --no_diagnose. Skipped when `seeds` is a single-segment
        # subproblem (num_nodes <= revision_len would mean it is already a single
        # revision target; we still run if num_nodes > revision_len since that's
        # the case where decompose-on-edge becomes interesting).
        if getattr(opts, "diagnose", False):
            diag = _run_diagnostics(seeds, revision_len)
            if stats_list is not None:
                stats_list.append({"iter_id": -1, **diag})

        # Accumulate per-iter stats for downstream aggregation / plotting.
        # Store as raw sums so that aggregating across batches gives the
        # correct weighted average (batches may have different sizes).
        if stats_list is not None:
            stats_list.append(
                {
                    "layer_id": layer_id,
                    "iter_id": i,
                    "sum_best": best_after * count_after,
                    "sum_avg": avg_after * count_after,
                    "count": count_after,
                }
            )
    return seeds


def run_post_revision(
    tours,
    revisers,
    get_cost_func,
    opts,
    post_revision_lens,
    post_revision_iters,
    stats_list=None,
):
    """Treat ``tours`` (B, N, 2) as a warm-start seed and run additional
    revisor passes. Width is implicitly 1 (no width-axis reduction).
    Standalone helper — not wired into the main pipeline by default;
    call it from your own script after :func:`fix_intersections_via_2opt`
    or any other pre-processing.

    Args:
        tours: (B, N, 2) coordinate-ordered warm-start seed.
        revisers: list of nn.Module revisor models, one per layer.
        get_cost_func: same callable shape used by :func:`reconnect`
            (``(input, pi) -> cost``).
        opts: argparse Namespace; needs ``.revision_lens``,
            ``.revision_iters``, ``.decode_strategy``, ``.device``, etc.
        post_revision_lens: list of int; revisor window sizes per layer.
        post_revision_iters: list of int; iterations per layer. Must have
            the same length as ``post_revision_lens``.
        stats_list: optional list mutated in place with per-iter stats
            (same format as :func:`LCP_TSP`).

    Returns:
        (tours_out, costs_out) with shapes ``(B, N, 2)`` and ``(B,)``.
    """
    seed = tours
    problem_size = seed.size(1)

    if len(revisers) == 0:
        costs_out = (
            (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1)
            + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
        )
        return seed, costs_out

    for layer_id, (reviser, rl, ri) in enumerate(
        zip(revisers, post_revision_lens, post_revision_iters)
    ):
        print(
            "[POST Revisor L={:>4}] starting layer {}/{} "
            "(revision_iters={}, width=1)".format(
                rl, layer_id + 1, len(revisers), ri
            )
        )

        start_time = time.time()
        shift_len = max(rl // ri, 1)
        # layer_id + 1000 sentinel keeps post-revisor entries out of the
        # same (layer_id, iter_id) namespace as the original revisor.
        seed = LCP_TSP(
            seed,
            get_cost_func,
            reviser,
            rl,
            ri,
            opts=opts,
            shift_len=shift_len,
            layer_id=layer_id + 1000,
            stats_list=stats_list,
        )
        layer_elapsed = time.time() - start_time

        cost_layer = (
            (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1)
            + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
        )
        # width=1: best == avg == mean.
        print(
            "[POST Revisor L={:>4}] layer {}/{} done: "
            "time={:6.2f}s, cost={:.4f}".format(
                rl, layer_id + 1, len(revisers), layer_elapsed, cost_layer.mean().item()
            )
        )

    costs_out = (
        (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1)
        + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
    )
    assert seed.shape == (seed.shape[0], problem_size, 2)
    assert costs_out.shape == (seed.shape[0],)
    return seed, costs_out


def reconnect(
    get_cost_func,
    batch,
    opts,
    revisers,
    stats_list=None,
):
    seed = batch
    problem_size = seed.size(1)
    if len(revisers) == 0:
        cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (
            seed[:, 0] - seed[:, -1]
        ).norm(p=2, dim=1)

    for revision_id in range(len(revisers)):
        assert opts.revision_lens[revision_id] <= seed.size(1)

        # NEW: layer-start header
        print(
            "[Revisor L={:>4}] starting layer {}/{} (revision_iters={}, width={})".format(
                opts.revision_lens[revision_id],
                revision_id + 1,
                len(revisers),
                opts.revision_iters[revision_id],
                getattr(opts, "width", 1),
            )
        )

        start_time = time.time()
        shift_len = max(
            opts.revision_lens[revision_id] // opts.revision_iters[revision_id], 1
        )
        seed = LCP_TSP(
            seed,
            get_cost_func,
            revisers[revision_id],
            opts.revision_lens[revision_id],
            opts.revision_iters[revision_id],
            opts=opts,
            shift_len=shift_len,
            layer_id=revision_id,
            stats_list=stats_list,
        )
        cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (
            seed[:, 0] - seed[:, -1]
        ).norm(p=2, dim=1)
        duration = time.time() - start_time

        # NEW: layer-end summary — best/avg across --width restarts, aggregated over eval batch
        best, avg, _ = _summarize_cost_across_width(cost_revised, opts)
        print(
            "[Revisor L={:>4}] layer {}/{} done: time={:6.2f}s, best={:.4f}, avg={:.4f}".format(
                opts.revision_lens[revision_id],
                revision_id + 1,
                len(revisers),
                duration,
                best,
                avg,
            )
        )

        if (
            revision_id == 0 and not opts.no_prune
        ):  # eliminate the underperforming ones after the first round of revisions
            cost_revised, cost_revised_minidx = cost_revised.reshape(
                -1, opts.eval_batch_size
            ).min(
                0
            )  # width, bs
            seed = seed.reshape(-1, opts.eval_batch_size, seed.shape[-2], 2)[
                cost_revised_minidx, torch.arange(opts.eval_batch_size)
            ]
    if opts.no_prune:
        cost_revised, cost_revised_minidx = cost_revised.reshape(
            -1, opts.eval_batch_size
        ).min(0)
        seed = seed.reshape(-1, opts.eval_batch_size, seed.shape[-2], 2)[
            cost_revised_minidx, torch.arange(opts.eval_batch_size)
        ]
    assert cost_revised.shape == (opts.eval_batch_size,)
    assert seed.shape == (opts.eval_batch_size, problem_size, 2)

    return seed, cost_revised


def sample_many():
    raise NotImplementedError
