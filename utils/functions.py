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
        'local': LOCAL,
        'tsp': TSP,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
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
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, "{} is not a valid directory or file".format(path)
   


    args = load_args(os.path.join(path, 'args.json'))

    
    if is_local:

        from nets.attention_local import AttentionModel
        model= AttentionModel(
        args['embedding_dim'],
        args['hidden_dim'],
        load_problem('local'),
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None),
    )

    else:
        raise NotImplementedError

    # Overwrite model parameters by parameters to load
    load_data = torch_load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})

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
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

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
def decomposition(seeds, coordinate_dim, revision_len, offset, shift_len = 1):
    # change decomposition point
    seeds = torch.cat([seeds[:, shift_len:],seeds[:, :shift_len]], 1)

    if offset!=0:
        decomposed_seeds = seeds[:, :-offset]
        offset_seeds = seeds[:,-offset:]
    else:
        decomposed_seeds = seeds
        offset_seeds = None
    # decompose original seeds
    decomposed_seeds = decomposed_seeds.reshape(-1, revision_len, coordinate_dim)
    return decomposed_seeds, offset_seeds

def coordinate_transformation(x):
    input = x.clone()
    max_x, indices_max_x = input[:,:,0].max(dim=1)
    max_y, indices_max_y = input[:,:,1].max(dim=1)
    min_x, indices_min_x = input[:,:,0].min(dim=1)
    min_y, indices_min_y = input[:,:,1].min(dim=1)
    # shapes: (batch_size, ); (batch_size, )
    
    diff_x = max_x - min_x
    diff_y = max_y - min_y
    xy_exchanged = diff_y > diff_x

    # shift to zero
    input[:, :, 0] -= (min_x).unsqueeze(-1)
    input[:, :, 1] -= (min_y).unsqueeze(-1)

    # exchange coordinates for those diff_y > diff_x
    input[xy_exchanged, :, 0], input[xy_exchanged, :, 1] =  input[xy_exchanged, :, 1], input[xy_exchanged, :, 0]
    
    # scale to (0, 1)
    scale_degree = torch.max(diff_x, diff_y)
    scale_degree = scale_degree.view(input.shape[0], 1, 1)
    input /= scale_degree + 1e-10
    return input

def revision(opts, revision_cost_func, reviser, decomposed_seeds, original_subtour, iter=None, embeddings=None):

    # tour length of segment TSPs
    reviser_size = original_subtour.shape[0]
    init_cost = revision_cost_func(decomposed_seeds, original_subtour)
    
    # coordinate transformation
    transformed_seeds = coordinate_transformation(decomposed_seeds)
    # augmentation
    if not opts.no_aug:
        seed2 = torch.cat((1 - transformed_seeds[:, :, [0]], transformed_seeds[:, :, [1]]), dim=2)
        seed3 = torch.cat((transformed_seeds[:, :, [0]], 1 - transformed_seeds[:, :, [1]]), dim=2)
        seed4 = torch.cat((1 - transformed_seeds[:, :, [0]], 1 - transformed_seeds[:, :, [1]]), dim=2)
        augmented_seeds = torch.cat((transformed_seeds, seed2, seed3, seed4), dim=0)
    else:
        augmented_seeds = transformed_seeds

    if iter is None:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(augmented_seeds, return_pi=True)
    elif iter == 0:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2, embeddings = reviser(augmented_seeds, return_pi=True, return_embedding=True)
    else:
        cost_revised1, sub_tour1, cost_revised2, sub_tour2 = reviser(augmented_seeds, return_pi=True, embeddings=embeddings)
    
    if not opts.no_aug:
        _, better_tour_idx = torch.cat([cost_revised1, cost_revised2], dim=0).reshape(8,-1).min(dim=0)
        sub_tour = torch.cat([sub_tour1, sub_tour2], dim=0).reshape(8,-1, reviser_size)[better_tour_idx, torch.arange(sub_tour1.shape[0]//4), :]
    else:
        _, better_tour_idx = torch.stack((cost_revised1, cost_revised2)).min(dim=0)
        sub_tour = torch.stack((sub_tour1, sub_tour2))[better_tour_idx, torch.arange(sub_tour1.shape[0])]

    cost_revised, _ = reviser.problem.get_costs(decomposed_seeds, sub_tour)
    reduced_cost = init_cost - cost_revised
    
    sub_tour[reduced_cost < 0] = original_subtour
    decomposed_seeds = decomposed_seeds.gather(1, sub_tour.unsqueeze(-1).expand_as(decomposed_seeds))
    
    if embeddings is not None:
        if not opts.no_aug:
            embeddings = embeddings.gather(1, sub_tour.repeat(4, 1).unsqueeze(-1).expand_as(embeddings))
        else:
            embeddings = embeddings.gather(1, sub_tour.unsqueeze(-1).expand_as(embeddings))

    return decomposed_seeds, embeddings

def LCP_TSP(
    seeds,
    cost_func,
    reviser,
    revision_len,
    revision_iter,
    opts,
    shift_len,
    bench_logger=None,
    revision_id=None,
    ):

    batch_size, num_nodes, coordinate_dim = seeds.shape
    offset = num_nodes % revision_len
    embeddings = None # used only in case problem_size == revision_len for efficiency

    # Per-pass chunk_2opt settings. When --chunk_2opt_after_each_iter is on
    # (default), we run chunk_2opt after each individual reviser pass — not
    # just after each cascade level. This is the more aggressive mode.
    chunk_2opt_on = getattr(opts, 'chunk_2opt', False)
    chunk_2opt_per_iter = getattr(opts, 'chunk_2opt_after_each_iter', True)
    chunk_2opt_iters = getattr(opts, 'chunk_2opt_iters', 10)
    chunk_2opt_override = getattr(opts, 'chunk_2opt_chunk_size', 0)

    for i in range(revision_iter):

        decomposed_seeds, offset_seed = decomposition(seeds,
                                        coordinate_dim,
                                        revision_len,
                                        offset,
                                        shift_len
                                        )

        original_subtour = torch.arange(0, revision_len, dtype=torch.long).to(decomposed_seeds.device)

        if revision_len == num_nodes:
            decomposed_seeds_revised, embeddings = revision(opts, cost_func, reviser, decomposed_seeds, original_subtour, iter=i, embeddings=embeddings)
            embeddings = torch.cat([embeddings[:, shift_len:],embeddings[:, :shift_len]], 1) # roll the embeddings
        else:
            decomposed_seeds_revised, _ = revision(opts, cost_func, reviser, decomposed_seeds, original_subtour)

        seeds = decomposed_seeds_revised.reshape(batch_size, -1, coordinate_dim)
        if offset_seed is not None:
            seeds = torch.cat([seeds,offset_seed], dim=1)

        # Per-pass chunk_2opt on the SAME chunks as the SHPP decomposition:
        # by default `cs = revision_len` (the SHPP chunk size). The user can
        # override this with `--chunk_2opt_chunk_size` to use a different
        # chunking granularity (e.g., 10-node chunks for a finer search after
        # a 20-node SHPP decomposition). chunk_2opt handles the offset != 0
        # case natively (the partial chunk is a regular 2-opt unit), so we
        # always call it when enabled.
        if chunk_2opt_on and chunk_2opt_per_iter:
            cs = chunk_2opt_override or revision_len
            # When the chunk_2opt chunk size is smaller than the SHPP chunk
            # size, the SHPP shift_len rotation only visits `revision_iter`
            # of the `cs` possible alignments. Auto-enable exhaustive_shifts
            # in that case to cover all alignments. The user can override
            # with `--no_chunk_2opt_exhaustive_shifts` for speed.
            if getattr(opts, 'no_chunk_2opt_exhaustive_shifts', False):
                exhaustive_shifts = False
            else:
                exhaustive_shifts = (cs < revision_len)
            cost_before = (seeds[:, 1:] - seeds[:, :-1]).norm(p=2, dim=2).sum(1) \
                        + (seeds[:, 0] - seeds[:, -1]).norm(p=2, dim=1)
            seeds, cost_after, c2o_stats = chunk_2opt(
                seeds,
                chunk_size=cs,
                n_iters=chunk_2opt_iters,
                include_flip=getattr(opts, 'chunk_2opt_flip', False),
                include_close=True,
                exhaustive_shifts=exhaustive_shifts,
            )
            if bench_logger is not None:
                stage = f'chunk_2opt_L{revision_id}_iter{i}' if revision_id is not None \
                        else f'chunk_2opt_iter{i}'
                bench_logger.add_stage(
                    stage,
                    cost_before=cost_before.detach(),
                    cost_after=cost_after.detach(),
                    duration_s=None,
                    extras={'chunk_size': cs, 'accepts': c2o_stats['accepts'],
                            'iters': c2o_stats['iters']},
                )
    return seeds


def reconnect(
        get_cost_func,
        batch,
        opts,
        revisers,
        bench_logger=None,
    ):
    seed = batch
    problem_size = seed.size(1)
    if len(revisers) == 0:
        cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)

    for revision_id in range(len(revisers)):
        assert opts.revision_lens[revision_id] <= seed.size(1)
        start_time = time.time()
        shift_len = max(opts.revision_lens[revision_id]//opts.revision_iters[revision_id], 1)
        seed = LCP_TSP(
            seed,
            get_cost_func,
            revisers[revision_id],
            opts.revision_lens[revision_id],
            opts.revision_iters[revision_id],
            opts=opts,
            shift_len=shift_len,
            bench_logger=bench_logger,
            revision_id=revision_id,
            )
        cost_revised = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
        duration = time.time() - start_time

        if bench_logger is not None:
            bench_logger.add_stage(
                f'cascade_{revision_id}',
                cost_before=None,
                cost_after=cost_revised.detach(),
                duration_s=duration,
                extras={'revision_len': opts.revision_lens[revision_id],
                       'revision_iters': opts.revision_iters[revision_id],
                       'shift_len': shift_len},
            )

        # End-of-pipeline chunk_2opt. Only runs when --chunk_2opt_after_each_iter
        # is OFF (otherwise the per-pass calls in LCP_TSP already covered this).
        # By default uses the last cascade level's chunk size (= SHPP chunk size
        # for the last level); override with --chunk_2opt_chunk_size.
        if (getattr(opts, 'chunk_2opt', False)
                and not getattr(opts, 'chunk_2opt_after_each_iter', True)
                and revision_id == len(revisers) - 1):
            cs = getattr(opts, 'chunk_2opt_chunk_size', 0) or opts.revision_lens[revision_id]
            if getattr(opts, 'no_chunk_2opt_exhaustive_shifts', False):
                exhaustive_shifts = False
            else:
                exhaustive_shifts = (cs < opts.revision_lens[revision_id])
            cost_before_eop = cost_revised.detach()
            seed, cost_revised, c2o_stats = chunk_2opt(
                seed,
                chunk_size=cs,
                n_iters=getattr(opts, 'chunk_2opt_iters', 10),
                include_flip=getattr(opts, 'chunk_2opt_flip', False),
                include_close=True,
                exhaustive_shifts=exhaustive_shifts,
            )
            if bench_logger is not None:
                bench_logger.add_stage(
                    f'chunk_2opt_end',
                    cost_before=cost_before_eop,
                    cost_after=cost_revised.detach(),
                    duration_s=None,
                    extras={'chunk_size': cs,
                            'accepts': c2o_stats['accepts'],
                            'iters': c2o_stats['iters']},
                )

        if revision_id == 0 and not opts.no_prune: # eliminate the underperforming ones after the first round of revisions
            cost_revised, cost_revised_minidx = cost_revised.reshape(-1, opts.eval_batch_size).min(0) # width, bs
            seed = seed.reshape(-1, opts.eval_batch_size, seed.shape[-2], 2)[cost_revised_minidx, torch.arange(opts.eval_batch_size)]
    if opts.no_prune:
            cost_revised, cost_revised_minidx = cost_revised.reshape(-1, opts.eval_batch_size).min(0)
            seed = seed.reshape(-1, opts.eval_batch_size, seed.shape[-2], 2)[cost_revised_minidx, torch.arange(opts.eval_batch_size)]
    assert cost_revised.shape == (opts.eval_batch_size,)
    assert seed.shape == (opts.eval_batch_size, problem_size, 2)

    return seed, cost_revised


def chunk_2opt(seed, chunk_size, n_iters=10, include_flip=False, include_close=True,
             exhaustive_shifts=False):
    """Chunk-level 2-opt: permute (and optionally flip) chunks to reduce the
    cost of inter-chunk boundary edges. Internal chunk contents are preserved,
    so this is a search over the `k = ceil(N / chunk_size)` chunk orderings
    (and orientations) of the tour. When `N % chunk_size != 0`, the final
    "partial" chunk is a regular 2-opt unit (just with fewer nodes), not a
    no-op.

    The move is a 2-opt segment reversal: pick two non-adjacent chunk boundaries
    `(a, b)` and `(c, d)` in the current order and reverse the segment
    `[b, ..., c]`. With `include_flip=True`, we additionally consider the
    variant where chunks `b` and `c` are flipped in orientation.

    Always best-improvement-and-only-if-positive: never produces a tour worse
    than the input (within 1e-9 of strict improvement), and exits early when
    no further improvement is found.

    Args:
        seed: (B, N, 2) float tensor — coordinates in tour order (closed loop).
              Matches the layout used by `reconnect` (utils/functions.py:296)
              and main.py:124.
        chunk_size: int. Need not divide N — the trailing `N % chunk_size`
                    nodes form a partial chunk that is also a 2-opt unit.
                    No-op (with a one-time warn) only when chunk_size <= 1
                    or chunk_size >= N.
        n_iters: max outer 2-opt sweeps. Stops early if no improvement.
        include_flip: if True, also try flipping chunks b and c in each move.
        include_close: if True, treat tour as closed loop and include the
                       wraparound edge (chunk k-1 → chunk 0) in the cost;
                       matches the convention used by `reconnect` and main.py:124.
        exhaustive_shifts: if True, try all `chunk_size` rotations of the
                           chunk boundaries and keep the best per-instance
                           result. This guarantees full coverage of all
                           possible chunk alignments (important when the
                           chunk size is smaller than the SHPP decomposition's
                           chunk size — in that case, the SHPP shift_len
                           rotation only visits `revision_iter` of the
                           `chunk_size` possible alignments). Cost: roughly
                           `chunk_size`× more compute per call. When False
                           (default), the search is run on the input as-is
                           (single alignment) and the caller is expected to
                           drive multiple rotations via the LCP_TSP pass loop.

    Returns:
        new_seed: (B, N, 2) — improved tour; never worse per-instance.
        new_cost: (B,) — closed-loop tour length of new_seed, recomputed
                  using the canonical formula (utils/functions.py:305) to
                  avoid stale `cost_internal`.
        stats: dict with:
            - 'initial_cost': (B,) cost before search
            - 'final_cost':   (B,) cost after search
            - 'iters':        int, number of sweeps actually run (summed
                               across all shifts when exhaustive_shifts=True)
            - 'accepts':      int, total moves accepted across sweeps (summed
                               across all shifts when exhaustive_shifts=True)
            - 'improved_mask': bool (B,) — instances where at least one move
                               was accepted (across any shift)
            - 'exhaustive_shifts': int — number of shifts tried (only present
                                        when the flag was on)
    """
    B, N, _ = seed.shape

    # Canonical closed-loop tour cost (matches utils/functions.py:305, :320).
    def _closed_cost(x):
        return (x[:, 1:] - x[:, :-1]).norm(p=2, dim=2).sum(1) \
             + (x[:, 0] - x[:, -1]).norm(p=2, dim=1)

    # Graceful no-op only when chunk_size is so small/large that no 2-opt is
    # possible. (N % chunk_size != 0 is fine — see below.)
    if chunk_size <= 1 or chunk_size >= N:
        warnings.warn(
            f"chunk_2opt: chunk_size={chunk_size} invalid for N={N}; "
            f"returning input unchanged."
        )
        cost = _closed_cost(seed).detach().clone()
        return seed, cost, {
            'initial_cost': cost.clone(),
            'final_cost': cost.clone(),
            'iters': 0,
            'accepts': 0,
            'improved_mask': torch.zeros(B, dtype=torch.bool, device=seed.device),
        }

    # Chunk decomposition: `k_complete` full chunks of size `s`, plus one
    # partial chunk of size `offset` if N is not a multiple of `s`. The
    # partial chunk is a regular 2-opt unit, just with fewer nodes.
    k_complete = N // chunk_size
    offset = N - k_complete * chunk_size  # = N % chunk_size
    k = k_complete + (1 if offset > 0 else 0)
    s = chunk_size
    device = seed.device

    if k <= 1:
        # Not enough units to form a 2-opt move.
        cost = _closed_cost(seed).detach().clone()
        return seed, cost, {
            'initial_cost': cost.clone(),
            'final_cost': cost.clone(),
            'iters': 0,
            'accepts': 0,
            'improved_mask': torch.zeros(B, dtype=torch.bool, device=device),
        }

    # Exhaustive-shifts mode: try all `chunk_size` rotations of the chunk
    # boundaries and keep the best per-instance result. This guarantees full
    # coverage of all possible chunk alignments (important when the chunk
    # size is smaller than the SHPP decomposition's chunk size, since the
    # SHPP shift_len rotation only visits `revision_iter` of the
    # `chunk_size` possible alignments). Cost: roughly `chunk_size`× more
    # compute per call.
    if exhaustive_shifts and chunk_size > 1:
        initial_cost = _closed_cost(seed).detach().clone()
        best_seed = seed.clone()
        best_cost = _closed_cost(seed)
        total_iters = 0
        total_accepts = 0
        improved_mask = torch.zeros(B, dtype=torch.bool, device=device)
        for shift in range(chunk_size):
            if shift > 0:
                shifted_seed = torch.roll(seed, shifts=shift, dims=1)
            else:
                shifted_seed = seed
            s, c, st = chunk_2opt(
                shifted_seed, chunk_size, n_iters=n_iters,
                include_flip=include_flip, include_close=include_close,
                exhaustive_shifts=False,  # recurse without the wrapper
            )
            if shift > 0:
                s = torch.roll(s, shifts=-shift, dims=1)
            improved = c < best_cost - 1e-9
            if improved.any():
                best_seed[improved] = s[improved]
                best_cost[improved] = c[improved]
                improved_mask |= improved
            total_iters += st.get('iters', 0)
            total_accepts += st.get('accepts', 0)
        stats = {
            'initial_cost': initial_cost,
            'final_cost': best_cost,
            'iters': total_iters,
            'accepts': total_accepts,
            'improved_mask': improved_mask,
            'exhaustive_shifts': chunk_size,
        }
        return best_seed, best_cost, stats

    if offset == 0:
        # Fast path: N is an exact multiple of chunk_size. All units have
        # the same size; we can use a simple reshape + gather + reshape.
        chunks = seed.reshape(B, k, s, 2).contiguous()                  # (B, k, s, 2)
        actual_sizes = None  # sentinel: all chunks have size s
    else:
        # Pad the partial chunk with its last actual node so the phantom
        # "edges" in the padded positions have length 0. The 2-opt algorithm
        # only uses start/end and intra-chunk edge sums, so the padding
        # doesn't change any cost (and the partial chunk can be permuted
        # alongside the complete ones in the same vectorized loop).
        chunks = torch.zeros(B, k, s, 2, device=device, dtype=seed.dtype)
        actual_sizes = torch.full((k,), s, device=device, dtype=torch.long)
        if k_complete > 0:
            chunks[:, :k_complete, :, :] = (
                seed[:, :k_complete * s, :].reshape(B, k_complete, s, 2)
            )
        # Partial chunk: first `offset` positions are real; the rest are
        # padded with the last actual node (so phantom edges have length 0).
        chunks[:, k_complete, :offset, :] = seed[:, k_complete * s:, :]
        if offset < s:
            chunks[:, k_complete, offset:, :] = seed[:, -1:, :]
        actual_sizes[k_complete] = offset

    # Forward start/end. For the partial chunk, end is at index (size - 1).
    start_fwd = chunks[:, :, 0, :].contiguous()                              # (B, k, 2)
    if actual_sizes is None:
        end_fwd = chunks[:, :, -1, :].contiguous()                            # (B, k, 2)
    else:
        # Gather along the s dimension (dim=2) to pick the last *actual* node
        # of each chunk. end_indices has shape (1, k, 1) — one index per chunk
        # along dim 2 — broadcast to (B, k, 1, 2) for the gather.
        end_idx = (actual_sizes - 1).view(1, k, 1, 1).expand(B, k, 1, 2)
        end_fwd = torch.gather(chunks, 2, end_idx).squeeze(2).contiguous()    # (B, k, 2)

    # State: a per-batch permutation of chunks + a per-batch orientation mask.
    order = torch.arange(k, device=device).expand(B, -1).contiguous()  # (B, k)
    flip = torch.zeros(B, k, dtype=torch.bool, device=device)           # (B, k)

    initial_cost = _closed_cost(seed).detach().clone()

    # Enumerate all 2-opt moves as (a, b, c, d) tuples where (a, b) and (c, d)
    # are the two chunk boundaries being rewired. Edges before: D[a, b] + D[c, d].
    # Edges after:  D[a, c] + D[b, d]. Segment reversed: [b, ..., c].
    # For closed tours, the wraparound case has c = k-1, d = 0.
    #
    # NB: unlike standard city-level 2-opt, the cost change is NOT just the
    # two boundary edges. When a segment of chunks is reversed, every internal
    # edge of the segment changes direction — and unlike single-city edges,
    # the boundary cost D(x, y) = ||end[x] - start[y]|| is generally NOT
    # symmetric: D(x, y) != D(y, x). We account for the internal edge cost
    # changes via the prefix-sum term `prefix_E[c] - prefix_E[b]` below.
    pair_a, pair_b, pair_c, pair_d = [], [], [], []
    for i in range(k):
        for j in range(i + 2, k):
            pair_a.append(i)
            pair_b.append(i + 1)
            pair_c.append(j - 1)
            pair_d.append(j)
    if include_close and k >= 3:
        for i in range(k - 2):
            pair_a.append(i)
            pair_b.append(i + 1)
            pair_c.append(k - 1)
            pair_d.append(0)

    P = len(pair_a)
    if P == 0:
        # k is too small for any 2-opt move (e.g., k == 2).
        stats = {
            'initial_cost': initial_cost,
            'final_cost': initial_cost.clone(),
            'iters': 0,
            'accepts': 0,
            'improved_mask': torch.zeros(B, dtype=torch.bool, device=device),
        }
        return seed, initial_cost, stats

    pair_a = torch.tensor(pair_a, device=device, dtype=torch.long)
    pair_b = torch.tensor(pair_b, device=device, dtype=torch.long)
    pair_c = torch.tensor(pair_c, device=device, dtype=torch.long)
    pair_d = torch.tensor(pair_d, device=device, dtype=torch.long)

    accepts = 0
    iters = 0
    improved_mask = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(n_iters):
        iters += 1

        # Effective start/end of the chunk at each position, given the current
        # `order` and `flip`. If `flip[t]` is True, start and end swap.
        chunk_idx = order                                                       # (B, k)
        idx_exp = chunk_idx.unsqueeze(-1).expand(-1, -1, 2)                    # (B, k, 2)
        s_gather = start_fwd.gather(1, idx_exp)                                 # (B, k, 2)
        e_gather = end_fwd.gather(1, idx_exp)                                   # (B, k, 2)
        s_eff = torch.where(flip.unsqueeze(-1), e_gather, s_gather)
        e_eff = torch.where(flip.unsqueeze(-1), s_gather, e_gather)

        # Boundary cost matrix D[a, b] = ||e_eff[a] - s_eff[b]||.
        D = (e_eff.unsqueeze(2) - s_eff.unsqueeze(1)).norm(dim=-1)              # (B, k, k)

        # No-flip delta for every (a, b, c, d). The full delta accounts for
        # the internal edges of the reversed segment too:
        #   delta = (D[a, c] + D[b, d] - D[a, b] - D[c, d])         (boundary improvement)
        #         - sum_{i=b}^{c-1} (D[i+1, i] - D[i, i+1])         (internal worsening)
        # where the second term is subtracted because E[i] = D[i+1, i] - D[i, i+1]
        # is the COST INCREASE from reversing internal edge (i, i+1). A positive
        # delta = (old cost) - (new cost) means the move is an improvement.
        D_ab = D[:, pair_a, pair_b]                                             # (B, P)
        D_cd = D[:, pair_c, pair_d]
        D_ac = D[:, pair_a, pair_c]
        D_bd = D[:, pair_b, pair_d]
        boundary_delta = D_ab + D_cd - D_ac - D_bd                               # (B, P)

        # E[i] = cost change of reversing the (i, i+1) edge: D[i+1, i] - D[i, i+1].
        # E has shape (B, k-1); we then prefix-sum to (B, k) with prefix_E[0] = 0.
        idx_i = torch.arange(k - 1, device=device)
        idx_ip1 = idx_i + 1
        D_ii1 = D[:, idx_i, idx_ip1]                                            # (B, k-1)
        D_i1i = D[:, idx_ip1, idx_i]                                            # (B, k-1)
        E = D_i1i - D_ii1                                                       # (B, k-1)
        prefix_E = torch.zeros(B, k, device=device, dtype=E.dtype)
        prefix_E[:, 1:] = E.cumsum(dim=-1)                                       # (B, k)
        # Internal sum for segment [b, c] is prefix_E[c] - prefix_E[b].
        # Subtract from boundary_delta: positive delta = improvement.
        internal_sum = prefix_E[:, pair_c] - prefix_E[:, pair_b]                # (B, P)
        delta_noflip = boundary_delta - internal_sum                             # (B, P)

        # NOTE: the all-flip variant (flipping chunks b and c as part of the
        # move) would need a similar accounting for the changed internal edges
        # of the flipped boundary chunks. For now we keep the implementation
        # minimal: ignore the all-flip path and just use the no-flip delta.
        # The CLI flag `--chunk_2opt_flip` is preserved for forward
        # compatibility but currently has no effect.
        delta_best = delta_noflip
        take_flip = torch.zeros_like(delta_noflip, dtype=torch.bool)

        # Best move per batch instance.
        best_p = delta_best.argmax(dim=1)                                       # (B,)
        best_delta = delta_best.gather(1, best_p.unsqueeze(1)).squeeze(1)       # (B,)

        if best_delta.max() <= 1e-9:
            break

        mask = best_delta > 1e-9
        if not mask.any():
            break

        idx_a = pair_a[best_p]
        idx_b = pair_b[best_p]
        idx_c = pair_c[best_p]
        idx_d = pair_d[best_p]
        use_flip = take_flip.gather(1, best_p.unsqueeze(1)).squeeze(1)

        # Apply the move per batch instance. The chunk identities in the
        # reversed segment are permuted, so we reverse `order` AND `flip`
        # together; then optionally toggle `flip` at the new boundary
        # positions for the all-flip variant.
        b_idx = mask.nonzero(as_tuple=True)[0]
        for b in b_idx.tolist():
            i_b = int(idx_b[b].item())
            i_c = int(idx_c[b].item())
            order[b, i_b:i_c + 1] = order[b, i_b:i_c + 1].flip(0)
            flip[b, i_b:i_c + 1] = flip[b, i_b:i_c + 1].flip(0)
            if bool(use_flip[b].item()):
                flip[b, i_b] = ~flip[b, i_b]
                flip[b, i_c] = ~flip[b, i_c]

        accepts += int(mask.sum().item())
        improved_mask |= mask

    # Reconstruct the tour from `order` and `flip`. Each chunk keeps its
    # original node content; only the position and orientation change.
    chunk_idx = order
    chunk_idx_exp = chunk_idx.view(B, k, 1, 1).expand(-1, -1, s, 2)
    new_chunks = torch.gather(chunks, 1, chunk_idx_exp)                         # (B, k, s, 2)

    if flip.any():
        new_chunks_flipped = new_chunks.flip(dims=[2])
        flip_mask = flip.view(B, k, 1, 1)
        new_chunks = torch.where(flip_mask, new_chunks_flipped, new_chunks)

    if actual_sizes is None:
        # Fast reconstruction: all units have the same size, so the gathered
        # tensor can be reshaped directly.
        new_seed = new_chunks.reshape(B, N, 2).contiguous()
    else:
        # Variable-size reconstruction: for each (batch, position), take only
        # the first actual_sizes[order[b, t]] nodes of the gathered chunk.
        # We use a per-batch loop here — B is typically small (~128) and the
        # loop is O(B * k * s) which is the same order as the main 2-opt loop.
        new_seed = torch.empty(B, N, 2, device=device, dtype=seed.dtype)
        for b in range(B):
            pos = 0
            for t in range(k):
                size_t = int(actual_sizes[int(chunk_idx[b, t].item())].item())
                chunk_data = new_chunks[b, t, :size_t, :]
                if bool(flip[b, t].item()):
                    chunk_data = chunk_data.flip(0)
                new_seed[b, pos:pos + size_t, :] = chunk_data
                pos += size_t
            assert pos == N, f"batch {b}: pos {pos} != N {N}"
        new_seed = new_seed.contiguous()

    new_cost = _closed_cost(new_seed).detach().clone()

    stats = {
        'initial_cost': initial_cost,
        'final_cost': new_cost,
        'iters': iters,
        'accepts': accepts,
        'improved_mask': improved_mask,
    }
    return new_seed, new_cost, stats


def sample_many():
    raise NotImplementedError




