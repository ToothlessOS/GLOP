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


def eval_dataset(dataset_path, opts):
    pp.pprint(vars(opts))

    revisers = []
    revision_lens = opts.revision_lens

    for reviser_size in revision_lens:
        reviser_path = f'pretrained/Reviser-stage2/reviser_{reviser_size}/epoch-299.pt'
        reviser, _ = load_model(reviser_path, is_local=True)
        revisers.append(reviser)

    for reviser in revisers:
        reviser.to(opts.device)
        reviser.eval()
        reviser.set_decode_type(opts.decode_strategy)

    results, duration, all_stats = _eval_dataset(dataset_path, opts, opts.device, revisers)

    costs, costs_revised, costs_revised_with_penalty, costs_warm_avg, tours = zip(*results)
    costs = torch.tensor(costs)               # best-of-width warm start per instance (existing)
    costs_warm_avg = torch.tensor(costs_warm_avg)  # avg-of-width warm start per instance (new)
    if opts.problem_type in ['cvrp', 'cvrplib']:
        costs_revised = torch.stack(costs_revised)
    else:
        costs_revised = torch.cat(costs_revised, dim=0)

    if opts.problem_type == 'pctsp':
        costs_revised_with_penalty = torch.cat(costs_revised_with_penalty, dim=0)

    def _ci(t):
        return (2 * torch.std(t) / math.sqrt(len(t))).item()

    print("=== Warm start (random insertion) ===")
    print("  avg  = {:.4f} +- {:.4f}".format(costs_warm_avg.mean().item(), _ci(costs_warm_avg)))
    print("  best = {:.4f} +- {:.4f}".format(costs.mean().item(), _ci(costs)))
    print("=== Final (after LCP revision) ===")
    print("  avg  = {:.4f} +- {:.4f}".format(costs_revised.mean().item(), _ci(costs_revised)))
    print("  best = {:.4f}".format(costs_revised.min().item()))
    if opts.problem_type == 'pctsp':
        print("=== Final with penalty ===")
        print("  avg  = {:.4f} +- {:.4f}".format(costs_revised_with_penalty.mean().item(),
                                                _ci(costs_revised_with_penalty)))
        print("  best = {:.4f}".format(costs_revised_with_penalty.min().item()))
    print("=== Total duration: {:.2f}s ===".format(duration))

    # Aggregate per-iteration stats across batches (weighted by count) and plot.
    if all_stats:
        plot_solver_curve(all_stats, opts)

    if opts.problem_type != 'cvrp':
        tours = torch.cat(tours, dim=0)
    return tours

def _eval_dataset(dataset_path, opts, device, revisers):
    start = time.time()
    if opts.problem_type == 'tsp':
        dataset = revisers[0].problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=0)
        if opts.problem_size <= 100:
            if opts.width >= 4:
                opts.width //= 4
                opts.tsp_aug = True
            else:
                opts.tsp_aug = False
        orders = [torch.randperm(opts.problem_size) for i in range(opts.width)]
        pi_all = [random_insertion_parallel(dataset, order) for order in orders] # instance: (p_size, 2)
        pi_all = torch.tensor(np.array(pi_all).astype(np.int64)).reshape(len(orders), opts.val_size, opts.problem_size) # width, val_size, p_size
    elif opts.problem_type == 'pctsp': # dataset (n_cons*val_size, p_size, 2), pi_all (width, n_cons*val_size, p_size), penalty (n_cons, val_size)
        from problems.pctsp import init
        opts.eval_batch_size = opts.eval_batch_size * opts.n_subset
        dataset, penalty= init(dataset_path, opts) # (n_val*n_subset, max_seq_len, 2), (val_size*n_subset, )
        dataset = dataset.cpu()
        max_seq_len = dataset.size(1)
        order = torch.arange(max_seq_len) # width=1 by default for pctsp
        pi_all = random_insertion_parallel(dataset, order) # (n_val*n_subset, max_seq_len)
        pi_all = torch.tensor(pi_all.astype(np.int64)).unsqueeze(0)  # (1, n_val*n_subset, max_seq_len)
        assert pi_all.shape == (1, opts.val_size*opts.n_subset, max_seq_len)
    elif opts.problem_type == 'cvrp':
        from problems.cvrp import init  
        dataset, n_tsps_per_route_lst = init(dataset_path, opts)
        opts.eval_batch_size = 1
    elif opts.problem_type == 'cvrplib':
        from problems.cvrp import init  
        ckpt_path = "./pretrained/Partitioner/cvrp/cvrp-2000-cvrplib.pt" if opts.ckpt_path == '' else opts.ckpt_path   
        partitioner = load_partitioner(2000, opts.device, ckpt_path, 300, 6)
        dataset, n_tsps_per_route_lst = init(dataset_path, opts, partitioner)
        opts.eval_batch_size = 1
        
    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)
    

    problem = load_problem('tsp')
    get_cost_func = lambda input, pi: problem.get_costs(input, pi, return_local=True)
    
    results = []
    all_stats = []  # accumulated per-iter (best, avg, count) across batches
    for batch_id, batch in tqdm(enumerate(dataloader), disable=opts.no_progress_bar):
        # tsp batch shape: (bs, problem size, 2)
        avg_cost = 0
        with torch.no_grad():
            if opts.problem_type in ['tsp', 'pctsp']:
                p_size = batch.size(1)
                batch = batch.repeat(opts.width, 1, 1) # (1,1,1) for pctsp
                pi_batch = pi_all[:, batch_id*opts.eval_batch_size: (batch_id+1)*opts.eval_batch_size, :].reshape(-1, p_size)
                seed = batch.gather(1, pi_batch.unsqueeze(-1).repeat(1,1,2))
            elif opts.problem_type in ['cvrp', 'cvrplib']:
                batch = batch.squeeze() # (n_subTSPs_for_width_routes, max_seq_len, 2)
                n_subTSPs, max_seq_len, _ = batch.shape
                n_tsps_per_route = n_tsps_per_route_lst[batch_id]
                assert sum(n_tsps_per_route) == n_subTSPs
                opts.eval_batch_size = n_subTSPs
                order = torch.arange(max_seq_len)
                pi_batch = random_insertion_parallel(batch, order)
                pi_batch = torch.tensor(pi_batch.astype(np.int64))
                assert pi_batch.shape == (n_subTSPs, max_seq_len)
                seed = batch.gather(1, pi_batch.unsqueeze(-1).repeat(1,1,2))
                assert seed.shape == (n_subTSPs, max_seq_len, 2)
            else:
                raise NotImplementedError
                
            seed = seed.to(device)
            cost_ori = (seed[:, 1:] - seed[:, :-1]).norm(p=2, dim=2).sum(1) + (seed[:, 0] - seed[:, -1]).norm(p=2, dim=1)
            if opts.problem_type in ['tsp', 'pctsp']:
                cost_ori_grouped = cost_ori.reshape(-1, opts.eval_batch_size)  # (width, eval_batch_size)
                cost_ori_best, _ = cost_ori_grouped.min(0)                       # (eval_batch_size,)
                cost_ori_avg = cost_ori_grouped.mean(0)                          # (eval_batch_size,)
                avg_cost = cost_ori_best.mean().item()
                avg_cost_warm = cost_ori_avg.mean().item()
            elif opts.problem_type in ['cvrp', 'cvrplib']:
                avg_cost = sum_cost(cost_ori, n_tsps_per_route).min()
                avg_cost_warm = float(cost_ori.mean().item())  # width=1 forced for CVRP (main.py enforces)
            else:
                raise NotImplementedError

            if opts.problem_size <= 100 and opts.problem_type=='tsp' and opts.tsp_aug:
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
                                        )

        if opts.problem_type == 'pctsp':
            costs_revised_with_penalty, costs_revised_minidx = (costs_revised.reshape(-1, opts.n_subset)+ \
                penalty[batch_id*opts.eval_batch_size: (batch_id+1)*opts.eval_batch_size].reshape(-1, opts.n_subset)).min(1)
            costs_revised, _ = costs_revised.reshape(-1, opts.n_subset).min(1)
            tours = tours.reshape(-1, opts.n_subset, max_seq_len, 2)[torch.arange(opts.eval_batch_size//opts.n_subset), costs_revised_minidx, :, :]
            assert costs_revised.size(0) == costs_revised_with_penalty.size(0) == tours.size(0) == opts.eval_batch_size//opts.n_subset
        elif opts.problem_type in ['cvrp', 'cvrplib']:
            assert costs_revised.shape == (n_subTSPs,)
            costs_revised, best_partition_idx = sum_cost(costs_revised, n_tsps_per_route).min(dim=0)
            subtour_start = sum(n_tsps_per_route[:best_partition_idx])
            tours = tours[subtour_start: subtour_start+n_tsps_per_route[best_partition_idx]]
            assert tours.shape == (n_tsps_per_route[best_partition_idx], max_seq_len, 2)
            tours = tours.reshape(-1, 2)
        
        if opts.problem_type == 'pctsp':
            results.append((avg_cost, costs_revised, costs_revised_with_penalty, avg_cost_warm, tours))
        elif opts.problem_type in ['tsp', 'cvrp', 'cvrplib']:
            results.append((avg_cost, costs_revised, None, avg_cost_warm, tours))
        else:
            raise NotImplementedError
        

    duration = time.time() - start

    return results, duration, all_stats

def _aggregate_solver_curve(all_stats):
    """Aggregate per-batch per-iter stats into per-(layer, iter) means.

    Each input entry has keys {layer_id, iter_id, sum_best, sum_avg, count}.
    We sum across batches and divide by the total count for each (layer, iter).
    Returns a dict: {layer_id: {'iters': [...], 'best': [...], 'avg': [...]}}.
    """
    agg = {}
    for s in all_stats:
        key = (s['layer_id'], s['iter_id'])
        if key not in agg:
            agg[key] = {'sum_best': 0.0, 'sum_avg': 0.0, 'count': 0}
        agg[key]['sum_best'] += s['sum_best']
        agg[key]['sum_avg'] += s['sum_avg']
        agg[key]['count'] += s['count']
    layers = sorted({k[0] for k in agg.keys()})
    out = {}
    for lid in layers:
        iters = sorted(k[1] for k in agg.keys() if k[0] == lid)
        best, avg = [], []
        for it in iters:
            entry = agg[(lid, it)]
            best.append(entry['sum_best'] / max(entry['count'], 1))
            avg.append(entry['sum_avg'] / max(entry['count'], 1))
        out[lid] = {'iters': iters, 'best': best, 'avg': avg}
    return out


def plot_solver_curve(all_stats, opts):
    """Plot avg/best cost per iteration for each revisor layer."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    agg = _aggregate_solver_curve(all_stats)
    if not agg:
        return

    layers = sorted(agg.keys())
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.5), sharey=True)
    if n_layers == 1:
        axes = [axes]
    colors = plt.cm.viridis([i / max(n_layers - 1, 1) for i in range(n_layers)])

    for ax, lid, color in zip(axes, layers, colors):
        data = agg[lid]
        x = [it + 1 for it in data['iters']]
        # Always plot both curves, even when they overlap (e.g. post-prune width=1).
        ax.plot(x, data['avg'], '--', color=color, linewidth=2.0,
                marker='o', markersize=5, label='avg across --width')
        ax.plot(x, data['best'], '-', color=color, linewidth=2.0,
                marker='s', markersize=5, label='best across --width')
        ax.set_title(f'Layer {lid + 1}: L={opts.revision_lens[lid]} '
                     f'({opts.revision_iters[lid]} iters)')
        ax.set_xlabel('Iteration within layer')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        # Annotate first and last values
        if data['best']:
            ax.annotate(f"{data['best'][0]:.3f}", xy=(x[0], data['best'][0]),
                        xytext=(3, 5), textcoords='offset points', fontsize=8,
                        color=color)
            ax.annotate(f"{data['best'][-1]:.3f}", xy=(x[-1], data['best'][-1]),
                        xytext=(-25, 5), textcoords='offset points', fontsize=8,
                        color=color)

    axes[0].set_ylabel('Closed-loop tour cost\n(eval-dataset mean)')
    fig.suptitle(f'GLOP sub-TSP solver convergence — '
                 f'{opts.problem_type}{opts.problem_size}, width={opts.width}, val_size={opts.val_size}',
                 fontsize=11)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    out_dir = 'results'
    os.makedirs(out_dir, exist_ok=True)
    tag = (f"{opts.problem_type}{opts.problem_size}_w{opts.width}"
           f"_lens{'-'.join(str(x) for x in opts.revision_lens)}"
           f"_iters{'-'.join(str(x) for x in opts.revision_iters)}")
    out_path = os.path.join(out_dir, f'solver_curve_{tag}.png')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"=== Solver curve saved to: {out_path} ===")


if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_size", type=int, default=200)
    parser.add_argument("--problem_type", type=str, default='tsp')
    parser.add_argument('--val_size', type=int, default=128,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--eval_batch_size', type=int, default=128,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--revision_lens', nargs='+', default=[20] ,type=int,
                        help='The sizes of revisers')
    parser.add_argument('--revision_iters', nargs='+', default=[10,], type=int,
                        help='Revision iterations (I_n)')
    parser.add_argument('--decode_strategy', type=str, default='sampling', help='decode strategy of the model')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--width', type=int, default=1, 
                        help='The initial solutions for a TSP instance generated with diversified insertion')
    parser.add_argument('--no_aug', action='store_true', help='Disable instance augmentation')
    parser.add_argument('--path', type=str, default='', 
                        help='The test dataset path for cross-distribution evaluation')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--n_subset', type=int, default=1, help='The number of stochastically constructed PCTSP node subsets')
    parser.add_argument('--n_partition', type=int, default=1, help='The number of stochastically constructed CVRP partitions')
    parser.add_argument('--ckpt_path', type=str, default='', help='Checkpoint path for CVRP eval')
    parser.add_argument('--no_prune', action='store_true', help='Do not prune the unpromising tours after the first round of revisions')
    parser.add_argument('--use_2opt', action='store_true',
                        help='Enable 2-opt post-processing on the GLOP tour')
    parser.add_argument('--two_opt_mode', type=str, default='final', choices=['final', 'per_iter'],
                        help="2-opt evaluation mode: 'final' runs it once after the whole pipeline; "
                             "'per_iter' runs it after each revisor iteration")
    parser.add_argument('--two_opt_iters', type=int, default=10,
                        help='Max number of 2-opt sweeps per invocation')
    parser.add_argument('--two_opt_kind', type=str, default='full',
                        choices=['full', 'knn', 'radius', 'range_radius'],
                        help="2-opt algorithm variant: 'full' (dense candidate set), "
                             "'knn' (k-NN-sparse; uses --two_opt_knn_k), "
                             "'radius' (tour-position-sparse; uses --two_opt_radius), "
                             "or 'range_radius' (tour-position-sparse over "
                             "[r_min, r_max]; uses --two_opt_radius_min / "
                             "--two_opt_radius_max)")
    parser.add_argument('--two_opt_knn_k', type=int, default=20,
                        help='k for KNN-sparse 2-opt (only used when --two_opt_kind=knn)')
    parser.add_argument('--two_opt_radius', type=int, default=None,
                        help='r for radius-sparse 2-opt (only used when '
                             '--two_opt_kind=radius). Default: 10%% of '
                             '--problem_size (floored at 2).')
    parser.add_argument('--two_opt_radius_min', type=int, default=2,
                        help='r_min for range-radius 2-opt (only used when '
                             '--two_opt_kind=range_radius). Default: 2.')
    parser.add_argument('--two_opt_radius_max', type=int, default=None,
                        help='r_max for range-radius 2-opt (only used when '
                             '--two_opt_kind=range_radius). Default: 10%% of '
                             '--problem_size (floored at max(r_min, 2)).')
    parser.add_argument('--two_opt_debug', action='store_true',
                        help='Print per-sweep 2-opt phase timings to stdout '
                             '(knn_graph construction, candidate extraction, '
                             'apply loop, gather, etc.) for perf investigation.')
    opts = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    device_id = opts.device_id
    device = torch.device(f"cuda:{device_id}" if use_cuda else "cpu")
    opts.device = device
    print('using device:', device)

    if opts.path == '':
        if opts.problem_type == 'tsp':
            opts.path = f'data/tsp/tsp{opts.problem_size}_test.pkl'
        elif opts.problem_type == 'cvrp':
            opts.path = f'data/vrp/vrp{opts.problem_size}_test_seed1234.pkl'
        elif opts.problem_type == 'pctsp':
            opts.path = f'data/pctsp/pctsp{opts.problem_size}_test_seed1234.pkl'
        else:
            raise NotImplementedError
        
    if opts.problem_type == 'cvrp':
        if opts.eval_batch_size != 1:
            opts.eval_batch_size = 1
            warnings.warn('Set eval_batch_size to 1 for CVRP!')
        if opts.width != 1:
            opts.width = 1
            warnings.warn('Set width to 1 for CVRP!')
        if opts.n_partition != 1:
            opts.n_partition = 1
            warnings.warn('Set n_partition to 1 for CVRP!')
    if opts.problem_type == 'pctsp':
        if opts.width != 1:
            opts.width = 1
            warnings.warn('Set width to 1 for PCTSP!')
        
    torch.manual_seed(opts.seed)
        
    tours = eval_dataset(opts.path, opts)