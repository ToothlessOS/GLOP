#!/usr/bin/env python

import os
import json
import pprint as pp
from time import time

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, RolloutBaseline, WarmupBaseline

try:
    from nets.attention_local import AttentionModel
    from utils import torch_load_cpu, load_problem
except:
    import sys

    sys.path.insert(0, "./")
    from nets.attention_local import AttentionModel
    from utils import torch_load_cpu, load_problem

"""
Modified run.py to support multiple datasets in training, given a ratio
"""


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(
            os.path.join(
                opts.log_dir,
                "{}_{}".format(opts.problem, opts.graph_size),
                opts.run_name,
            )
        )

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), "w") as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Figure out what's the problem
    problem = load_problem(opts.problem)

    # Load data from load_path
    load_data = {}
    assert (
        opts.load_path is None or opts.resume is None
    ), "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print("  [*] Loading data from {}".format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
    ).to(opts.device)

    # if opts.use_cuda and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get("model", {})})

    # Initialize baselines
    # When --RI_train2 is set we keep TWO independent baselines — one per dataset —
    # so each dataset's rollout EMA can be calibrated to its own cost distribution.
    # In the single-dataset case baseline2 aliases baseline1 to preserve the
    # original behavior exactly.

    def _make_baseline():
        if opts.baseline == "rollout":
            print("load rollout baseline ......")
            return RolloutBaseline(model, problem, opts)
        # argparse stores --baseline None as the string "None", not Python None,
        # so accept both spellings for NoBaseline.
        assert opts.baseline in (None, "None"), "Unknown baseline: {}".format(opts.baseline)
        return NoBaseline()

    baseline1 = _make_baseline()
    baseline2 = _make_baseline() if opts.RI_train2 else baseline1

    if opts.bl_warmup_epochs > 0:
        baseline1 = WarmupBaseline(
            baseline1, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta
        )
        if opts.RI_train2:
            baseline2 = WarmupBaseline(
                baseline2, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta
            )

    # Load baseline from data, make sure script is called with same type of baseline
    if "baseline" in load_data:
        baseline1.load_state_dict(load_data["baseline"])
        if opts.RI_train2:
            baseline2.load_state_dict(load_data["baseline"])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{"params": model.parameters(), "lr": opts.lr_model}]
        + (
            [{"params": baseline1.get_learnable_parameters(), "lr": opts.lr_critic}]
            if len(baseline1.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if "optimizer" in load_data:
        optimizer.load_state_dict(load_data["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: opts.lr_decay**epoch
    )
    lr_scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)
    lr_schedulers = [lr_scheduler, lr_scheduler2]

    # Start the actual training loop

    # make validation dataset
    graph_size = opts.graph_size
    filename = f"data/tsp/tsp_{opts.data_distribution}{graph_size}_val_seed1234.pkl"
    val_size = 10000
    val_dataset = problem.make_dataset(
        size=graph_size,
        num_samples=val_size,
        filename=filename,
        distribution=opts.data_distribution,
    )

    if opts.resume:
        epoch_resume = int(
            os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1]
        )

        torch.set_rng_state(load_data["rng_state"])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data["cuda_rng_state"])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline1.epoch_callback(model, epoch_resume)
        if opts.RI_train2:
            baseline2.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        start_time = time()

        # Load pre-generated training tensors (or fall back to per-epoch generation
        # when neither RI_train nor RI_train2 is set, in which case train_dataset
        # remains None and train_epoch generates data on the fly).
        if opts.RI_train:
            filename = opts.RI_path
            train_dataset = torch.load(filename, map_location=opts.device)
            print("----load training samples from {}----".format(filename))
            print(len(train_dataset))
        else:
            train_dataset = None

        train_dataset2 = None
        if opts.RI_train2:
            assert opts.RI_path2 is not None, (
                "--RI_train2 was set but no --RI_path2 was provided"
            )
            filename2 = opts.RI_path2
            train_dataset2 = torch.load(filename2, map_location=opts.device)
            print("----load stage-2 training samples from {}----".format(filename2))
            print(len(train_dataset2))

        if train_dataset is None and train_dataset2 is None:
            raise RuntimeError(
                "No training dataset enabled: pass --RI_train and/or --RI_train2, "
                "or omit both to fall back to per-epoch generation (not supported "
                "in the alternating curriculum loop)."
            )

        # Alternating curriculum: spend opts.n_epochs1 epochs on dataset 1, then
        # opts.n_epochs2 epochs on dataset 2, repeating until opts.n_epochs total
        # epochs have been consumed. Each dataset uses its own baseline so the
        # rollout EMA tracks each dataset's cost distribution independently.
        epoch = opts.epoch_start
        epoch_end = opts.epoch_start + opts.n_epochs
        total_epochs_remaining = lambda cur: epoch_end - cur  # noqa: E731

        while epoch < epoch_end:
            # Block 1: dataset 1
            if train_dataset is not None:
                block = min(opts.n_epochs1, total_epochs_remaining(epoch))
                if block > 0:
                    print(
                        ">> Block on dataset 1: {} epoch(s) starting at epoch {}".format(
                            block, epoch
                        )
                    )
                    for _ in range(block):
                        train_epoch(
                            model,
                            optimizer,
                            baseline1,
                            lr_schedulers,
                            epoch,
                            val_dataset,
                            problem,
                            tb_logger,
                            opts,
                            train_dataset,
                        )
                        epoch += 1

            # Block 2: dataset 2
            if train_dataset2 is not None and epoch < epoch_end:
                block = min(opts.n_epochs2, total_epochs_remaining(epoch))
                if block > 0:
                    print(
                        ">> Block on dataset 2: {} epoch(s) starting at epoch {}".format(
                            block, epoch
                        )
                    )
                    for _ in range(block):
                        train_epoch(
                            model,
                            optimizer,
                            baseline2,
                            lr_schedulers,
                            epoch,
                            val_dataset,
                            problem,
                            tb_logger,
                            opts,
                            train_dataset2,
                        )
                        epoch += 1

        end_time = time()
        print("total training duration:", end_time - start_time)


if __name__ == "__main__":
    run(get_options())
