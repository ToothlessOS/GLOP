"""Sanity tests for the 2-opt post-processing in utils/post_process.py.

Run with:  python -m pytest tests/test_post_process.py -q
       or:  python tests/test_post_process.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.post_process import full_2opt, maybe_two_opt


def tour_cost(coords):
    # coords: (B, N, D) in tour order -> (B,) closed-loop Euclidean cost
    return (coords[:, 1:] - coords[:, :-1]).norm(p=2, dim=2).sum(1) + \
           (coords[:, 0] - coords[:, -1]).norm(p=2, dim=1)


def test_shape_roundtrip():
    torch.manual_seed(0)
    x = torch.rand(4, 12, 2)
    out = full_2opt(x, iters=10)
    assert out.shape == x.shape


def test_does_not_mutate_input():
    torch.manual_seed(1)
    x = torch.rand(3, 15, 2)
    x_before = x.clone()
    _ = full_2opt(x, iters=10)
    assert torch.equal(x, x_before)


def test_never_worsens_random():
    torch.manual_seed(2)
    x = torch.rand(8, 20, 2)
    c0 = tour_cost(x)
    out = full_2opt(x, iters=25)
    c1 = tour_cost(out)
    # allow tiny fp slack
    assert (c1 <= c0 + 1e-5).all(), (c0, c1)


def test_multi_sweep_no_crash():
    # The old implementation crashed on the 2nd sweep. Ensure many sweeps run.
    torch.manual_seed(3)
    x = torch.rand(2, 30, 2)
    out = full_2opt(x, iters=50)
    assert out.shape == x.shape


def test_known_crossed_square():
    # Unit square visited in a self-crossing order: 2-opt must uncross it.
    # Corners: A(0,0) B(1,0) C(1,1) D(0,1). Bad order A,C,B,D crosses.
    coords = torch.tensor([[[0., 0.], [1., 1.], [1., 0.], [0., 1.]]])
    out = full_2opt(coords, iters=10)
    c1 = tour_cost(out).item()
    # Optimal Hamiltonian cycle on the unit square = perimeter = 4.0
    assert abs(c1 - 4.0) < 1e-5, c1


def test_maybe_two_opt_gating():
    torch.manual_seed(4)
    x = torch.rand(2, 10, 2)

    class Opts:
        pass

    off = Opts()  # no use_2opt attr -> disabled
    assert torch.equal(maybe_two_opt(x, off), x)

    on = Opts()
    on.use_2opt = True
    on.two_opt_iters = 10
    out = maybe_two_opt(x, on)
    assert (tour_cost(out) <= tour_cost(x) + 1e-5).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print("all tests passed")
