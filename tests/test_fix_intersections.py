"""Unit tests for utils.diagnosis.fix_intersections_via_2opt and
utils.functions.run_post_revision.

Plain asserts under `if __name__ == "__main__":` — no pytest required,
matching the project's de-facto style (cf. tests/test_diagnosis.py).

Run from the repo root:
    cd /home/toothlessos/Projects/nrp/GLOP && python tests/test_fix_intersections.py
"""

import math
import os
import sys

# Make the project root importable when this file is run directly.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from utils.diagnosis import fix_intersections_via_2opt
from utils.functions import run_post_revision


def _closed_loop_cost(tours):
    """Closed-loop tour cost (B,). Matches the formula used in main.py
    and utils/functions.py."""
    return (
        (tours[:, 1:] - tours[:, :-1]).norm(p=2, dim=2).sum(1)
        + (tours[:, 0] - tours[:, -1]).norm(p=2, dim=1)
    )


# ---------------------------------------------------------------------------
# fix_intersections_via_2opt
# ---------------------------------------------------------------------------


def _test_fix_intersections_figure_eight():
    # Figure-eight: (0,0) -> (1,1) crosses (1,0) -> (0,1).
    tours = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    cost_before = _closed_loop_cost(tours)
    fixed, info = fix_intersections_via_2opt(tours.clone())
    cost_after = _closed_loop_cost(fixed)
    assert info["final_intersections"] == 0
    assert cost_after.item() < cost_before.item()


def _test_fix_intersections_clean_square_noop():
    # CCW unit square: no intersections, no worsening.
    tours = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    cost_before = _closed_loop_cost(tours)
    fixed, info = fix_intersections_via_2opt(tours.clone())
    cost_after = _closed_loop_cost(fixed)
    assert info["final_intersections"] == 0
    assert torch.isclose(cost_after, cost_before, atol=1e-5).item()


def _test_fix_intersections_chord_5point():
    # 5-point fixture with a chord crossing at a non-vertex point.
    tours = torch.tensor(
        [
            [
                [0.0, 0.0],   # 0
                [2.0, 2.0],   # 1
                [1.0, 0.0],   # 2
                [0.0, 2.0],   # 3
                [2.0, 0.0],   # 4
            ]
        ],
        dtype=torch.float32,
    )
    cost_before = _closed_loop_cost(tours)
    fixed, info = fix_intersections_via_2opt(
        tours.clone(), max_iter=20
    )
    cost_after = _closed_loop_cost(fixed)
    assert info["final_intersections"] == 0
    assert cost_after.item() < cost_before.item()
    assert info["iters_used"] <= 20


def _test_fix_intersections_trivial_short_circuit():
    # N=2: no intersections possible, no-op.
    tours = torch.zeros(2, 2, 2, dtype=torch.float32)
    tours[:, :, 0] = torch.arange(2, dtype=torch.float32)
    fixed, info = fix_intersections_via_2opt(tours.clone())
    assert torch.equal(fixed, tours)
    assert info["iters_used"] == 0
    assert info["initial_intersections"] == 0
    assert info["final_intersections"] == 0
    # Cost is finite (not NaN) for the N=2 case.
    assert torch.isfinite(info["initial_cost"]).all().item()
    assert torch.isfinite(info["final_cost"]).all().item()


def _test_fix_intersections_batched_mixed():
    # 3-instance batch: 1 clean, 1 figure-eight, 1 out-of-order square.
    # The clean instance must remain unchanged; the two bad ones must
    # have their crossings removed and strictly lower cost.
    clean = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    crossed = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    ooo = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    seeds = torch.cat([clean, crossed, ooo], dim=0)
    cost_before = _closed_loop_cost(seeds)
    fixed, info = fix_intersections_via_2opt(seeds.clone())
    cost_after = _closed_loop_cost(fixed)
    assert info["final_intersections"] == 0
    # Instance 0 was clean; cost should be unchanged.
    assert torch.isclose(cost_after[0], cost_before[0], atol=1e-5).item()
    # Instances 1 and 2 must have strictly lower cost.
    assert cost_after[1].item() < cost_before[1].item()
    assert cost_after[2].item() < cost_before[2].item()


# ---------------------------------------------------------------------------
# run_post_revision (empty-revisers short-circuit smoke test)
# ---------------------------------------------------------------------------


def _test_run_post_revision_empty_revisers():
    # When revisers=[] the function should return the input cost and
    # seed unchanged, with no LCP_TSP call. This path requires no model
    # weights and is the only one we can exercise in the unit suite.
    tours = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    out_tours, out_costs = run_post_revision(
        tours=tours,
        revisers=[],
        get_cost_func=None,  # not used on the empty path
        opts=None,            # not used on the empty path
        post_revision_lens=[],
        post_revision_iters=[],
    )
    assert torch.equal(out_tours, tours)
    assert out_costs.shape == (1,)
    assert torch.isfinite(out_costs).all().item()
    expected = _closed_loop_cost(tours)
    assert torch.isclose(out_costs, expected, atol=1e-5).all().item()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    _test_fix_intersections_figure_eight,
    _test_fix_intersections_clean_square_noop,
    _test_fix_intersections_chord_5point,
    _test_fix_intersections_trivial_short_circuit,
    _test_fix_intersections_batched_mixed,
    _test_run_post_revision_empty_revisers,
]


def main():
    failed = 0
    for test in ALL_TESTS:
        name = test.__name__
        try:
            test()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {name}: {type(e).__name__}: {e}")

    total = len(ALL_TESTS)
    print(f"\n{'-' * 60}")
    print(f"  {total - failed}/{total} tests passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
