"""Unit tests for utils.diagnosis.

Plain asserts under `if __name__ == "__main__":` — no pytest required, matching
the project's de-facto style (cf. the now-removed tests/test_chunk_2opt.py and
tests/test_heatmap_guided.py from history).

Run from the repo root:
    cd /home/toothlessos/Projects/nrp/GLOP && python tests/test_diagnosis.py
"""

import math
import os
import sys
import tempfile

# Make the project root importable when this file is run directly
# (e.g. `python tests/test_diagnosis.py` from the repo root).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from utils.diagnosis import (
    check_convex_hull,
    check_no_self_intersection,
    plot_tsp_tours,
)


# ---------------------------------------------------------------------------
# check_no_self_intersection
# ---------------------------------------------------------------------------

def _test_self_intersection_square_ccw():
    seeds = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    has_inter, n_inter, frac, n_pairs = check_no_self_intersection(seeds)
    assert has_inter.tolist() == [False]
    assert int(n_inter.item()) == 0
    assert int(n_pairs.item()) == 2  # only (0,2) is non-adjacent
    assert frac.item() == 0.0


def _test_self_intersection_square_crossed():
    # Figure-eight: (0,0) -> (1,1) crosses (1,0) -> (0,1).
    seeds = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    has_inter, n_inter, _, _ = check_no_self_intersection(seeds)
    assert has_inter.tolist() == [True]
    assert int(n_inter.item()) >= 1


def _test_self_intersection_pentagon():
    # Regular pentagon, CCW.
    import math as _m
    pts = []
    for k in range(5):
        theta = 2 * _m.pi * k / 5
        pts.append([_m.cos(theta), _m.sin(theta)])
    seeds = torch.tensor([pts], dtype=torch.float32)
    has_inter, _, _, _ = check_no_self_intersection(seeds)
    assert has_inter.tolist() == [False]


def _test_self_intersection_chord():
    # 5-point tour with a proper crossing at a non-vertex point.
    # Tour order in `seeds`: idx 0 -> idx 1 -> idx 2 -> idx 3 -> idx 4 -> idx 0.
    # Edges (0,0)->(2,2) and (0,2)->(2,0) cross at (1, 1) — not a tour vertex.
    seeds = torch.tensor(
        [
            [
                [0.0, 0.0],   # 0: (0,0)
                [2.0, 2.0],   # 1: (2,2)
                [1.0, 0.0],   # 2: (1,0)
                [0.0, 2.0],   # 3: (0,2)
                [2.0, 0.0],   # 4: (2,0)
            ]
        ],
        dtype=torch.float32,
    )
    has_inter, _, _, _ = check_no_self_intersection(seeds)
    assert has_inter.tolist() == [True]


def _test_self_intersection_trivial():
    # N < 3: no edges can cross.
    for n in (1, 2):
        seeds = torch.zeros(2, n, 2, dtype=torch.float32)
        seeds[:, :, 0] = torch.arange(n, dtype=torch.float32)
        has_inter, n_inter, frac, n_pairs = check_no_self_intersection(seeds)
        assert has_inter.tolist() == [False, False]
        assert int(n_inter.sum().item()) == 0
        assert int(n_pairs.sum().item()) == 0
        # Fraction must be 0 (not NaN) when there are no valid pairs.
        assert torch.isnan(frac).sum().item() == 0
        assert frac.sum().item() == 0.0
    # N=3: triangle.
    seeds = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]], dtype=torch.float32
    )
    has_inter, _, _, _ = check_no_self_intersection(seeds)
    assert has_inter.tolist() == [False]


def _test_self_intersection_batched_mixed():
    square_ccw = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32
    )
    square_crossed = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
    )
    square_ooo = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32
    )
    seeds = torch.cat([square_ccw, square_crossed, square_ooo], dim=0)
    has_inter, n_inter, frac, n_pairs = check_no_self_intersection(seeds)
    assert has_inter.tolist() == [False, True, True]
    assert int(n_inter[1].item()) >= 1
    # Sanity: fraction == n_inter / n_pairs for each row.
    expected_frac = n_inter.float() / n_pairs.float().clamp_min(1)
    assert torch.allclose(frac, expected_frac, atol=1e-5)


# ---------------------------------------------------------------------------
# check_convex_hull
# ---------------------------------------------------------------------------

def _test_convex_hull_square_ccw():
    seeds = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    consistency = check_convex_hull(seeds)
    assert consistency.tolist() == [1.0]


def _test_convex_hull_square_cw():
    # Reversed: hull vertices still in cyclic order, just opposite direction.
    seeds = torch.tensor(
        [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]],
        dtype=torch.float32,
    )
    consistency = check_convex_hull(seeds)
    assert consistency.tolist() == [1.0]


def _test_convex_hull_square_cyclic_shift():
    # Same square, starting at a different vertex.
    seeds = torch.tensor(
        [[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    consistency = check_convex_hull(seeds)
    assert consistency.tolist() == [1.0]


def _test_convex_hull_pentagon_with_interior():
    # Pentagon (CCW hull) + one interior point appended at the end.
    pts = []
    for k in range(5):
        theta = 2 * math.pi * k / 5
        pts.append([math.cos(theta), math.sin(theta)])
    pts.append([0.0, 0.0])  # interior point
    seeds = torch.tensor([pts], dtype=torch.float32)
    consistency = check_convex_hull(seeds)
    assert consistency.tolist() == [1.0]


def _test_convex_hull_out_of_order():
    # Square visited out of cyclic order — should fail the hull property.
    seeds = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    consistency = check_convex_hull(seeds)
    assert consistency.tolist() == [0.0]


def _test_convex_hull_collinear():
    # All points on a line — degenerate hull, trivially satisfies property.
    seeds = torch.tensor(
        [[[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    consistency = check_convex_hull(seeds)
    assert consistency.tolist() == [1.0]


def _test_convex_hull_with_duplicate():
    # Duplicate point at index 1 — should not break the hull check.
    seeds = torch.tensor(
        [
            [
                [0.0, 0.0],
                [0.0, 0.0],  # duplicate
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    consistency = check_convex_hull(seeds)
    assert consistency.tolist() == [1.0]


def _test_convex_hull_batched_mixed():
    # Uniform N within a batch — the implementation pads/truncates implicitly
    # by per-instance scipy calls, but the tensor shape must match.
    square_ccw = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32
    )
    square_ooo = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32
    )
    square_cyclic = torch.tensor(
        [[[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]], dtype=torch.float32
    )
    seeds = torch.cat([square_ccw, square_ooo, square_cyclic], dim=0)
    consistency = check_convex_hull(seeds)
    assert consistency.tolist() == [1.0, 0.0, 1.0]


def _test_convex_hull_trivial():
    # N < 3: trivially true.
    for n in (1, 2):
        seeds = torch.zeros(2, n, 2, dtype=torch.float32)
        seeds[:, :, 0] = torch.arange(n, dtype=torch.float32)
        consistency = check_convex_hull(seeds)
        assert consistency.tolist() == [1.0, 1.0]


# ---------------------------------------------------------------------------
# plot_tsp_tours
# ---------------------------------------------------------------------------


def _test_plot_tsp_tours_intersection():
    # Figure-eight: hull is OK (4 corners), but the two diagonals cross.
    seeds = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    has_i, n_i, _, _, inter = check_no_self_intersection(seeds)
    consistency, _ = check_convex_hull(seeds)
    assert has_i.tolist() == [True]  # precondition

    with tempfile.TemporaryDirectory() as d:
        path = plot_tsp_tours(
            seeds,
            has_intersection=has_i,
            num_intersections=n_i,
            intersection_results=inter,
            consistency=consistency,
            out_dir=d,
            tag="intersection_test",
            max_plots=4,
        )
        assert path is not None
        assert os.path.isfile(path)
        assert path.endswith("tour_diag_intersection_test.png")
        assert os.path.getsize(path) > 1024  # not empty


def _test_plot_tsp_tours_hull_violation():
    # Square visited out of cyclic order — hull violated, no intersection.
    seeds = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]],
        dtype=torch.float32,
    )
    has_i, n_i, _, _, inter = check_no_self_intersection(seeds)
    consistency, _ = check_convex_hull(seeds)
    assert consistency.tolist() == [0.0]  # precondition

    with tempfile.TemporaryDirectory() as d:
        path = plot_tsp_tours(
            seeds,
            has_intersection=has_i,
            num_intersections=n_i,
            intersection_results=inter,
            consistency=consistency,
            out_dir=d,
            tag="hull_fail_test",
            max_plots=4,
        )
        assert path is not None
        assert os.path.isfile(path)
        assert path.endswith("tour_diag_hull_fail_test.png")
        assert os.path.getsize(path) > 1024


def _test_plot_tsp_tours_clean_skips():
    # Clean CCW square: no intersections, hull OK -> no file should be created.
    seeds = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        dtype=torch.float32,
    )
    has_i, n_i, _, _, inter = check_no_self_intersection(seeds)
    consistency, _ = check_convex_hull(seeds)
    assert has_i.tolist() == [False]  # precondition
    assert consistency.tolist() == [1.0]  # precondition

    with tempfile.TemporaryDirectory() as d:
        path = plot_tsp_tours(
            seeds,
            has_intersection=has_i,
            num_intersections=n_i,
            intersection_results=inter,
            consistency=consistency,
            out_dir=d,
            tag="clean_test",
        )
        assert path is None
        assert os.listdir(d) == []  # nothing written


def _test_plot_tsp_tours_batched():
    # 3-instance batch: 1 clean, 1 intersecting (figure-eight), 1 hull-fail.
    # Note: the figure-eight fixture also violates the convex-hull property
    # (its hull vertices are visited in order 0,1,2,3 but the CCW hull
    # ordering is 0,2,1,3), so both flags fire for instance 1.
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
    has_i, n_i, _, _, inter = check_no_self_intersection(seeds)
    consistency, _ = check_convex_hull(seeds)
    assert has_i.tolist() == [False, True, True]
    # The figure-eight fixture also fails the hull check: its hull vertex
    # order (0,1,2,3) is not a cyclic rotation/reversal of the CCW hull
    # sequence (0,2,1,3).
    assert consistency.tolist() == [1.0, 0.0, 0.0]

    with tempfile.TemporaryDirectory() as d:
        path = plot_tsp_tours(
            seeds,
            has_intersection=has_i,
            num_intersections=n_i,
            intersection_results=inter,
            consistency=consistency,
            out_dir=d,
            tag="batch_test",
            max_plots=8,
        )
        assert path is not None
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 1024


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    _test_self_intersection_square_ccw,
    _test_self_intersection_square_crossed,
    _test_self_intersection_pentagon,
    _test_self_intersection_chord,
    _test_self_intersection_trivial,
    _test_self_intersection_batched_mixed,
    _test_convex_hull_square_ccw,
    _test_convex_hull_square_cw,
    _test_convex_hull_square_cyclic_shift,
    _test_convex_hull_pentagon_with_interior,
    _test_convex_hull_out_of_order,
    _test_convex_hull_collinear,
    _test_convex_hull_with_duplicate,
    _test_convex_hull_batched_mixed,
    _test_convex_hull_trivial,
    _test_plot_tsp_tours_intersection,
    _test_plot_tsp_tours_hull_violation,
    _test_plot_tsp_tours_clean_skips,
    _test_plot_tsp_tours_batched,
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
