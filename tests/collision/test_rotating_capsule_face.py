"""Real angular motion coupled to a deforming face, with continuous frame transitions."""

from math import acos, pi

import numpy as np
import pytest
from rotating_face_cases import switching_segment, turning_face

import phymeo as ph


def row(model, paths, mode="all_pairs"):
    with ph.CollisionWorkspace(model, broad_phase=mode) as workspace:
        return ph.detect(model, paths, ph.CollisionPolicy(), workspace).pairs[0]


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("change_frame", [False, True])
def test_true_turn_hits_an_independently_deforming_face(swap, change_frame):
    model, paths, cap, cloth = turning_face(swap=swap, change_frame=change_frame)
    hit = row(model, paths, "bvh")
    assert hit.status == "hit" and hit.sweep.start_gap_m > 0 and hit.sweep.end_gap_m > 0
    lo, hi = hit.sweep.first_t
    assert lo <= 0.5 <= hi and hi - lo < 1e-8
    assert hit.sweep.witness_time == hi
    w = hit.sweep.witness
    cap_weights, cloth_weights = (
        (w.second_weights, w.first_weights) if swap else (w.first_weights, w.second_weights)
    )
    assert {i for i, _ in cap_weights} == {15, 6}
    assert {i for i, _ in cloth_weights} == {27, 36, 49}
    assert abs(sum(v for _, v in cloth_weights) - 1) < 1e-12
    assert w.gap_m <= 1e-10 and w.capsule_axis_span
    cp, fp = (w.second_point, w.first_point) if swap else (w.first_point, w.second_point)
    t = hit.sweep.witness_time
    cloth_point = sum(
        dict(cloth_weights)[node] * ((1 - t) * paths[cloth].start[i] + t * paths[cloth].end[i])
        for i, node in enumerate((27, 36, 49))
    )
    np.testing.assert_allclose(fp, cloth_point, atol=1e-10)
    np.testing.assert_allclose(cp, fp, atol=1e-7)
    # The endpoint chord would stay at x=0 (apart from common translation);
    # every material point of this triangle has local x>=5/8.
    clear = row(*turning_face(clear=True, swap=swap, change_frame=change_frame)[:2], "sap")
    assert clear.candidate and clear.status == "separated" and clear.sweep.first_t is None


def test_one_workspace_birth_contact_exit_and_rejected_retry():
    model, _, _, _ = turning_face()
    phases = ((0, 0.4), (0.4, 0.53), (0.53, 0.55), (0.55, 0.70), (0.70, 1.0))
    with ph.CollisionWorkspace(model, broad_phase="bvh") as work:
        rows = []
        for i, (a, b) in enumerate(phases):
            _, paths, _, _ = turning_face(a, b)
            if i == 1:
                last = work.last_result
                failed = ph.detect(model, paths, ph.CollisionPolicy(candidate_capacity=0), work)
                assert failed.status == "capacity_exceeded" and work.last_result is last
            rows.append(ph.detect(model, paths, ph.CollisionPolicy(), work).pairs[0])
        assert [p.status for p in rows] == ["separated", "hit", "hit", "hit", "separated"]
        lo, hi = rows[1].sweep.first_t
        assert 0.4 + 0.13 * lo <= 0.5 <= 0.4 + 0.13 * hi
        assert rows[2].sweep.first_t == (0, 0) and rows[2].sweep.end_gap_m < 0
        assert rows[3].sweep.first_t == (0, 0) and rows[3].sweep.end_gap_m > 0


def test_true_arc_can_reject_a_chord_only_hit():
    model, paths, cap, cloth = turning_face()
    paths[cloth] = ph.CollisionMotion([[-0.25, -0.25, 0], [0.25, -0.25, 0], [0, 0.25, 0]])
    paths[cap] = ph.CollisionMotion(
        [[0, -1, -0.5], [0, -1, 0.5]], turn=ph.RigidTurn([0, 0, 0], [0, 0, 0], [0, 0, 1], pi)
    )
    p = row(model, paths)
    assert p.candidate and p.status == "separated"
    # The straight endpoint chord crosses the triangle at t=1/2.
    paths[cap] = ph.CollisionMotion(paths[cap].start, [[0, 1, -0.5], [0, 1, 0.5]])
    assert row(model, paths).status == "hit"


def test_degeneracy_resource_limit_and_unhandled_thickness_are_explicit():
    model, paths, cap, cloth = turning_face()
    first = paths[cloth].start
    degenerate = {**paths, cloth: ph.CollisionMotion(first, first[[0, 2, 1]])}
    assert row(model, degenerate).status == "unresolved"
    large = {
        **paths,
        cap: ph.CollisionMotion(
            paths[cap].start, turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), 100)
        ),
    }
    assert row(model, large).reason == "rotation_face_sector_capacity"
    thick = ph.CollisionModel(
        [model.shapes[0], ph.CollisionShape(cloth, "triangle", (27, 36, 49), 0.02)]
    )
    assert row(thick, paths).status == "unsupported"


def test_first_contact_on_both_edge_interiors_with_moving_opposite_vertex():
    model, paths, cap, cloth = turning_face()
    paths[cap] = ph.CollisionMotion(
        [[0, -1, -0.5], [0, -1, 0.5]], turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), pi)
    )
    start = np.array([[0.75, 0, 0], [1.25, 0, 0], [1, 0.4, 0]])
    end = start.copy()
    end[2, 2] = 0.25
    paths[cloth] = ph.CollisionMotion(start, end)
    p = row(model, paths)
    expected = acos(0.125) / pi
    assert p.status == "hit" and p.sweep.first_t[0] <= expected <= p.sweep.first_t[1]
    w = p.sweep.witness
    assert dict(w.first_weights)[15] == pytest.approx(0.5, abs=1e-7)
    face = dict(w.second_weights)
    assert 0 < face[27] < 1 and 0 < face[36] < 1 and face[49] < 1e-7


def test_unresolved_isolated_midarc_touch_is_never_reported_as_clear():
    model, paths, cap, cloth = turning_face()
    paths[cap] = ph.CollisionMotion(
        [[0, -1, -0.5], [0, -1, 0.5]], turn=ph.RigidTurn((0, 0, 0), (0, 0, 0), (0, 0, 1), 2 * pi)
    )
    paths[cloth] = ph.CollisionMotion([[-0.25, 1.125, -1], [0.25, 1.125, -1], [0, 1.125, 1]])
    p = row(model, paths)
    assert p.status == "unresolved" and p.reason in {
        "rotation_contact_unresolved",
        "rotation_cell_capacity",
    }


def test_persistent_contact_switches_material_caps_through_full_shaft():
    model, _ = switching_segment(0, 0.5)
    segments = []
    with ph.CollisionWorkspace(model, broad_phase="sap") as work:
        for a, b in ((0, 0.5), (0.5, 1)):
            _, paths = switching_segment(a, b)
            p = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs[0]
            assert p.status == "hit" and p.sweep.first_t == (0.0, 0.0)
            assert p.sweep.start_gap_m < 0 and p.sweep.end_gap_m < 0
            segments.extend((p.sweep.witness, p.witness))
    spans = [tuple(x.axis_parameter for x in w.capsule_axis_span) for w in segments]
    assert spans == [(0.0, 0.0), (0.0, 1.0), (0.0, 1.0), (1.0, 1.0)]


def test_reversing_axis_and_angle_retains_first_event():
    model, paths, cap, _ = turning_face(change_frame=True)
    old = row(model, paths)
    motion = paths[cap]
    turn = motion.turn
    paths[cap] = ph.CollisionMotion(
        motion.start,
        turn=ph.RigidTurn(turn.pivot_start, turn.pivot_end, -np.asarray(turn.axis), -turn.angle),
    )
    assert row(model, paths) == old
