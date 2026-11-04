"""Boundary addresses are geometric, stable under transport, and independent of the triangle list order."""

import numpy as np
import pytest

from phymeo.control.material_grasp import ParallelJawWindow


def jaws(R=np.eye(3), shift=np.zeros(3), axial=0.02):
    uv = [[-0.04, -0.04], [-0.04, 0.04], [0.04, 0.04], [0.04, -0.04]]
    return ParallelJawWindow(
        *[np.array([[x, y, z] for x, z in uv]) @ R.T + shift for y in (-axial, axial)]
    )


def test_shared_diagonal_and_fan_edges_are_not_targets_and_blocked_ties_use_stable_addresses():
    from phymeo.control.boundary_edge import BoundaryEdges

    vertices = np.array([[-0.1, 0, -0.1], [0.1, 0, -0.1], [0.1, 0, 0.1], [-0.1, 0, 0.1], [0, 0, 0]])
    topologies = (((0, 1, 2), (0, 2, 3)), ((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
    R = np.array([[0.0, 0, 1], [1.0, 0, 0], [0, 1.0, 0]])
    for faces in topologies:
        edges = BoundaryEdges(tuple(reversed(faces)))
        assert set(edges.edges) == {(0, 1), (1, 2), (2, 3), (0, 3)}
        for rotation, shift in ((np.eye(3), np.zeros(3)), (R, np.array([1.25, -2.0, 3.0]))):
            x = vertices @ rotation.T + shift
            # A real finite capsule around this short axis masks the otherwise equally near (0,1).
            axis = np.array([[-0.015, 0, -0.1], [0.015, 0, -0.1]]) @ rotation.T + shift

            def finite_rod(target):
                p = target.evaluate(x)
                ab = axis[1] - axis[0]
                w = np.clip((p - axis[0]) @ ab / (ab @ ab), 0.0, 1.0)
                return (
                    ("wrist", "capsule") if np.linalg.norm(p - axis[0] - w * ab) <= 0.015 else None
                )

            result = edges.choose(
                x, jaws(rotation, shift), axial_clearance=0.005, obstruction=finite_rod
            )
            assert result.target.nodes == (0, 3)
            assert result.target.fraction == pytest.approx(0.5, abs=1e-13)
            assert [trial.edge for trial in result.trials] == [(0, 1), (0, 3)]
            assert result.trials[0].obstruction == ("wrist", "capsule")
            assert result.axial_candidates == 4


def test_only_part_of_oblique_edge_lies_in_aperture_and_axial_or_structural_failure_is_visible():
    from phymeo.control.boundary_edge import BoundaryEdges

    x = np.array([[0.02, -0.08, 0.03], [0.06, 0.08, 0.03], [0.6, 0.2, 0.4]])
    edges = BoundaryEdges([(0, 1, 2)])
    sel = edges.choose(x, jaws(), axial_clearance=0.005, obstruction=lambda p: None)
    assert sel.target.nodes == (0, 1)
    p = sel.target.evaluate(x)
    assert -0.015 - 1e-14 <= p[1] <= 0.015 + 1e-14
    assert 0 < sel.target.fraction < 1
    reject = edges.choose(
        x, jaws(), axial_clearance=0.005, obstruction=lambda p: ("B_wrist", "face_5")
    )
    assert reject.target is None and reject.trials and all(t.obstruction for t in reject.trials)
    far = edges.choose(
        x + [0, 1, 0],
        jaws(),
        axial_clearance=0.005,
        obstruction=lambda p: pytest.fail("no reachable address"),
    )
    assert far.target is None and not far.trials and far.axial_candidates == 0


def test_duplicate_nonmanifold_and_closed_surfaces_have_no_admissible_boundary():
    from phymeo.control.boundary_edge import BoundaryEdges

    for faces in (
        ((0, 1, 2), (2, 1, 0)),
        ((0, 1, 2), (0, 1, 3), (0, 1, 4)),
        ((0, 1, 2), (0, 3, 4)),
        ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    ):
        with pytest.raises(ValueError):
            BoundaryEdges(faces)


def test_real_mounted_wrist_rejects_sheet_through_its_axis_and_free_mouth_latches_one_address():
    from phymeo.experiments.compliant_cloth import frame_record
    from phymeo.experiments.compliant_handoff import build_handoff
    from phymeo.experiments.handoff_boundary import MountedBoundaryAccess

    case = build_handoff(robot="articulated-palm", fabric="soft", grid=(3, 2), self_contact=True)
    original = frame_record(case.program, case.initial, 0.0)
    window = ParallelJawWindow(
        original["patch_vertices_m"]["B_left"], original["patch_vertices_m"]["B_right"]
    )
    x = np.asarray(original["cloth_position_m"])
    # A twelve by five mm physical sheet patch with its corner placed at either real hardware point.
    patch = 0.1 * (x - x[0])

    def placed(point, t):
        return {**original, "t": t, "cloth_position_m": patch + point}

    mouth = MountedBoundaryAccess(case)
    target = mouth.observe(placed(window.center, 0.0), window, 0.003)
    assert (
        target is not None
        and target.nodes == (0, 1)
        and target.fraction == pytest.approx(0.0, abs=1e-10)
    )
    assert mouth.report["trials"][-1]["obstruction"] is None
    new = placed(window.center + [0, 0.5, 0], 0.1)
    assert mouth.observe(new, window, 0.003) is target
    assert not window.contains(target.evaluate(new["cloth_position_m"]), clearance=0.0035)
    a, b = np.array(original["body_capsules_m"]["B_wrist"])
    wrist = a + (b - a) * (-a[1] / (b[1] - a[1]))
    jam = MountedBoundaryAccess(case)
    assert jam.observe(placed(wrist, 0.0), window, 0.003) is None
    assert jam.report["axial_candidates"] == 6
    assert any(trial["obstruction"][0] == "B_wrist" for trial in jam.report["trials"])


def test_axial_clip_returns_an_address_that_repasses_the_exact_measured_guard():
    from phymeo.control.boundary_edge import BoundaryEdges

    # Real oblique segment with a clipped fraction whose interpolation rounds just outside 3 mm.
    x = np.array(
        [
            [-0.03061761211773856, 0.0025852478801060333, 0.14250928511017405],
            [-0.02450414142190419, 0.020840440819402232, 0.13558260295856767],
            [-0.4, 0.2, 0.4],
        ]
    )
    shift = np.array([0.07622293678678818, 0.0, 0.023378537495358565])
    window = jaws(shift=shift, axial=0.012999999999999998)
    original = window.approach_target(x, (0, 1), axial_clearance=0.003)
    assert (
        window.approach_target(x, (0, 1), preferred=original, axial_clearance=0.003) is not original
    )
    chosen = (
        BoundaryEdges([(0, 1, 2)])
        .choose(
            x,
            window,
            axial_clearance=0.003,
            obstruction=lambda a: None if a.nodes == (0, 1) else ("rod", "face"),
        )
        .target
    )
    assert chosen is not None and chosen.fraction == np.nextafter(original.fraction, 0.0)
    assert window.approach_target(x, (0, 1), preferred=chosen, axial_clearance=0.003) is chosen


def test_mounted_adapter_rejects_axes_outside_its_declared_planar_hardware():
    from dataclasses import replace

    from phymeo.experiments.compliant_cloth import frame_record
    from phymeo.experiments.compliant_handoff import build_handoff
    from phymeo.experiments.handoff_boundary import MountedBoundaryAccess

    case = build_handoff(robot="articulated-palm", fabric="soft", grid=(3, 2), self_contact=True)
    receiver = case.grippers["B"]
    axes = receiver.model.joint_axis.copy()
    axes[0] = [0, 0, 1]
    tilted = replace(receiver, model=replace(receiver.model, joint_axis=axes))
    alternative = replace(case, grippers={**case.grippers, "B": tilted})
    with pytest.raises(ValueError, match="planar"):
        MountedBoundaryAccess(alternative)
    original = frame_record(case.program, case.initial, 0.0)
    R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="jaw axis"):
        MountedBoundaryAccess(case).observe(original, jaws(R), 0.003)
