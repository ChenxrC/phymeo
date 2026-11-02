"""The structural collision geometry follows the real palm and moving fingers."""

import numpy as np
import pytest

import phymeo as ph
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.experiments.compliant_handoff import build_handoff
from phymeo.experiments.handoff_grippers import grippers_for


@pytest.mark.parametrize("arm,home", [("A", -0.06), ("B", 0.115)])
def test_mounted_wrist_has_actual_finger_reach_and_connected_shanks(arm, home):
    g = grippers_for("articulated-palm", roles=(arm,))[arm]
    np.testing.assert_allclose(g.position(g.initial), [home, 0.039], atol=1e-10)
    by_name = {k.key: k for k in g.capsules}
    assert set(by_name) == {
        arm + "_" + key for key in ("upper", "fore", "wrist", "left_shank", "right_shank")
    }
    wrist = by_name[arm + "_wrist"]
    assert abs(g.tool_point[0] - wrist.start[0]) == pytest.approx(0.045)
    assert wrist.start[0] == wrist.end[0]
    pose = forward_kinematics(g.model, g.initial.q)

    def point(shape, at):
        return pose.position[shape.link_index] + pose.rotation[shape.link_index] @ at

    endpoints = [point(wrist, p) for p in (wrist.start, wrist.end)]
    for side, link in zip(("left", "right"), g.fingers, strict=True):
        shank = by_name[arm + "_" + side + "_shank"]
        assert shank.link_index == link
        start = point(shank, shank.start)
        axis = endpoints[1] - endpoints[0]
        u = (start - endpoints[0]) @ axis / (axis @ axis)
        assert -1e-12 <= u <= 1 + 1e-12
        np.testing.assert_allclose(start, endpoints[0] + u * axis, atol=1e-12)
        assert abs(shank.end[0]) == pytest.approx(0.022)
        assert abs(shank.end[1]) - shank.radius_m == pytest.approx(0.004)


@pytest.mark.parametrize("tilt", [0.0, 2.0])
def test_full_new_assembly_clears_the_rounded_material_sheet_at_initial_pose(tilt):
    case = build_handoff(robot="articulated-palm", fabric="soft", initial_tilt_deg=tilt)
    prog = case.program
    assert prog.cloth_links is not None and len(prog.robot_contact.capsules) == 10
    initial = prog.observe_robot_contact(case.initial)
    assert initial["cloth_links"] == [] and initial["cloth_link_pairs"] == 240
    cloth = case.initial.domains["cloth"].position
    links = prog.robot_contact.capsules
    poses = {
        n: forward_kinematics(g.model, case.initial.domains[n].q) for n, g in case.grippers.items()
    }
    faces = [
        ph.CollisionShape(ph.CollisionKey(0, 0, i), "triangle", tuple(ids), group=1, mask=2)
        for i, ids in enumerate(prog.faces)
    ]
    caps = [
        ph.CollisionShape(
            ph.CollisionKey(0, 1 + i, 0), "capsule", (0, 1), link.radius_m + 0.0015, group=2, mask=1
        )
        for i, link in enumerate(links)
    ]
    model = ph.CollisionModel(tuple(faces + caps))
    motion = {f.key: ph.CollisionMotion(cloth[list(f.vertex_ids)]) for f in faces}
    for cap, link in zip(caps, links, strict=True):
        pose = poses[link.domain]
        pts = (
            np.array([link.start, link.end]) @ pose.rotation[link.link_index].T
            + pose.position[link.link_index]
        )
        motion[cap.key] = ph.CollisionMotion(pts)
    with ph.CollisionWorkspace(model) as workspace:
        detected = ph.detect(model, motion, ph.CollisionPolicy(margin=0.0015), workspace)
    rd = [pair for pair in detected.pairs if pair.family == "RD"]
    assert len(rd) == 240
    assert all(r.status == "separated" for r in rd), [
        (r.first, r.second, r.status) for r in rd if r.status != "separated"
    ]
