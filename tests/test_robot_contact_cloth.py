"""The cloth and robot-only programs consume the same robot contact loads."""

import numpy as np
from test_compliant_cloth_contact import step, triangle_case

import phymeo as ph


def make(max_depth):
    base, state, inputs = triangle_case(height=0.03, speed=0.0)
    assert hasattr(ph.CompliantClothContactProgram, "observe_robot_contact"), (
        "robot contact is not integrated"
    )
    model = base.robots["plate"].model
    system = ph.RobotContactSystem(
        {"plate": model},
        capsules=(ph.LinkCapsule("ball", "plate", 0, np.zeros(3), np.zeros(3), 0.1),),
        planes=(ph.CollisionPlane("floor", np.array([0.0, 0.0, 1.0]), -0.09),),
    )
    p = ph.CompliantClothContactProgram(
        base.scene,
        "sheet",
        base.material,
        base.surfaces,
        state.domains["sheet"].position,
        base.patches,
        robot_contact=system,
        max_penetration_m=max_depth,
    )
    return p, state, inputs


def test_body_reaction_advances_same_robot_once_and_preserves_free_cloth():
    p, state, inputs = make(0.02)
    out = step(p, state, inputs, dt=0.0001)
    assert out.report.accepted, out.report.reason
    row = out.report.substeps[0].diagnostics["robot_contact"]
    force = row["contacts"][0]["force_on_a_n"]
    np.testing.assert_allclose(2 * out.state.domains["plate"].v[0], 0.0001 * force[2], atol=1e-14)
    np.testing.assert_allclose(out.state.domains["sheet"].velocity, state.domains["sheet"].velocity)
    assert p.observe_robot_contact(out.state)["checked_pairs"] == 1


def test_excess_body_overlap_rolls_back_every_participant():
    p, state, inputs = make(0.001)
    out = step(p, state, inputs)
    assert not out.report.accepted and "robot penetration" in out.report.reason
    np.testing.assert_array_equal(out.state.domains["plate"].q, state.domains["plate"].q)
    np.testing.assert_array_equal(
        out.state.domains["sheet"].position, state.domains["sheet"].position
    )
