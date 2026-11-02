"""Single-arm benchmark uses the same soft sheet and drive, with no second arm or fixed cloth."""

import numpy as np

from phymeo.experiments.single_gripper_bend import (
    bend_measurements,
    build_single_gripper,
    loaded_bend,
)


def test_single_arm_has_only_its_own_dynamic_domains_and_colliders():
    case = build_single_gripper()
    assert set(case.initial.domains) == {"A", "cloth"}
    assert set(case.program.robots) == {"A"} and set(case.grippers) == {"A"}
    assert set(p.geometry.key for p in case.program.patches) == {"table", "A_left", "A_right"}
    assert len(case.initial.domains["cloth"].position) == 21
    assert np.all(case.program.cloth.inverse_mass > 0)
    assert np.isclose(sum(1 / case.program.cloth.inverse_mass), 0.048)
    assert all(c.domain == "A" for c in case.program.robot_contact.capsules)
    assert len(case.program.material.bending.indices) > 0 and len(case.program.finite_pads) == 2


def test_bend_metrics_gate_rigid_motion_and_a_single_support():
    case = build_single_gripper()
    state = case.initial
    angle, rms = bend_measurements(case.program, state.domains["cloth"].position)
    assert angle < 1e-10 and rms < 1e-12
    weight = 0.048 * 9.81
    o = dict(
        bottom_m=0.025,
        table_up_n=0.0,
        jaw_min_normal_n=1.0,
        jaw_up_n=weight,
        rest_hinge_change_deg=40.0,
        plane_rms_m=0.004,
    )
    assert loaded_bend(o, weight)
    for name, value in [
        ("rest_hinge_change_deg", 0.0),
        ("plane_rms_m", 0.0),
        ("jaw_min_normal_n", 0.0),
        ("jaw_up_n", 0.0),
        ("bottom_m", 0.0),
        ("table_up_n", weight),
    ]:
        assert not loaded_bend(dict(o, **{name: value}), weight)
