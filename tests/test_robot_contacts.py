"""Independent mechanics checks for joint stops and body collisions."""

import numpy as np
import pytest

import phymeo as ph


def slider(*, mass=1.0, axis=(1.0, 0.0, 0.0), base=(0.0, 0.0, 0.0), kind="prismatic"):
    return ph.RobotModel(
        np.array([-1]),
        (kind,),
        np.array([base]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([axis]),
        np.array([mass]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )


def test_joint_stops_are_forces_with_spring_energy_not_position_clamps():
    assert hasattr(ph, "JointLimits"), "public joint-stop model is missing"
    bounds = ph.JointLimits(np.array([-0.2]), np.array([0.3]), 1000.0, 8.0)
    s = ph.RobotState(np.array([0.32]), np.array([0.5]))
    row = bounds.evaluate(s)
    np.testing.assert_allclose(row["effort"], [-24.0], atol=1e-12)
    assert row["potential_j"] == pytest.approx(0.2)
    assert row["violation"][0] == pytest.approx(0.02)
    assert s.q[0] == 0.32
    opening = bounds.evaluate(ph.RobotState(s.q, -s.v))
    assert opening["effort"][0] == pytest.approx(-20.0)
    low = bounds.evaluate(ph.RobotState(np.array([-0.21]), np.array([-0.25])))
    assert low["effort"][0] == pytest.approx(12.0)
    assert not np.any(bounds.evaluate(ph.RobotState(np.zeros(1), np.ones(1)))["effort"])
    assert not bounds.lower.flags.writeable
    for args in (
        (np.array([0.0]), np.array([0.0]), 1.0, 0.0),
        (np.array([0.0]), np.array([np.inf]), 1.0, 0.0),
        (np.array([0.0]), np.array([1.0]), -1.0, 0.0),
    ):
        with pytest.raises((TypeError, ValueError)):
            ph.JointLimits(*args)
    with pytest.raises(ValueError):
        bounds.evaluate(ph.RobotState(np.zeros(2), np.zeros(2)))


def test_bounded_ik_does_not_hide_unreachable_targets_or_change_the_seed():
    assert hasattr(ph, "JointLimits"), "public joint-stop model is missing"
    model = slider()
    limits = ph.JointLimits(np.array([-0.2]), np.array([0.3]), 1000.0, 8.0)
    seed = np.array([0.32])  # permitted physical soft-stop compression
    result = ph.solve_point_ik(
        model, seed, 0, np.zeros(3), np.array([0.5, 0.0, 0.0]), limits=limits
    )
    assert not result.converged
    assert result.status == "stalled"
    assert result.configuration[0] == 0.3
    assert result.error_m == pytest.approx(0.2)
    assert seed[0] == 0.32
    good = ph.solve_point_ik(model, seed, 0, np.zeros(3), np.array([0.1, 0.0, 0.0]), limits=limits)
    assert good.converged and good.error_m < 1e-7


def capsule(key, domain, start, end, radius=0.1, link=0):
    return ph.LinkCapsule(
        key, domain, link, np.asarray(start, dtype=float), np.asarray(end, dtype=float), radius
    )


def test_capsule_pair_reactions_power_and_offset_moment():
    assert hasattr(ph, "RobotContactSystem"), "robot body-contact module is missing"
    a, b = slider(kind="revolute", axis=(0.0, 1.0, 0.0)), slider(axis=(0.0, 0.0, 1.0))
    c1 = capsule("arm", "a", [0.8, 0, 0], [1.2, 0, 0])
    c2 = capsule("other", "b", [1, -0.2, 0.15], [1, 0.2, 0.15])
    system = ph.RobotContactSystem(
        {"a": a, "b": b},
        capsules=(c1, c2),
        material=ph.RobotContactMaterial(1000.0, 0.0, 0.4, 0.01),
    )
    state = {
        "a": ph.RobotState(np.zeros(1), np.array([0.1])),
        "b": ph.RobotState(np.zeros(1), np.array([-0.2])),
    }
    out = system.evaluate(state)
    row = out["contacts"][0]
    assert row["gap_m"] == pytest.approx(-0.05)
    assert row["normal_force_n"] == pytest.approx(50.0)
    np.testing.assert_allclose(row["force_on_a_n"] + row["force_on_b_n"], 0.0, atol=1e-14)
    point, force = row["point_m"], row["force_on_a_n"]
    assert out["efforts"]["a"][0] == pytest.approx(np.cross(point, force)[1])
    assert out["efforts"]["b"][0] == pytest.approx(-force[2])
    power = sum(out["efforts"][n] @ state[n].v for n in state)
    assert power == pytest.approx(force @ row["relative_velocity_m_s"])
    tangent = force - row["normal_force_n"] * row["normal"]
    assert tangent @ row["relative_velocity_m_s"] <= 0
    assert np.linalg.norm(tangent) <= 0.4 * row["normal_force_n"] + 1e-14
    assert out["potential_j"] == pytest.approx(1.25)


def test_segment_witnesses_and_explicit_filters():
    assert hasattr(ph, "RobotContactSystem"), "robot body-contact module is missing"
    from phymeo.geometry.capsule import closest_segment_points

    cases = [
        ([0, 0, 0], [2, 0, 0], [1, -1, 1], [1, 1, 1], [1, 0, 0], [1, 0, 1]),
        ([0, 0, 0], [2, 0, 0], [1, 1, 0], [3, 1, 0], None, None),
        ([0, 0, 0], [0, 0, 0], [1, -1, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0]),
        ([0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0], [1, 0, 0], [2, 1, 0]),
    ]
    for a, b, c, d, x, y in cases:
        p, q = closest_segment_points(*[np.array(z, dtype=float) for z in (a, b, c, d)])
        if x is not None:
            np.testing.assert_allclose(p, x, atol=1e-14)
            np.testing.assert_allclose(q, y, atol=1e-14)
        else:
            assert np.linalg.norm(p - q) == pytest.approx(1.0)
    model = slider()
    c1 = capsule("one", "a", [0, 0, 0], [1, 0, 0])
    c2 = capsule("two", "b", [0, 0, 0], [1, 0, 0])
    models = {"a": model, "b": model}
    states = {n: ph.RobotState(np.zeros(1), np.zeros(1)) for n in models}
    with pytest.raises(ph.SolverFailure, match="coincident"):
        ph.RobotContactSystem(models, capsules=(c1, c2)).evaluate(states)
    out = ph.RobotContactSystem(
        models, capsules=(c1, c2), excluded_pairs=(("one", "two"),)
    ).evaluate(states)
    assert not out["contacts"]
    with pytest.raises(ValueError, match="exclusion"):
        ph.RobotContactSystem(models, capsules=(c1, c2), excluded_pairs=(("one", "typo"),))
    with pytest.raises(ValueError):
        ph.LinkCapsule("bad", "a", -1, np.zeros(3), np.ones(3), 0.1)


def test_actual_two_mass_collision_and_plane_stop():
    assert hasattr(ph, "CompliantRobotContactProgram"), "robot contact integrator is missing"
    models = {"a": slider(), "b": slider(mass=2.0)}
    bodies = (capsule("a", "a", [0, 0, 0], [0, 0, 0]), capsule("b", "b", [0, 0, 0], [0, 0, 0]))
    system = ph.RobotContactSystem(
        models, capsules=bodies, material=ph.RobotContactMaterial(2000.0, 4.0, 0.0, 0.01)
    )
    program = ph.CompliantRobotContactProgram(system, max_penetration_m=0.03)
    state = ph.SceneState(
        {
            "a": ph.RobotState(np.array([-0.15]), np.array([1.0])),
            "b": ph.RobotState(np.array([0.15]), np.array([-0.5])),
        }
    )
    inputs = ph.SceneInputs({n: ph.RobotInputs.zeros(1) for n in models})
    peak = 0.0
    with program.stepper(ph.StepSettings(substeps=400, record_substeps=True)) as runtime:
        out = runtime.step(state, inputs, 0.2)
    assert out.report.accepted, out.report.reason
    assert out.state.domains["a"].v[0] < 0 < out.state.domains["b"].v[0]
    assert out.state.domains["a"].v[0] + 2 * out.state.domains["b"].v[0] == pytest.approx(
        0.0, abs=1e-12
    )
    for rep in out.report.substeps:
        for row in rep.diagnostics["robot_contact"]["contacts"]:
            peak = max(peak, row["normal_force_n"])
    assert peak > 10.0
    assert np.array_equal(state.domains["a"].q, [-0.15])
    # Same integration with a stationary ground and finite vertical mass.
    m = slider(axis=(0.0, 0.0, 1.0))
    plane = ph.CollisionPlane("ground", np.array([0.0, 0.0, 1.0]), 0.0)
    sphere = capsule("ball", "a", [0, 0, 0], [0, 0, 0])
    p = ph.CompliantRobotContactProgram(
        ph.RobotContactSystem({"a": m}, capsules=(sphere,), planes=(plane,)), max_penetration_m=0.03
    )
    start = ph.SceneState({"a": ph.RobotState(np.array([0.13]), np.array([-1.0]))})
    with p.stepper(ph.StepSettings(substeps=400)) as runtime:
        hit = runtime.step(start, ph.SceneInputs({"a": ph.RobotInputs.zeros(1)}), 0.15)
    assert hit.report.accepted and hit.state.domains["a"].v[0] > 0
    assert any(r.diagnostics["robot_contact"]["contacts"] for r in hit.report.substeps)


def test_driven_stop_equilibrium_and_transaction_rollback():
    assert hasattr(ph, "CompliantRobotContactProgram"), "robot contact integrator is missing"
    model = slider()
    bounds = ph.JointLimits(np.array([-0.2]), np.array([0.3]), 1000.0, 50.0)
    system = ph.RobotContactSystem({"r": model}, limits={"r": bounds})
    program = ph.CompliantRobotContactProgram(system)
    state = ph.SceneState({"r": ph.RobotState(np.array([0.29]), np.zeros(1))})
    inputs = ph.SceneInputs(
        {"r": ph.RobotInputs(np.array([10.0]), np.zeros((1, 3)), np.zeros((1, 3)))}
    )
    with program.stepper(ph.StepSettings(substeps=1000)) as runtime:
        out = runtime.step(state, inputs, 0.5)
    assert out.report.accepted
    assert out.state.domains["r"].q[0] == pytest.approx(0.31, abs=0.0003)
    assert abs(out.state.domains["r"].v[0]) < 0.01
    assert out.report.substeps[-1].diagnostics["robot_contact"]["limits"]["r"]["effort"][
        0
    ] == pytest.approx(-10.0, abs=0.3)
    strict = ph.CompliantRobotContactProgram(system, max_joint_violation=0.001)
    with strict.stepper(ph.StepSettings(substeps=100)) as runtime:
        bad = runtime.step(state, inputs, 0.05)
    assert not bad.report.accepted and "joint limit" in bad.report.reason
    np.testing.assert_array_equal(bad.state.domains["r"].q, state.domains["r"].q)
    np.testing.assert_array_equal(bad.state.domains["r"].v, state.domains["r"].v)


def test_adjacent_link_contact_survives_unrelated_shape_exclusions():
    model = ph.RobotModel(
        np.array([-1, 0]),
        ("prismatic", "prismatic"),
        np.array([[0., 0., 0.], [0., 0., .15]]),
        np.array([[0., 0., 0., 1.], [0., 0., 0., 1.]]),
        np.array([[1., 0., 0.], [0., 0., 1.]]),
        np.ones(2),
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.zeros(3),
    )
    models = {"r": model}
    state = {"r": ph.RobotState(np.zeros(2), np.zeros(2))}
    shapes = (
        capsule("parent", "r", [0, 0, 0], [0, 0, 0]),
        capsule("guard", "r", [4, 0, 0], [4, 0, 0]),
        capsule("child", "r", [0, 0, 0], [0, 0, 0], link=1),
    )
    active = ph.RobotContactSystem(models, capsules=shapes).evaluate(state)
    assert active["checked_pairs"] == 2  # the two shapes on link zero cannot load each other
    assert [(r["a"], r["b"]) for r in active["contacts"]] == [("parent", "child")]
    np.testing.assert_allclose(active["efforts"]["r"], [0., 100.], atol=1e-12)

    kept = ph.RobotContactSystem(
        models, capsules=shapes, excluded_pairs=(("guard", "child"),)
    ).evaluate(state)
    assert kept["checked_pairs"] == 1
    np.testing.assert_array_equal(kept["efforts"]["r"], active["efforts"]["r"])
    clear = ph.RobotContactSystem(
        models, capsules=shapes, excluded_pairs=(("child", "parent"),)
    ).evaluate(state)
    assert clear["checked_pairs"] == 1 and clear["contacts"] == []
    np.testing.assert_array_equal(clear["efforts"]["r"], np.zeros(2))
    with pytest.raises(ValueError, match="duplicate collision exclusion"):
        ph.RobotContactSystem(
            models, capsules=shapes, excluded_pairs=(("parent", "child"), ("child", "parent"))
        )
    with pytest.raises(ValueError, match="duplicate collision shape key"):
        ph.RobotContactSystem(models, capsules=(*shapes, shapes[0]))
