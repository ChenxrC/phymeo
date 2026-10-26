"""Independent checks for the explicitly limited rigid sphere/plane contact law."""

from dataclasses import replace

import numpy as np
import pytest

import phymeo
from phymeo import (
    Plane,
    RigidBodyInputs,
    RigidBodyModel,
    RigidBodyState,
    RigidSemiImplicitSolver,
    Stepper,
)


def api(name):
    cls = getattr(phymeo, name, None)
    assert cls is not None, f"{name} is not implemented"
    return cls


def scene(dtype=np.float64, count=1):
    model = RigidBodyModel(np.ones(count, dtype), np.ones((count, 3), dtype), np.zeros(3))
    state = RigidBodyState(
        np.tile(np.array([0, 1, 0], dtype), (count, 1)),
        np.tile(np.array([0, 0, 0, 1], dtype), (count, 1)),
        np.tile(np.array([5, -20, 3], dtype), (count, 1)),
        np.tile(np.array([0, 2, 0], dtype), (count, 1)),
    )
    return model, state, RigidBodyInputs.zeros(count, dtype=dtype)


def shapes(values=(0.2,), dtype=np.float64):
    return api("SphereShapes")(np.array(values, dtype=dtype))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_shapes_own_readonly_positive_radii(dtype):
    source = np.array([0.2, 0.4], dtype)
    geometry = api("SphereShapes")(source)
    source[:] = 9
    np.testing.assert_allclose(geometry.radii, [0.2, 0.4])
    assert geometry.radii.dtype == dtype
    assert not geometry.radii.flags.writeable


@pytest.mark.parametrize(
    "radii",
    [
        np.array([0.0]),
        np.array([-1.0]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([[1.0]]),
        np.array([1]),
        np.array([True]),
        np.array([1 + 2j]),
        np.array([1], np.float16),
    ],
)
def test_invalid_geometry_is_rejected(radii):
    with pytest.raises((TypeError, ValueError)):
        api("SphereShapes")(radii)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_plastic_contact_preserves_tangent_and_free_rotation_without_launch_velocity(dtype):
    model, state, inputs = scene(dtype)
    before = state.copy()
    solver = api("RigidSpherePlaneSolver")(shapes(dtype=dtype))
    result = Stepper(solver, required_features={"sphere_plane_contact"}).step(
        model, state, inputs, 0.1
    )
    assert result.report.accepted
    np.testing.assert_allclose(result.state.position, [[0.5, 0.2, 0.3]], atol=1e-7)
    np.testing.assert_allclose(result.state.velocity, [[5, 0, 3]], atol=1e-6)
    np.testing.assert_array_equal(result.state.angular_velocity, state.angular_velocity)
    np.testing.assert_allclose(
        result.state.orientation, [[0, np.sin(0.1), 0, np.cos(0.1)]], atol=1e-7
    )
    # Unit mass: exactly the normal kinetic energy is removed by this contact law.
    lost_energy = 0.5 * np.sum(state.velocity.astype(np.float64) ** 2 - result.state.velocity**2)
    assert lost_energy == pytest.approx(200, abs=2e-5)
    for field in ("position", "orientation", "velocity", "angular_velocity"):
        np.testing.assert_array_equal(getattr(state, field), getattr(before, field))
        assert not np.shares_memory(getattr(result.state, field), getattr(state, field))


def test_masses_and_radii_do_not_change_the_plastic_normal_velocity_condition():
    _, state, inputs = scene(count=2)
    model = RigidBodyModel(np.array([1.0, 0.25]), np.ones((2, 3)), np.zeros(3))
    result = api("RigidSpherePlaneSolver")(shapes((0.2, 0.4))).step(model, state, inputs, 0.1)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.position[:, 1], [0.2, 0.4])
    np.testing.assert_array_equal(result.state.velocity[:, 1], [0, 0])


def test_separating_velocity_is_preserved_even_if_initial_overlap_is_stabilized():
    model, state, inputs = scene()
    state.position[0, 1], state.velocity[0, 1] = 0.1, 0.5
    result = api("RigidSpherePlaneSolver")(shapes()).step(model, state, inputs, 0.1)
    assert result.report.accepted
    assert result.state.position[0, 1] == pytest.approx(0.2)
    assert result.state.velocity[0, 1] == 0.5


def test_noncontact_step_matches_full_free_rigid_solution():
    model, state, inputs = scene()
    state.position[0, 1] = 100
    inputs.torque[0, 0] = 2
    expected = RigidSemiImplicitSolver().step(model, state, inputs, 0.1)
    actual = api("RigidSpherePlaneSolver")(shapes()).step(model, state, inputs, 0.1)
    assert actual.report.accepted
    for field in ("position", "orientation", "velocity", "angular_velocity"):
        np.testing.assert_array_equal(getattr(actual.state, field), getattr(expected.state, field))


def test_fixed_penetration_is_reported_without_moving_fixed_body():
    _, state, inputs = scene()
    model = RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3))
    state.position[0, 1] = 0.25
    result = api("RigidSpherePlaneSolver")(shapes((0.5,)), reference_length=0.1).step(
        model, state, inputs, 0.1
    )
    assert not result.report.accepted and result.report.reason
    assert result.report.max_constraint_error == pytest.approx(0.25)
    assert result.report.max_residual == pytest.approx(2.5)
    np.testing.assert_array_equal(result.state.position, state.position)
    np.testing.assert_array_equal(result.state.orientation, state.orientation)
    np.testing.assert_array_equal(result.state.velocity, [[0, 0, 0]])
    np.testing.assert_array_equal(result.state.angular_velocity, [[0, 0, 0]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty_geometry_and_state_are_supported(dtype):
    model, state, inputs = scene(dtype=dtype, count=0)
    result = Stepper(api("RigidSpherePlaneSolver")(shapes((), dtype))).step(
        model, state, inputs, 0.1
    )
    assert result.report.accepted
    assert result.state.orientation.shape == (0, 4)


@pytest.mark.parametrize("dtype,radius,height", [(np.float32, 0.1, 0.6), (np.float64, 1e-20, 1.0)])
def test_represented_contact_height_does_not_shrink_the_sphere(dtype, radius, height):
    model, state, inputs = scene(dtype=dtype)
    state.position[:] = 0
    state.velocity[:] = 0
    geometry = shapes((radius,), dtype)
    result = api("RigidSpherePlaneSolver")(geometry, plane=Plane(height=height), tolerance=0).step(
        model, state, inputs, 0.1
    )
    assert result.report.accepted
    gap = (float(result.state.position[0, 1]) - height) - float(geometry.radii[0])
    assert gap >= 0
    assert result.report.max_constraint_error == 0
    assert result.state.velocity[0, 1] == 0


@pytest.mark.parametrize("bad", ["prediction", "contact_height"])
def test_nonfinite_prediction_or_contact_bound_returns_original_snapshot(bad):
    model, state, inputs = scene()
    geometry, plane, dt = shapes(), Plane(), 10.0
    if bad == "prediction":
        state.velocity[:] = 1e308
    else:
        state.velocity[:] = 0
        geometry, plane = shapes((1e308,)), Plane(height=1e308)
    result = api("RigidSpherePlaneSolver")(geometry, plane=plane).step(model, state, inputs, dt)
    assert not result.report.accepted and result.report.reason
    for field in ("position", "orientation", "velocity", "angular_velocity"):
        np.testing.assert_array_equal(getattr(result.state, field), getattr(state, field))


@pytest.mark.parametrize("difference", ["radii", "plane", "free"])
def test_fallback_cannot_change_contact_physics(difference):
    primary = api("RigidSpherePlaneSolver")(shapes())
    if difference == "free":
        fallback = RigidSemiImplicitSolver()
    elif difference == "radii":
        fallback = api("RigidSpherePlaneSolver")(shapes((0.3,)))
    else:
        fallback = api("RigidSpherePlaneSolver")(shapes(), plane=Plane(height=0.1))
    with pytest.raises(ValueError, match="physical problem"):
        Stepper(primary, fallback)


def test_signature_is_immutable_and_excludes_solver_tolerances():
    source = np.array([0.2])
    primary = api("RigidSpherePlaneSolver")(api("SphereShapes")(source))
    signature = primary.problem_signature()
    source[:] = 0.9
    fallback = replace(primary, tolerance=1e-3, reference_length=2.0)
    assert signature == fallback.problem_signature()
    assert isinstance(signature.radii, tuple)
    Stepper(primary, fallback)


@pytest.mark.parametrize("bad", ["count", "dtype"])
def test_geometry_must_match_the_rigid_state(bad):
    model, state, inputs = scene()
    geometry = shapes((0.2, 0.4)) if bad == "count" else shapes(dtype=np.float32)
    with pytest.raises(ValueError):
        api("RigidSpherePlaneSolver")(geometry).step(model, state, inputs, 0.1)


@pytest.mark.parametrize(
    "options",
    [
        {"tolerance": -1},
        {"tolerance": np.inf},
        {"tolerance": True},
        {"reference_length": 0},
        {"reference_length": np.nan},
        {"plane": None},
    ],
)
def test_invalid_solver_parameters_are_rejected(options):
    with pytest.raises((TypeError, ValueError)):
        api("RigidSpherePlaneSolver")(shapes(), **options)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_multi_step_resting_contact_preserves_spin_and_has_no_normal_jitter(dtype):
    _, state, inputs = scene(dtype=dtype)
    model = RigidBodyModel(np.ones(1, dtype), np.ones((1, 3), dtype))
    state.velocity[:] = 0
    radius = dtype(0.2)
    state.position[0, 1] = radius
    stepper = Stepper(api("RigidSpherePlaneSolver")(shapes((radius,), dtype)))
    for _ in range(100):
        result = stepper.step(model, state, inputs, 0.01)
        assert result.report.accepted
        state = result.state
        assert state.position[0, 1] == radius
        assert state.velocity[0, 1] == 0
    np.testing.assert_allclose(state.orientation, [[0, np.sin(1), 0, np.cos(1)]], atol=3e-6)


def test_contact_preserves_anisotropic_gyroscopic_and_torque_prediction():
    _, state, inputs = scene()
    model = RigidBodyModel(np.ones(1), np.array([[1.0, 2.0, 3.0]]), np.zeros(3))
    state.angular_velocity[:] = [1, 2, 3]
    inputs.torque[:] = [2, -1, 4]
    free = RigidSemiImplicitSolver().step(model, state, inputs, 0.1)
    contact = api("RigidSpherePlaneSolver")(shapes()).step(model, state, inputs, 0.1)
    assert contact.report.accepted
    assert contact.state.velocity[0, 1] == 0
    np.testing.assert_array_equal(contact.state.orientation, free.state.orientation)
    np.testing.assert_array_equal(contact.state.angular_velocity, free.state.angular_velocity)


def test_same_contact_fallback_uses_original_snapshot_after_a_failed_candidate():
    model, state, inputs = scene()
    cls = api("RigidSpherePlaneSolver")

    class RejectedCandidate(cls):
        def step(self, model, working, working_inputs, dt):
            working.position[:] = 999
            working_inputs.torque[:] = 888
            return phymeo.StepResult(working, phymeo.StepReport(accepted=False, reason="proposal"))

    result = Stepper(RejectedCandidate(shapes()), cls(shapes())).step(model, state, inputs, 0.1)
    assert result.report.accepted and result.report.fallback_used
    assert result.report.primary_failure == "proposal"
    np.testing.assert_allclose(result.state.position, [[0.5, 0.2, 0.3]])
    np.testing.assert_allclose(result.state.velocity, [[5, 0, 3]])
    np.testing.assert_array_equal(state.position, [[0, 1, 0]])
    np.testing.assert_array_equal(inputs.torque, [[0, 0, 0]])


@pytest.mark.parametrize(
    "options", [{"reference_length": np.float32(1)}, {"tolerance": np.float32(0)}]
)
def test_numpy_scalar_settings_do_not_round_away_small_fp64_penetration(options):
    _, state, inputs = scene()
    model = RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3))
    state.position[:] = 0
    settings = {"tolerance": 0.0, **options}
    result = api("RigidSpherePlaneSolver")(shapes((1e-48,)), **settings).step(
        model, state, inputs, 0.1
    )
    assert not result.report.accepted
    assert result.report.max_residual == pytest.approx(1e-48, rel=1e-10, abs=0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fixed", [False, True])
def test_large_radius_cannot_cancel_a_small_plane_offset(dtype, fixed):
    _, state, inputs = scene(dtype=dtype)
    radius = dtype(1e16)
    state.position[:] = 0
    state.position[0, 1] = radius
    state.velocity[:] = state.angular_velocity[:] = 0
    model = RigidBodyModel(
        np.array([0 if fixed else 1], dtype), np.ones((1, 3), dtype), np.zeros(3)
    )
    result = api("RigidSpherePlaneSolver")(
        shapes((radius,), dtype), plane=Plane(height=0.6), tolerance=0
    ).step(model, state, inputs, 0.1)
    if fixed:
        assert not result.report.accepted
        assert result.report.max_constraint_error == pytest.approx(0.6)
    else:
        assert result.report.accepted
        # Difference from the represented radius is exact here, before removing the offset.
        assert (float(result.state.position[0, 1]) - float(radius)) - 0.6 >= 0
        assert result.state.position[0, 1] == np.nextafter(radius, dtype(np.inf))


def test_positive_penetration_normalization_cannot_underflow_to_zero_tolerance_success():
    _, state, inputs = scene()
    state.position[:] = state.velocity[:] = state.angular_velocity[:] = 0
    model = RigidBodyModel(np.zeros(1), np.ones((1, 3)), np.zeros(3))
    smallest = np.nextafter(0.0, 1.0)
    result = Stepper(
        api("RigidSpherePlaneSolver")(shapes((smallest,)), reference_length=2, tolerance=0)
    ).step(model, state, inputs, 0.1)
    assert not result.report.accepted
    assert result.report.max_constraint_error == smallest
    assert result.report.max_residual > 0
