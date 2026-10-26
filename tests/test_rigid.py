"""Rigid dynamics checked with independent principal-axis and frame solutions."""

import numpy as np
import pytest

import phymeo


def api(name):
    result = getattr(phymeo, name, None)
    assert result is not None, f"{name} is not implemented"
    return result


def state(dtype=np.float64, *, orientation=None, angular_velocity=None):
    return api("RigidBodyState")(
        np.zeros((1, 3), dtype=dtype),
        np.array([[0.0, 0.0, 0.0, 1.0]] if orientation is None else orientation, dtype=dtype),
        np.zeros((1, 3), dtype=dtype),
        np.array([[0.0, 0.0, 0.0]] if angular_velocity is None else angular_velocity, dtype=dtype),
    )


def model(dtype=np.float64, inertia=(1.0, 1.0, 1.0)):
    return api("RigidBodyModel")(
        np.ones(1, dtype=dtype), np.array([inertia], dtype=dtype), np.zeros(3)
    )


@pytest.mark.parametrize("dtype, atol", [(np.float32, 3e-6), (np.float64, 1e-13)])
def test_spherical_body_constant_angular_velocity_matches_exact_rotation(dtype, atol):
    current = state(dtype, angular_velocity=[[0.0, 0.0, 2.0]])
    initial = current.copy()
    solver = api("RigidSemiImplicitSolver")()
    body = model(dtype, inertia=(2.0, 2.0, 2.0))
    inputs = api("RigidBodyInputs").zeros(1, dtype=dtype)
    for _ in range(100):
        result = solver.step(body, current, inputs, 0.01)
        assert result.report.accepted
        current = result.state
    np.testing.assert_allclose(
        current.orientation, [[0.0, 0.0, np.sin(1.0), np.cos(1.0)]], atol=atol
    )
    np.testing.assert_allclose(current.angular_velocity, [[0.0, 0.0, 2.0]], atol=atol)
    np.testing.assert_allclose(np.linalg.norm(current.orientation, axis=1), 1.0, atol=atol)
    assert current.orientation.dtype == dtype
    np.testing.assert_array_equal(initial.orientation, [[0.0, 0.0, 0.0, 1.0]])


def test_principal_axis_torque_uses_new_angular_velocity_for_orientation():
    body = model(inertia=(1.0, 1.0, 2.0))
    current = state()
    inputs = api("RigidBodyInputs")(np.zeros((1, 3)), np.array([[0.0, 0.0, 4.0]]))
    solver = api("RigidSemiImplicitSolver")()
    for _ in range(10):
        result = solver.step(body, current, inputs, 0.1)
        assert result.report.accepted
        current = result.state
    # alpha_z=2, omega_z=2; discrete angle is 2*0.1^2*10*11/2 = 1.1.
    np.testing.assert_allclose(current.angular_velocity, [[0.0, 0.0, 2.0]], atol=1e-14)
    np.testing.assert_allclose(
        current.orientation, [[0.0, 0.0, np.sin(0.55), np.cos(0.55)]], atol=1e-14
    )


def test_world_torque_is_transformed_to_body_principal_inertia():
    s = np.sqrt(0.5)
    current = state(orientation=[[0.0, 0.0, s, s]])
    # The world x axis corresponds to body -y after a 90-degree rotation about z.
    inputs = api("RigidBodyInputs")(np.zeros((1, 3)), np.array([[2.0, 0.0, 0.0]]))
    result = api("RigidSemiImplicitSolver")().step(
        model(inertia=(1.0, 2.0, 3.0)), current, inputs, 0.1
    )
    assert result.report.accepted
    np.testing.assert_allclose(result.state.angular_velocity, [[0.1, 0.0, 0.0]], atol=1e-14)
    # Left multiplication by a world-x rotation of 0.01 rad; y component is negative.
    a, b = np.sin(0.005), np.cos(0.005)
    np.testing.assert_allclose(
        result.state.orientation, [[a * s, -a * s, b * s, b * s]], atol=1e-14
    )


def test_anisotropic_body_includes_gyroscopic_angular_acceleration():
    current = state(angular_velocity=[[1.0, 2.0, 3.0]])
    result = api("RigidSemiImplicitSolver")().step(
        model(inertia=(1.0, 2.0, 3.0)), current, api("RigidBodyInputs").zeros(1), 0.01
    )
    assert result.report.accepted
    # omega x (I omega) = [6,-6,2], so alpha = [-6,3,-2/3].
    np.testing.assert_allclose(
        result.state.angular_velocity, [[0.94, 2.03, 3.0 - 0.02 / 3]], atol=1e-14
    )
    np.testing.assert_array_equal(current.angular_velocity, [[1.0, 2.0, 3.0]])


def test_fixed_body_preserves_pose_and_zeroes_both_velocities():
    body = api("RigidBodyModel")(np.zeros(1), np.zeros((1, 3)))
    current = state(orientation=[[0.0, 0.0, 1.0, 0.0]], angular_velocity=[[1.0, 2.0, 3.0]])
    current.position[:] = [[1.0, 2.0, 3.0]]
    current.velocity[:] = 5.0
    inputs = api("RigidBodyInputs")(np.full((1, 3), 100.0), np.full((1, 3), 100.0))
    result = api("RigidSemiImplicitSolver")().step(body, current, inputs, 0.1)
    assert result.report.accepted
    np.testing.assert_array_equal(result.state.position, current.position)
    np.testing.assert_array_equal(result.state.orientation, current.orientation)
    np.testing.assert_array_equal(result.state.velocity, np.zeros((1, 3)))
    np.testing.assert_array_equal(result.state.angular_velocity, np.zeros((1, 3)))


def test_rigid_translation_matches_force_and_gravity_discrete_solution():
    body = api("RigidBodyModel")(np.array([0.5]), np.ones((1, 3)), np.array([0.0, -10.0, 0.0]))
    inputs = api("RigidBodyInputs")(np.array([[2.0, 0.0, 0.0]]), np.zeros((1, 3)))
    result = api("RigidSemiImplicitSolver")().step(body, state(), inputs, 0.25)
    np.testing.assert_allclose(result.state.velocity, [[0.25, -2.5, 0.0]])
    np.testing.assert_allclose(result.state.position, [[0.0625, -0.625, 0.0]])


def test_tiny_rotations_are_not_discarded():
    result = api("RigidSemiImplicitSolver")().step(
        model(), state(angular_velocity=[[1e-20, 0.0, 0.0]]), api("RigidBodyInputs").zeros(1), 0.1
    )
    assert result.report.accepted
    assert result.state.orientation[0, 0] == pytest.approx(5e-22, rel=1e-12, abs=0.0)


def test_empty_rigid_system_is_supported():
    body = api("RigidBodyModel")(np.empty(0), np.empty((0, 3)))
    current = api("RigidBodyState")(
        np.empty((0, 3)), np.empty((0, 4)), np.empty((0, 3)), np.empty((0, 3))
    )
    result = api("RigidSemiImplicitSolver")().step(
        body, current, api("RigidBodyInputs").zeros(0), 0.1
    )
    assert result.report.accepted
    assert result.state.orientation.shape == (0, 4)


def test_rigid_state_and_input_buffers_are_owned():
    current = state()
    copied = current.copy()
    copied.orientation[:] = [[0.0, 0.0, 1.0, 0.0]]
    copied.angular_velocity[:] = 1.0
    np.testing.assert_array_equal(current.orientation, [[0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(current.angular_velocity, np.zeros((1, 3)))
    inputs = api("RigidBodyInputs").zeros(1)
    assert not np.shares_memory(inputs.force, inputs.torque)
    result = api("RigidSemiImplicitSolver")().step(model(), current, inputs, 0.1)
    result.state.orientation[:] = [[0.0, 0.0, 1.0, 0.0]]
    np.testing.assert_array_equal(current.orientation, [[0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(inputs.torque, np.zeros((1, 3)))


def test_model_inertia_is_owned_and_read_only():
    inertia = np.ones((1, 3))
    body = api("RigidBodyModel")(np.ones(1), inertia)
    inertia[:] = 9.0
    np.testing.assert_array_equal(body.inertia_diagonal, np.ones((1, 3)))
    assert not body.inertia_diagonal.flags.writeable
    assert not body.inverse_mass.flags.writeable
    assert body.body_count == 1


@pytest.mark.parametrize(
    "inertia",
    [
        [[0.0, 1.0, 1.0]],
        [[-1.0, 1.0, 1.0]],
        [[1.0, 1.0, 3.0]],
        [[np.nan, 1.0, 1.0]],
        [[1.0, np.inf, 1.0]],
        [[1.0, 1.0]],
    ],
)
def test_invalid_dynamic_inertia_is_rejected(inertia):
    constructor = api("RigidBodyModel")
    with pytest.raises(ValueError):
        constructor(np.ones(1), np.array(inertia))


@pytest.mark.parametrize(
    "quaternion",
    [[[0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 2.0]], [[0.0, 0.0, 0.0]], [[np.nan, 0.0, 0.0, 1.0]]],
)
def test_invalid_orientation_is_rejected(quaternion):
    api("RigidBodyState")
    with pytest.raises(ValueError):
        state(orientation=quaternion)


@pytest.mark.parametrize("bad", ["orientation", "angular_velocity", "torque"])
def test_mutated_nonfinite_rigid_input_is_revalidated(bad):
    current, inputs = state(), api("RigidBodyInputs").zeros(1)
    if bad == "torque":
        inputs.torque[0, 0] = np.nan
    else:
        getattr(current, bad)[0, 0] = np.nan
    with pytest.raises(ValueError):
        api("RigidSemiImplicitSolver")().step(model(), current, inputs, 0.1)


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf, True])
def test_invalid_rigid_time_step_is_rejected(dt):
    solver = api("RigidSemiImplicitSolver")()
    with pytest.raises(ValueError):
        solver.step(model(), state(), api("RigidBodyInputs").zeros(1), dt)


def test_rotation_overflow_is_rejected_without_altering_input():
    current = state(angular_velocity=[[1e308, 0.0, 0.0]])
    result = api("RigidSemiImplicitSolver")().step(
        model(), current, api("RigidBodyInputs").zeros(1), 10.0
    )
    assert not result.report.accepted
    assert result.report.reason
    np.testing.assert_array_equal(result.state.orientation, current.orientation)
    np.testing.assert_array_equal(result.state.angular_velocity, current.angular_velocity)


def test_stepper_accepts_rigid_capability_with_explicit_rigid_inputs():
    solver = api("RigidSemiImplicitSolver")()
    assert solver.capabilities.state_kind == "rigid_bodies"
    result = phymeo.Stepper(solver).step(model(), state(), api("RigidBodyInputs").zeros(1), 0.1)
    assert result.report.accepted


def test_near_unit_input_is_interpreted_as_the_normalized_rotation():
    s = np.sqrt(0.5) * (1 + 4e-7)
    current = state(orientation=[[0.0, 0.0, s, s]])
    result = api("RigidSemiImplicitSolver")().step(
        model(inertia=(1.0, 2.0, 3.0)),
        current,
        api("RigidBodyInputs")(np.zeros((1, 3)), np.array([[2.0, 0.0, 0.0]])),
        0.1,
    )
    assert result.report.accepted
    np.testing.assert_allclose(result.state.angular_velocity, [[0.1, 0.0, 0.0]], atol=1e-14)


def test_torque_free_anisotropic_energy_error_converges_with_time_step():
    def drift(dt, steps):
        body = model(inertia=(1.0, 2.0, 3.0))
        current = state(angular_velocity=[[1.0, 2.0, 3.0]])
        solver, inputs = api("RigidSemiImplicitSolver")(), api("RigidBodyInputs").zeros(1)
        for _ in range(steps):
            result = solver.step(body, current, inputs, dt)
            assert result.report.accepted
            current = result.state
        # Inverse-quaternion vector formula is independent of the solver's matrix path.
        vector, scalar = -current.orientation[0, :3], current.orientation[0, 3]
        omega = current.angular_velocity[0]
        body_omega = omega + 2 * np.cross(vector, np.cross(vector, omega) + scalar * omega)
        energy = 0.5 * np.dot(np.array([1.0, 2.0, 3.0]), body_omega**2)
        return abs(energy - 18.0)

    coarse, fine = drift(0.004, 250), drift(0.002, 500)
    assert coarse > 0.0
    assert 0.35 < fine / coarse < 0.65
