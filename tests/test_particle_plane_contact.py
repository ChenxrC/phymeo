"""Independent dynamic plane laws, native time integration and outer rollback."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo import (
    DomainBinding,
    FreeSceneProgram,
    ParticleModel,
    ParticleState,
    SceneInputs,
    SceneModel,
    SceneState,
    SceneStepper,
    SolverFailure,
    StepInputs,
    StepSettings,
)
from phymeo.core.scene import HistoryField


def contact(*, domain="body", indices=None, normals=None, offsets=None, friction=None):
    import phymeo

    return phymeo.ParticlePlaneContacts(
        domain,
        np.array([0]) if indices is None else indices,
        np.array([[0.0, 1, 0]]) if normals is None else normals,
        np.zeros(1) if offsets is None else offsets,
        np.zeros(1) if friction is None else friction,
    )


def case(
    *,
    positions=None,
    velocities=None,
    inverse_mass=None,
    gravity=None,
    force=None,
    contacts=None,
    base=None,
    settings=None,
    inner_settings=None,
    relaxation=1.0,
):
    import phymeo

    positions = np.zeros((1, 3)) if positions is None else np.asarray(positions, dtype=float)
    count = len(positions)
    velocities = (
        np.zeros_like(positions) if velocities is None else np.asarray(velocities, dtype=float)
    )
    inverse_mass = np.ones(count) if inverse_mass is None else np.asarray(inverse_mass, dtype=float)
    model = ParticleModel(
        inverse_mass, np.zeros(3) if gravity is None else np.asarray(gravity, dtype=float)
    )
    scene = SceneModel((DomainBinding("body", "particles", model),))
    state = SceneState({"body": ParticleState(positions, velocities)})
    inputs = SceneInputs(
        {
            "body": StepInputs(
                np.zeros_like(positions) if force is None else np.asarray(force, dtype=float)
            )
        }
    )
    program = phymeo.ParticlePlaneContactProgram(
        scene,
        FreeSceneProgram() if base is None else base,
        (contact(),) if contacts is None else contacts,
        inner_settings=inner_settings,
        relaxation=relaxation,
    )
    runtime = SceneStepper(
        scene, program, StepSettings(substeps=1, tolerance=1e-10) if settings is None else settings
    )
    return runtime, state, inputs


def diagnostic(result, step=-1):
    return result.report.substeps[step].diagnostics["plane_contact"]


def test_public_api_owns_plane_arrays_and_normalizes_coefficients():
    normals, offsets = np.array([[0.0, 1 + 5e-13, 0]]), np.array([2.0])
    binding = contact(normals=normals, offsets=offsets)
    normals[:] = 7
    offsets[:] = 9
    np.testing.assert_allclose(binding.normals, [[0, 1, 0]], atol=1e-15)
    np.testing.assert_allclose(binding.offsets, [2 / (1 + 5e-13)], atol=1e-15)
    for name in ("normals", "offsets", "friction", "indices"):
        assert not getattr(binding, name).flags.writeable


@pytest.mark.parametrize(
    "kwargs",
    [
        {"indices": np.array([-0.5])},
        {"indices": np.array([-1])},
        {"indices": np.array([True])},
        {"normals": np.zeros((1, 3))},
        {"normals": np.array([[0.0, 2, 0]])},
        {"normals": np.ones((1, 2))},
        {"normals": np.array([[0.0, np.nan, 0]])},
        {"normals": np.array([[0.0, 1, 0]], np.float32)},
        {"offsets": np.array([np.inf])},
        {"offsets": np.ones(2)},
        {"friction": np.array([-0.1])},
        {"friction": np.ones(2)},
        {"domain": ""},
    ],
)
def test_invalid_contact_data_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        contact(**kwargs)


@pytest.mark.parametrize("binding", ["missing_domain", "out_of_range", "fixed"])
def test_binding_rejects_unsupported_nodes(binding):
    contacts = (
        (contact(domain="missing"),)
        if binding == "missing_domain"
        else (contact(indices=np.array([1 if binding == "out_of_range" else 0])),)
    )
    with pytest.raises((TypeError, ValueError)):
        case(contacts=contacts, inverse_mass=[0] if binding == "fixed" else [1])


@pytest.mark.parametrize("omega", [0, -0.1, 1.1, np.nan])
def test_relaxation_requires_positive_bounded_value(omega):
    with pytest.raises((TypeError, ValueError)):
        case(relaxation=omega)


def test_inner_time_interval_cannot_have_multiple_substeps():
    with pytest.raises(ValueError, match="substep"):
        case(inner_settings=StepSettings(substeps=2))


def test_separated_particle_follows_native_free_motion_without_contact_load():
    runtime, state, inputs = case(positions=[[0, 2, 0]], gravity=[0, -1, 0])
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.domains["body"].position, [[0, 1.99, 0]])
    np.testing.assert_allclose(result.state.domains["body"].velocity, [[0, -0.1, 0]])
    np.testing.assert_array_equal(diagnostic(result)["world_impulses"], np.zeros((1, 3)))


def test_approach_uses_position_level_discrete_contact_then_stops_next_step():
    runtime, state, inputs = case(positions=[[0, 0.01, 0]], velocities=[[0, -1, 0]])
    with runtime:
        first = runtime.step(state, inputs, 0.02)
        assert first.report.accepted
        second = runtime.step(first.state, inputs, 0.02)
    assert second.report.accepted
    np.testing.assert_allclose(first.state.domains["body"].position, [[0, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(first.state.domains["body"].velocity, [[0, -0.5, 0]], atol=1e-14)
    np.testing.assert_allclose(diagnostic(first)["world_impulses"], [[0, 0.5, 0]], atol=1e-14)
    np.testing.assert_allclose(second.state.domains["body"].velocity, [[0, 0, 0]], atol=1e-14)
    assert diagnostic(first)["contact_work_j"] == pytest.approx(-0.25)
    assert 0.5 * np.sum(first.state.domains["body"].velocity ** 2) == pytest.approx(0.125)


def test_static_gravity_support_computes_impulse_without_caller_supplied_contact_load():
    runtime, state, inputs = case(gravity=[0, -9.81, 0])
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.domains["body"].position, np.zeros((1, 3)), atol=1e-14)
    np.testing.assert_allclose(result.state.domains["body"].velocity, np.zeros((1, 3)), atol=1e-14)
    np.testing.assert_allclose(diagnostic(result)["world_impulses"], [[0, 0.0981, 0]], atol=1e-14)
    np.testing.assert_allclose(
        diagnostic(result)["environment_impulses"], [[0, -0.0981, 0]], atol=1e-14
    )
    np.testing.assert_allclose(diagnostic(result)["world_forces"], [[0, 9.81, 0]], atol=1e-12)
    np.testing.assert_array_equal(inputs.domains["body"].force, np.zeros((1, 3)))


@pytest.mark.parametrize("mu, acceleration", [(0.6, 0.0), (0.2, -5.81)])
def test_friction_couples_support_and_tangential_motion(mu, acceleration):
    binding = contact(normals=np.array([[1.0, 0, 0]]), friction=np.array([mu]))
    runtime, state, inputs = case(gravity=[0, -9.81, 0], force=[[-20, 0, 0]], contacts=(binding,))
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert result.report.accepted
    np.testing.assert_allclose(
        result.state.domains["body"].velocity, [[0, acceleration * 0.01, 0]], atol=1e-13
    )
    np.testing.assert_allclose(
        result.state.domains["body"].position, [[0, acceleration * 0.0001, 0]], atol=1e-13
    )
    np.testing.assert_allclose(
        diagnostic(result)["world_impulses"],
        [[0.2, (0.0981 if mu == 0.6 else 0.04), 0]],
        atol=1e-13,
    )
    assert diagnostic(result)["contact_work_j"] <= 1e-14


def test_same_node_two_planes_aggregate_load_once():
    binding = contact(
        indices=np.array([0, 0]),
        normals=np.array([[1.0, 0, 0], [0, 1.0, 0]]),
        offsets=np.zeros(2),
        friction=np.zeros(2),
    )
    runtime, state, inputs = case(force=[[-2, -3, 0]], contacts=(binding,))
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted
    np.testing.assert_allclose(result.state.domains["body"].velocity, [[0, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(
        diagnostic(result)["world_impulses"], [[0.2, 0, 0], [0, 0.3, 0]], atol=1e-14
    )
    np.testing.assert_allclose(
        diagnostic(result)["input_integrals"]["body"]["force"], [[0.2, 0.3, 0]], atol=1e-14
    )


def test_separating_contact_has_zero_impulse():
    runtime, state, inputs = case(velocities=[[0, 1, 0]])
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted
    np.testing.assert_array_equal(diagnostic(result)["world_impulses"], np.zeros((1, 3)))
    np.testing.assert_allclose(result.state.domains["body"].position, [[0, 0.1, 0]])


def test_initial_penetration_is_rejected_without_unaccounted_position_fix():
    runtime, state, inputs = case(positions=[[0, -0.01, 0]])
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted
    np.testing.assert_array_equal(
        result.state.domains["body"].position, state.domains["body"].position
    )


def test_insufficient_outer_iteration_budget_rolls_back_entire_frame():
    runtime, state, inputs = case(
        gravity=[0, -1, 0],
        relaxation=0.5,
        settings=StepSettings(substeps=2, position_iterations=1, tolerance=1e-10),
    )
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted
    np.testing.assert_array_equal(
        result.state.domains["body"].velocity, state.domains["body"].velocity
    )
    np.testing.assert_array_equal(
        result.state.domains["body"].position, state.domains["body"].position
    )


class HistoryFree(FreeSceneProgram):
    history_fields = (HistoryField("count", (1,), np.int64),)
    capabilities = replace(FreeSceneProgram.capabilities, history="explicit")

    def __init__(self, failure=None):
        self.failure = failure
        self.factory_closes = self.workspace_closes = self.workspaces = 0

    def create_workspace(self, scene, state, inputs, h, settings, context):
        native = super().create_workspace(scene, state, inputs, h, settings, context)
        count = state.history["count"].copy()
        owner = self
        self.workspaces += 1

        class Workspace:
            def __getattr__(self, name):
                return getattr(native, name)

            def predict(self):
                if owner.failure == "predict" and count[0] == 1:
                    raise SolverFailure("last-substep native prediction")
                native.predict()

            def result(self):
                return SceneState(native.result().domains, {"count": count + 1})

            def close(self):
                owner.workspace_closes += 1
                native.close()
                if owner.failure == "close" and count[0] == 1:
                    raise SolverFailure("last-substep native close")

        return Workspace()

    def close(self):
        self.factory_closes += 1


def test_native_history_advances_once_per_outer_substep_and_factory_closes_once():
    base = HistoryFree()
    runtime, state, inputs = case(
        base=base, gravity=[0, -1, 0], settings=StepSettings(substeps=2, tolerance=1e-10)
    )
    state = SceneState(state.domains, {"count": np.zeros(1, np.int64)})
    result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted
    np.testing.assert_array_equal(result.state.history["count"], [2])
    assert base.workspaces == base.workspace_closes
    assert base.factory_closes == 0
    runtime.close()
    runtime.close()
    assert base.factory_closes == 1


@pytest.mark.parametrize("failure", ["predict", "close"])
def test_late_substep_failure_restores_history_and_retry_matches_fresh(failure):
    base = HistoryFree(failure)
    runtime, state, inputs = case(
        base=base, gravity=[0, -1, 0], settings=StepSettings(substeps=2, tolerance=1e-10)
    )
    state = SceneState(state.domains, {"count": np.zeros(1, np.int64)})
    with runtime:
        failed = runtime.step(state, inputs, 0.1)
        assert not failed.report.accepted
        np.testing.assert_array_equal(failed.state.history["count"], [0])
        np.testing.assert_array_equal(
            failed.state.domains["body"].position, state.domains["body"].position
        )
        assert base.workspaces == base.workspace_closes
        base.failure = None
        retried = runtime.step(state, inputs, 0.1)
    fresh, initial, force = case(
        base=HistoryFree(), gravity=[0, -1, 0], settings=StepSettings(substeps=2, tolerance=1e-10)
    )
    initial = SceneState(initial.domains, {"count": np.zeros(1, np.int64)})
    with fresh:
        expected = fresh.step(initial, force, 0.1)
    assert retried.report.accepted and expected.report.accepted
    np.testing.assert_array_equal(retried.state.history["count"], expected.state.history["count"])
    np.testing.assert_array_equal(
        retried.state.domains["body"].position, expected.state.domains["body"].position
    )


def test_unrepresentable_actual_displacement_does_not_fake_support():
    runtime, state, inputs = case(
        positions=[[0, 1e16, 0]],
        gravity=[0, -1, 0],
        contacts=(contact(offsets=np.array([1e16])),),
        settings=StepSettings(substeps=1, position_iterations=2, tolerance=1e-10),
    )
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["body"].velocity, [[0, 0, 0]])


def test_bound_contact_change_rejected_before_trial():
    binding = contact()
    runtime, state, inputs = case(contacts=(binding,))
    binding.offsets.flags.writeable = True
    binding.offsets[:] = 1
    with runtime, pytest.raises(ValueError, match="bound"):
        runtime.step(state, inputs, 0.1)


def test_absolute_actual_load_gate_is_stricter_than_relative_replay_budget():
    from phymeo import SceneLoadReplay

    force = -2273179069.627076
    h = 0.056872685527390325
    p = 129281798.3743464
    model = ParticleModel(np.ones(1), np.zeros(3))
    scene = SceneModel((DomainBinding("body", "particles", model),))
    initial = SceneState({"body": ParticleState(np.zeros((1, 3)), np.zeros((1, 3)))})
    inputs = SceneInputs({"body": StepInputs(np.array([[0.0, force, 0.0]]))})
    with SceneLoadReplay(
        scene, FreeSceneProgram(), initial, inputs, h, channels={"body": ("force",)}
    ) as replay:
        trial = replay.evaluate({"body": {"force": np.array([[0.0, p, 0.0]])}})
    assert trial.result.report.accepted
    assert abs(trial.integrated_loads["body"]["force"][0, 1] - p) > 1e-10
    runtime, state, inputs = case(force=[[0, force, 0]])
    with runtime:
        result = runtime.step(state, inputs, h)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["body"].position, np.zeros((1, 3)))


@pytest.mark.parametrize(
    "field,value", [("tolerance", 1e-5), ("reference_length", 2.0), ("reference_speed", 2.0)]
)
def test_inner_physical_tolerances_cannot_silently_differ(field, value):
    outer = StepSettings(substeps=1, tolerance=1e-10)
    runtime, state, inputs = case(settings=outer, inner_settings=replace(outer, **{field: value}))
    with runtime, pytest.raises(ValueError, match="tolerance|reference|scale"):
        runtime.step(state, inputs, 0.1)


def test_tangent_frame_keeps_the_stored_plane_equation_exact():
    from fractions import Fraction

    normal = np.array([[0.7870960817250996, -0.034878532383426676, -0.6158435240479301]])
    binding = contact(normals=normal)
    runtime, state, inputs = case(positions=[[1e8, 0, 0]], contacts=(binding,))
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert result.report.accepted
    expected = float(Fraction(float(binding.normals[0, 0])) * Fraction(100000000))
    assert diagnostic(result)["gaps"][0] == expected


@pytest.mark.parametrize("h", [0.01, 0.005])
@pytest.mark.parametrize("mu, acceleration", [(0.6, 0.0), (0.2, -5.81)])
def test_two_second_static_and_sliding_windows_match_independent_discrete_solution(
    h, mu, acceleration
):
    binding = contact(normals=np.array([[1.0, 0, 0]]), friction=np.array([mu]))
    runtime, state, inputs = case(gravity=[0, -9.81, 0], force=[[-20, 0, 0]], contacts=(binding,))
    count = round(2 / h)
    with runtime:
        for step in range(1, count + 1):
            result = runtime.step(state, inputs, h)
            assert result.report.accepted, result.report.reason
            state = result.state
            np.testing.assert_allclose(
                state.domains["body"].velocity, [[0, acceleration * step * h, 0]], atol=3e-10
            )
            np.testing.assert_allclose(
                state.domains["body"].position,
                [[0, acceleration * h * h * step * (step + 1) / 2, 0]],
                atol=3e-10,
            )
            impulse = diagnostic(result)["world_impulses"][0]
            assert impulse[0] == pytest.approx(20 * h, abs=1e-10)
            assert diagnostic(result)["contact_work_j"] <= 1e-12
    # The known first-order time error is checked, not mistaken for an exact-time trajectory.
    continuous = 0.5 * acceleration * 2**2
    expected_error = abs(acceleration) * 2 * h / 2
    assert abs(state.domains["body"].position[0, 1] - continuous) == pytest.approx(
        expected_error, abs=3e-10
    )


def test_two_dynamic_nodes_material_and_contact_satisfy_independent_equilibrium():
    import phymeo
    from phymeo import ClothProgram, DistanceConstraints

    h = 0.01
    model = ParticleModel(np.ones(2), np.array([0.0, -9.81, 0]))
    constraints = DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([0.001]))
    native = ClothProgram(model, constraints, domain="body")
    state = SceneState(
        {"body": ParticleState(np.array([[0.0, 0, 0], [0, 1.0, 0]]), np.zeros((2, 3)))}
    )
    inputs = SceneInputs({"body": StepInputs.zeros(2)})
    program = phymeo.ParticlePlaneContactProgram(native.scene, native, (contact(),))
    with SceneStepper(native.scene, program, StepSettings(substeps=1, tolerance=1e-10)) as runtime:
        result = runtime.step(state, inputs, h)
    assert result.report.accepted, result.report.reason
    alpha = 0.001 / h**2
    multiplier = 9.81 * h * h / (1 + alpha)
    upper = 1 - 9.81 * h * h + multiplier
    contact_impulse = 9.81 * h * (2 + alpha) / (1 + alpha)
    actual = result.state.domains["body"]
    np.testing.assert_allclose(actual.position, [[0, 0, 0], [0, upper, 0]], atol=2e-12)
    np.testing.assert_allclose(actual.velocity, [[0, 0, 0], [0, (upper - 1) / h, 0]], atol=2e-10)
    diag = diagnostic(result)
    np.testing.assert_allclose(diag["world_impulses"], [[0, contact_impulse, 0]], atol=2e-10)
    native_diag = diag["inner_report"].substeps[0].diagnostics
    np.testing.assert_allclose(native_diag["distance_multipliers"], [multiplier], atol=2e-12)
    # Full two-node momentum: gravity and plane impulse appear once, internal forces cancel.
    np.testing.assert_allclose(
        np.sum(actual.velocity, axis=0), [0, contact_impulse - 2 * 9.81 * h, 0], atol=2e-10
    )
    energy = (
        0.5 * np.sum(actual.velocity**2)
        + 9.81 * np.sum(actual.position[:, 1])
        + 0.5 * (np.linalg.norm(actual.position[1] - actual.position[0]) - 1) ** 2 / 0.001
    )
    assert energy <= 9.81 + 1e-10


def test_returned_state_and_contact_ledger_cannot_pollute_a_fresh_trial():
    runtime, state, inputs = case(gravity=[0, -1, 0])
    with runtime:
        first = runtime.step(state, inputs, 0.1)
        assert first.report.accepted
        first.state.domains["body"].position[:] = 100
        impulses = diagnostic(first)["world_impulses"]
        impulses.flags.writeable = True
        impulses[:] = 100
        second = runtime.step(state, inputs, 0.1)
    assert second.report.accepted
    np.testing.assert_allclose(second.state.domains["body"].position, [[0, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(diagnostic(second)["world_impulses"], [[0, 0.1, 0]], atol=1e-14)


def test_inner_velocity_postprocessing_incompatible_with_displacement_is_rejected():
    class WrongVelocity(FreeSceneProgram):
        def create_workspace(self, *args):
            native = super().create_workspace(*args)
            original = native.result

            def result():
                state = original()
                state.domains["body"].velocity[:, 1] += 0.1
                return state

            native.result = result
            return native

    runtime, state, inputs = case(
        base=WrongVelocity(),
        settings=StepSettings(substeps=1, position_iterations=2, tolerance=1e-10),
    )
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["body"].velocity, [[0, 0, 0]])
