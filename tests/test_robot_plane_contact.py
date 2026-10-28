"""Independent robot/plane mechanics and complete replay transaction checks."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo import (
    RobotInputs,
    RobotModel,
    RobotProgram,
    RobotState,
    SceneInputs,
    SceneState,
    SceneStepper,
    StepSettings,
)


def slider():
    return RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0, 0, 1]]),
        np.array([[0.0, 1, 0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.array([0.0, -9.81, 0]),
    )


def rod():
    return replace(
        slider(),
        joint_types=("revolute",),
        joint_axis=np.array([[0.0, 0, 1]]),
        mass=np.ones(1),
        com=np.array([[0.0, -0.5, 0]]),
        inertia_diagonal=np.full((1, 3), 1 / 12),
        gravity=np.zeros(3),
    )


def contact(**kwargs):
    import phymeo

    values = dict(
        domain="robot",
        indices=np.array([0]),
        local_points=np.zeros((1, 3)),
        normals=np.array([[0.0, 1, 0]]),
        offsets=np.zeros(1),
        friction=np.zeros(1),
    )
    values.update(kwargs)
    return phymeo.RobotPlaneContacts(**values)


def case(
    *, model=None, q=None, v=None, effort=None, contacts=None, base=None, settings=None, **kwargs
):
    import phymeo

    model = slider() if model is None else model
    native = RobotProgram(model) if base is None else base
    n = model.dof_count
    state = SceneState(
        {
            "robot": RobotState(
                np.zeros(n) if q is None else np.array(q, float),
                np.zeros(n) if v is None else np.array(v, float),
            )
        }
    )
    original = RobotInputs.zeros(n)
    if effort is not None:
        original.effort[:] = effort
    inputs = SceneInputs({"robot": original})
    program = phymeo.RobotPlaneContactProgram(
        native.scene, native, (contact(),) if contacts is None else contacts, **kwargs
    )
    runtime = SceneStepper(
        native.scene,
        program,
        StepSettings(substeps=1, tolerance=1e-10) if settings is None else settings,
        registry=program.registry,
    )
    return runtime, state, inputs


def diag(result):
    return result.report.substeps[-1].diagnostics["robot_plane_contact"]


def rotating_case(**kwargs):
    return case(
        model=rod(),
        q=[-0.3],
        v=[1.0],
        contacts=(
            contact(local_points=np.array([[0.0, -1, 0]]), offsets=np.array([-np.cos(0.2)])),
        ),
        **kwargs,
    )


def test_binding_owns_normalizes_and_locks_arrays():
    points = np.array([[1.0, 2, 3]])
    binding = contact(
        local_points=points, normals=np.array([[0.0, 1 + 5e-13, 0]]), offsets=np.array([2.0])
    )
    points[:] = 0
    np.testing.assert_array_equal(binding.local_points, [[1.0, 2, 3]])
    assert binding.offsets[0] == 2 / (1 + 5e-13)
    for name in ("indices", "local_points", "normals", "offsets", "friction"):
        assert not getattr(binding, name).flags.writeable


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(indices=np.array([-1])),
        dict(indices=np.array([0.5])),
        dict(indices=np.array([True])),
        dict(local_points=np.ones((1, 2))),
        dict(local_points=np.zeros((1, 3), np.float32)),
        dict(normals=np.zeros((1, 3))),
        dict(normals=np.array([[0.0, 2, 0]])),
        dict(offsets=np.array([np.inf])),
        dict(friction=np.array([-0.1])),
        dict(domain=""),
    ],
)
def test_invalid_bindings_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        contact(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(reference_angular_speed=0),
        dict(reference_angular_speed=np.inf),
        dict(relaxation=0),
        dict(relaxation=1.1),
        dict(inner_settings=StepSettings(substeps=2)),
    ],
)
def test_invalid_program_settings_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        case(**kwargs)


@pytest.mark.parametrize("binding", [dict(domain="absent"), dict(indices=np.array([1]))])
def test_binding_must_name_robot_and_valid_link(binding):
    with pytest.raises((ValueError, TypeError)):
        case(contacts=(contact(**binding),))


def test_prismatic_support_computes_native_effort_once():
    runtime, state, inputs = case()
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert result.report.accepted, result.report.reason
    np.testing.assert_allclose(result.state.domains["robot"].q, [0], atol=1e-14)
    np.testing.assert_allclose(result.state.domains["robot"].v, [0], atol=1e-14)
    np.testing.assert_allclose(diag(result)["world_impulses"], [[0, 0.1962, 0]], atol=1e-14)
    np.testing.assert_allclose(diag(result)["generalized_integral"], [0.1962], atol=1e-14)
    np.testing.assert_allclose(diag(result)["environment_impulses"], [[0, -0.1962, 0]], atol=1e-14)
    np.testing.assert_array_equal(inputs.domains["robot"].effort, [0])


def test_separating_material_point_keeps_native_free_motion():
    runtime, state, inputs = case(model=replace(slider(), gravity=np.zeros(3)), q=[0.2], v=[0.3])
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert result.report.accepted, result.report.reason
    np.testing.assert_allclose(result.state.domains["robot"].q, [0.23], atol=1e-15)
    np.testing.assert_array_equal(diag(result)["world_impulses"], [[0, 0, 0]])


def test_rotating_joint_uses_final_jacobian_and_material_velocity():
    runtime, state, inputs = rotating_case()
    with runtime:
        result = runtime.step(state, inputs, 0.15)
    assert result.report.accepted, result.report.reason
    r = result.state.domains["robot"]
    d = diag(result)
    np.testing.assert_allclose(r.q, [-0.2], atol=2e-10)
    np.testing.assert_allclose(r.v, [2 / 3], atol=2e-9)
    expected = (1 / 3) * (2 / 3 - 1) / np.sin(-0.2)
    np.testing.assert_allclose(d["world_impulses"], [[0, expected, 0]], atol=2e-9)
    np.testing.assert_allclose(d["generalized_integral"], [-1 / 9], atol=2e-10)
    np.testing.assert_allclose(
        d["material_velocities"], [[np.cos(0.2) * 2 / 3, -np.sin(0.2) * 2 / 3, 0]], atol=2e-9
    )
    assert np.linalg.norm(d["material_velocities"] - d["chord_velocities"]) > 1e-2
    np.testing.assert_allclose(d["environment_moments"][0, 2], 1 / 9, atol=2e-9)
    assert d["contact_work_j"] < 0
    np.testing.assert_allclose(
        d["frozen_kinetic_change_from_free"], 0.5 / 3 * ((2 / 3) ** 2 - 1), atol=2e-9
    )


def test_initial_penetration_is_refused_without_repair():
    runtime, state, inputs = case(q=[-0.01])
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["robot"].q, [-0.01])


def test_outer_budget_failure_keeps_original_state():
    runtime, state, inputs = rotating_case(
        settings=StepSettings(substeps=1, tolerance=1e-10, position_iterations=1)
    )
    with runtime:
        result = runtime.step(state, inputs, 0.15)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["robot"].q, [-0.3])
    np.testing.assert_array_equal(result.state.domains["robot"].v, [1])


def test_unresponsive_point_cannot_hide_penetration():
    # A point on the fixed revolute axis cannot move to a higher plane.
    runtime, state, inputs = case(model=rod(), contacts=(contact(offsets=np.array([0.01])),))
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert not result.report.accepted


@pytest.mark.parametrize("mass", [1 / 3, 1e-20])
def test_large_velocity_cancellation_cannot_fake_robot_dynamics(mass):
    model = replace(slider(), mass=np.array([mass]), gravity=np.zeros(3))
    runtime, state, inputs = case(model=model, v=[-1e16])
    with runtime:
        result = runtime.step(state, inputs, 0.125)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["robot"].v, [-1e16])


def test_world_background_cannot_swallow_generalized_motion():
    runtime, state, inputs = case(model=replace(slider(), gravity=np.zeros(3)), q=[1e16], v=[1.0])
    with runtime:
        result = runtime.step(state, inputs, 0.1)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["robot"].q, [1e16])


def test_exactly_representable_large_robot_stop_remains_valid():
    runtime, state, inputs = case(
        model=replace(slider(), mass=np.array([3.0]), gravity=np.zeros(3)), v=[-1e16]
    )
    with runtime:
        result = runtime.step(state, inputs, 0.125)
    assert result.report.accepted, result.report.reason
    np.testing.assert_array_equal(result.state.domains["robot"].v, [0])
    np.testing.assert_array_equal(diag(result)["generalized_integral"], [3e16])


def test_audit_residual_rounding_cannot_hide_a_strict_budget_violation():
    from fractions import Fraction

    from phymeo.solvers._robot_replay_ledger import upper_float

    limit = 1e-10
    exact = Fraction(limit) + Fraction(1, 2**200)
    assert float(exact) == limit
    assert upper_float(exact) > limit


def cartesian():
    return RobotModel(
        np.array([-1, 0]),
        ("prismatic", "prismatic"),
        np.zeros((2, 3)),
        np.tile([0.0, 0, 0, 1], (2, 1)),
        np.array([[1.0, 0, 0], [0, 1.0, 0]]),
        np.ones(2),
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.array([0.0, -9.81, 0]),
    )


class HistoryRobot(RobotProgram):
    from phymeo import HistoryField

    history_fields = (HistoryField("count", (1,), np.int64),)
    capabilities = replace(RobotProgram.capabilities, history="explicit")

    def __init__(self, model, failure=None):
        super().__init__(model)
        self.failure = failure
        self.workspaces = self.workspace_closes = self.factory_closes = 0

    def create_workspace(self, scene, state, inputs, h, settings, context):
        from phymeo import SolverFailure

        native = super().create_workspace(scene, state, inputs, h, settings, context)
        count = state.history["count"].copy()
        owner = self
        self.workspaces += 1

        class Workspace:
            def __getattr__(self, name):
                return getattr(native, name)

            def predict(self):
                if owner.failure == "predict" and count[0] == 1:
                    raise SolverFailure("last native predict")
                native.predict()

            def result(self):
                return SceneState(native.result().domains, {"count": count + 1})

            def close(self):
                owner.workspace_closes += 1
                native.close()
                if owner.failure == "close" and count[0] == 1:
                    raise SolverFailure("last native close")

        return Workspace()

    def close(self):
        self.factory_closes += 1


def test_history_advances_once_and_borrowed_factory_closes_once():
    base = HistoryRobot(slider())
    runtime, state, inputs = case(base=base, settings=StepSettings(substeps=2, tolerance=1e-10))
    state = SceneState(state.domains, {"count": np.zeros(1, np.int64)})
    result = runtime.step(state, inputs, 0.02)
    assert result.report.accepted, result.report.reason
    np.testing.assert_array_equal(result.state.history["count"], [2])
    assert base.workspaces == base.workspace_closes
    assert base.factory_closes == 0
    runtime.close()
    runtime.close()
    assert base.factory_closes == 1


@pytest.mark.parametrize("failure", ["predict", "close"])
def test_last_substep_failure_rolls_back_history_and_retry_matches_fresh(failure):
    base = HistoryRobot(slider(), failure)
    runtime, state, inputs = case(base=base, settings=StepSettings(substeps=2, tolerance=1e-10))
    state = SceneState(state.domains, {"count": np.zeros(1, np.int64)})
    with runtime:
        failed = runtime.step(state, inputs, 0.02)
        assert not failed.report.accepted
        np.testing.assert_array_equal(failed.state.history["count"], [0])
        np.testing.assert_array_equal(failed.state.domains["robot"].q, [0])
        assert base.workspaces == base.workspace_closes
        base.failure = None
        retry = runtime.step(state, inputs, 0.02)
    fresh, initial, forces = case(
        base=HistoryRobot(slider()), settings=StepSettings(substeps=2, tolerance=1e-10)
    )
    initial = SceneState(initial.domains, {"count": np.zeros(1, np.int64)})
    with fresh:
        expected = fresh.step(initial, forces, 0.02)
    assert retry.report.accepted and expected.report.accepted
    np.testing.assert_array_equal(retry.state.domains["robot"].q, expected.state.domains["robot"].q)
    np.testing.assert_array_equal(retry.state.history["count"], expected.state.history["count"])


def test_returned_candidate_and_diagnostics_cannot_contaminate_replay():
    runtime, state, inputs = case()
    with runtime:
        first = runtime.step(state, inputs, 0.01)
        first.state.domains["robot"].q[:] = 9
        diag(first)["world_impulses"].flags.writeable = True
        diag(first)["world_impulses"][:] = 8
        second = runtime.step(state, inputs, 0.01)
    assert second.report.accepted
    np.testing.assert_array_equal(second.state.domains["robot"].q, [0])
    np.testing.assert_allclose(diag(second)["world_impulses"], [[0, 0.1962, 0]], atol=1e-14)


def test_incompatible_native_velocity_postprocessing_is_rejected():
    class BadProgram(RobotProgram):
        def create_workspace(self, *args):
            native = super().create_workspace(*args)

            class Workspace:
                def __getattr__(self, name):
                    return getattr(native, name)

                def prepare_velocity(self):
                    native.candidate.v += 0.1

            return Workspace()

    runtime, state, inputs = case(base=BadProgram(slider()))
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert not result.report.accepted
    np.testing.assert_array_equal(result.state.domains["robot"].v, [0])


def test_energy_gate_catches_small_velocity_error_on_large_background():
    class TinyEnergyError(RobotProgram):
        def create_workspace(self, *args):
            native = super().create_workspace(*args)

            class Workspace:
                def __getattr__(self, name):
                    return getattr(native, name)

                def predict(self):
                    native.predict()
                    if native.inputs.effort[1] != 0:
                        native.candidate.v[0] = np.nextafter(native.candidate.v[0], np.inf)
                        native.candidate.q = native.initial.q + native.h * native.candidate.v

            return Workspace()

    model = cartesian()
    runtime, state, inputs = case(
        model=model,
        base=TinyEnergyError(model),
        v=[1e7, 0],
        contacts=(contact(indices=np.array([1])),),
        settings=StepSettings(substeps=1, tolerance=1e-8, position_iterations=3),
    )
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert not result.report.accepted
    assert result.report.substeps[-1].position.residuals["frozen_energy"] > 1e-3
    assert result.report.substeps[-1].position.residuals["contact_work"] <= 1e-8
    np.testing.assert_array_equal(result.state.domains["robot"].v, [1e7, 0])


@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_two_rotating_links_share_ancestor_reaction_once(order):
    model = RobotModel(
        np.array([-1, 0]),
        ("revolute", "revolute"),
        np.array([[0.0, 0, 0], [0, -1.0, 0]]),
        np.tile([0.0, 0, 0, 1], (2, 1)),
        np.tile([0.0, 0, 1], (2, 1)),
        np.ones(2),
        np.tile([0.0, -0.5, 0], (2, 1)),
        np.full((2, 3), 1 / 12),
        np.array([0.0, -9.81, 0]),
    )
    q = np.array([-0.3, -0.4])
    offsets = np.array([-np.cos(0.3), -np.cos(0.3) - np.cos(0.7)])[order]
    binding = contact(
        indices=np.array([0, 1])[order],
        local_points=np.tile([0.0, -1, 0], (2, 1)),
        normals=np.tile([0.0, 1, 0], (2, 1)),
        offsets=offsets,
        friction=np.zeros(2),
    )
    runtime, state, inputs = case(model=model, q=q, contacts=(binding,))
    with runtime:
        result = runtime.step(state, inputs, 0.01)
    assert result.report.accepted, result.report.reason
    np.testing.assert_allclose(result.state.domains["robot"].q, q, atol=1e-10)
    np.testing.assert_allclose(result.state.domains["robot"].v, [0, 0], atol=1e-8)
    np.testing.assert_allclose(
        diag(result)["world_impulses"][:, 1], np.array([0.0981, 0.04905])[order], atol=2e-8
    )
    expected = 0.01 * 9.81 * np.array([1.5 * np.sin(-0.3) + 0.5 * np.sin(-0.7), 0.5 * np.sin(-0.7)])
    np.testing.assert_allclose(diag(result)["generalized_integral"], expected, atol=2e-8)


def test_rotational_and_prismatic_error_scales_have_distinct_units():
    from fractions import Fraction

    from phymeo.solvers._robot_replay_ledger import FrozenRobotLedger

    model = replace(
        cartesian(),
        joint_types=("revolute", "prismatic"),
        joint_axis=np.array([[0.0, 0, 1], [0, 1.0, 0]]),
    )
    ledger = FrozenRobotLedger(
        model,
        RobotState(np.zeros(2), np.zeros(2)),
        RobotInputs.zeros(2),
        0.01,
        StepSettings(reference_speed=2.0),
        4.0,
    )
    assert ledger.joint_error([Fraction(1), Fraction(0)]) == Fraction(1, 4)
    assert ledger.joint_error([Fraction(0), Fraction(1)]) == Fraction(1, 2)


@pytest.mark.parametrize(
    "field,value", [("tolerance", 1e-6), ("reference_length", 2.0), ("reference_speed", 2.0)]
)
def test_inner_physical_scales_cannot_be_silently_looser(field, value):
    settings = StepSettings(substeps=1, tolerance=1e-10)
    runtime, state, inputs = case(
        settings=settings, inner_settings=replace(settings, **{field: value})
    )
    with pytest.raises(ValueError, match="inner tolerance"):
        with runtime:
            runtime.step(state, inputs, 0.01)


@pytest.mark.parametrize("h", [0.01, 0.005])
@pytest.mark.parametrize("mu,acceleration", [(0.6, 0.0), (0.2, -5.81)])
def test_two_second_robot_static_and_sliding_windows(h, mu, acceleration):
    binding = contact(
        indices=np.array([1]), normals=np.array([[1.0, 0, 0]]), friction=np.array([mu])
    )
    runtime, state, inputs = case(model=cartesian(), effort=[-20.0, 0.0], contacts=(binding,))
    with runtime:
        for k in range(1, round(2 / h) + 1):
            result = runtime.step(state, inputs, h)
            assert result.report.accepted, (k, result.report.reason)
            state = result.state
            np.testing.assert_allclose(
                state.domains["robot"].q, [0, acceleration * h * h * k * (k + 1) / 2], atol=2e-9
            )
            np.testing.assert_allclose(
                state.domains["robot"].v, [0, acceleration * h * k], atol=2e-9
            )
            np.testing.assert_allclose(
                diag(result)["world_impulses"],
                [[20 * h, (9.81 if mu == 0.6 else 4) * h, 0]],
                atol=2e-9,
            )
    assert result.report.max_residual <= 1e-10


def test_robot_contact_preserves_native_cloth_material_side_domain():
    from test_tyx_robot_plane_contact import verify_material_side

    verify_material_side("numpy")


@pytest.mark.parametrize("h", [0.02, 0.01])
def test_two_second_rotating_contact_matches_scalar_stop(h):
    runtime, state, inputs = rotating_case()
    previous = -0.3
    previous_v = 1.0
    with runtime:
        for k in range(1, round(2 / h) + 1):
            q = min(-0.3 + k * h, -0.2)
            v = (q - previous) / h
            expected = (1 / 3) * (v - previous_v) / np.sin(q)
            result = runtime.step(state, inputs, h)
            assert result.report.accepted, (k, result.report.reason)
            state = result.state
            np.testing.assert_allclose(state.domains["robot"].q, [q], atol=2e-9)
            np.testing.assert_allclose(state.domains["robot"].v, [v], atol=2e-8)
            np.testing.assert_allclose(
                diag(result)["world_impulses"], [[0, expected, 0]], atol=3e-8
            )
            previous, previous_v = q, v


@pytest.mark.parametrize("origin", [1e16, -1e16])
def test_world_translation_cannot_erase_local_rotating_contact(origin):
    model = replace(
        rod(),
        joint_position=np.array([[origin, origin, 0.0]]),
        com=np.zeros((1, 3)),
        inertia_diagonal=np.ones((1, 3)),
    )
    binding = contact(local_points=np.array([[1.0, 0, 0]]), offsets=np.array([origin]))
    runtime, state, inputs = case(model=model, q=[0.25], v=[-1.0], contacts=(binding,))
    # Relative geometry is independent of the world translation: free gap flips sign.
    assert np.sin(0.25) > 0 > np.sin(-0.25)
    with runtime:
        result = runtime.step(state, inputs, 0.5)
    assert not result.report.accepted
    assert "geometry" in result.report.reason
    np.testing.assert_array_equal(result.state.domains["robot"].q, [0.25])
    np.testing.assert_array_equal(result.state.domains["robot"].v, [-1.0])


@pytest.mark.parametrize("origin", [1e16, -1e16])
def test_world_translation_cannot_erase_com_lever_in_empty_contact_program(origin):
    model = replace(rod(), joint_position=np.array([[origin, origin, 0.0]]))
    runtime, state, inputs = case(model=model, q=[0.25], v=[-1.0], contacts=())
    with runtime:
        result = runtime.step(state, inputs, 0.5)
    assert not result.report.accepted
    assert "geometry" in result.report.reason
    np.testing.assert_array_equal(result.state.domains["robot"].q, [0.25])


@pytest.mark.parametrize("origin", [0.0, 1.0, -1.0])
def test_resolved_translated_rotor_retains_physical_contact(origin):
    model = replace(
        rod(),
        joint_position=np.array([[origin, origin, 0.0]]),
        com=np.zeros((1, 3)),
        inertia_diagonal=np.ones((1, 3)),
    )
    binding = contact(local_points=np.array([[1.0, 0, 0]]), offsets=np.array([origin]))
    runtime, state, inputs = case(model=model, q=[0.25], v=[-1.0], contacts=(binding,))
    with runtime:
        result = runtime.step(state, inputs, 0.5)
    assert result.report.accepted, result.report.reason
    np.testing.assert_allclose(result.state.domains["robot"].q, [0], atol=1e-10)
    np.testing.assert_allclose(result.state.domains["robot"].v, [-0.5], atol=1e-10)
    np.testing.assert_allclose(diag(result)["world_impulses"], [[0, 0.5, 0]], atol=1e-10)


def test_candidate_entering_unresolved_world_geometry_rolls_back():
    model = replace(
        cartesian(),
        joint_types=("prismatic", "revolute"),
        joint_axis=np.array([[1.0, 0, 0], [0, 0, 1.0]]),
        gravity=np.zeros(3),
    )
    binding = contact(indices=np.array([1]), local_points=np.array([[1.0, 0, 0]]))
    runtime, state, inputs = case(model=model, v=[1e16, 0], contacts=(binding,))
    with runtime:
        result = runtime.step(state, inputs, 1.0)
    assert not result.report.accepted
    assert "geometry" in result.report.reason
    np.testing.assert_array_equal(result.state.domains["robot"].q, [0, 0])
    np.testing.assert_array_equal(result.state.domains["robot"].v, [1e16, 0])


def test_affine_guard_checks_the_sum_path_used_by_robot_plane_positions():
    from phymeo.dynamics.robot_kinematics import forward_kinematics

    model = replace(
        slider(),
        joint_orientation=np.array(
            [[-0.6936968752508276, -0.5423304147828774, 0.4539085407869848, 0.13648957128403663]]
        ),
        gravity=np.zeros(3),
    )
    local = np.array([[102011162.88264461, -29070837.842375822, -2703980.560245362]])
    rotation = forward_kinematics(model, np.zeros(1)).rotation[0]
    normal = rotation @ np.array([0.876333392445091, -0.37449663626426416, -0.3029720361888696])
    normal /= np.linalg.norm(normal)
    position = np.sum(rotation * local, axis=1)
    binding = contact(
        local_points=local, normals=normal[None], offsets=np.array([normal @ position])
    )
    runtime, state, inputs = case(
        model=model, contacts=(binding,), settings=StepSettings(substeps=1, tolerance=1e-8)
    )
    with runtime:
        result = runtime.step(state, inputs, 1.0)
    assert not result.report.accepted
    assert "geometry" in result.report.reason
