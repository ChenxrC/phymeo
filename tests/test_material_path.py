"""Actual XPBD projection-path evidence, never a replacement material solve."""

from dataclasses import replace

import numpy as np
import pytest

from phymeo import (
    BackendContext,
    ClothParticipant,
    DistanceConstraints,
    ParticleModel,
    ParticleState,
    StepInputs,
    StepSettings,
    TyxClothParticipant,
)
from phymeo.constraints.distance import project_distances
from phymeo.solvers.cloth import ClothProgram, make_cloth_runtime


def material_case(kind="shared"):
    model = ParticleModel(np.ones(3), np.zeros(3))
    state = ParticleState(
        np.array([[0.0, 0, 0], [1.2, 0.2, 0], [1.5, 1.4, 0]]),
        np.array([[0.2, 0.8, 0], [-0.3, 0.1, 0], [0.1, -0.5, 0]]),
    )
    constraints = DistanceConstraints(
        np.array([[0, 1], [1, 2]]), np.ones(2), np.array([0.001, 0.002])
    )
    if kind == "none":
        constraints = None
    elif kind == "empty":
        constraints = DistanceConstraints(np.empty((0, 2), int), np.empty(0), np.empty(0))
    elif kind == "fixed":
        model = replace(model, inverse_mass=np.array([0.0, 1, 1]))
    elif kind == "coincident":
        state.position[1] = state.position[0]
        state.velocity[:] = 0
    return model, state, constraints


def run_material(mode="numpy", trace=True, kind="shared", sweeps=None):
    model, initial, constraints = material_case(kind)
    factory = (
        ClothParticipant(model, constraints, record_material_path=trace)
        if mode == "numpy"
        else TyxClothParticipant(model, constraints, mode, record_material_path=trace)
    )
    settings = StepSettings(substeps=1, tolerance=1e-8 if mode == "numpy" else 2e-5)
    ws = factory.create_workspace(
        "cloth",
        initial,
        StepInputs.zeros(3),
        0.1,
        settings,
        BackendContext(factory.capabilities.backend),
    )
    try:
        ws.predict()
        for count in range(1, 51):
            ws.position_sweep()
            if (sweeps is not None and count == sweeps) or (
                sweeps is None and ws.position_metrics().max_residual <= settings.tolerance
            ):
                break
        else:
            raise AssertionError("material did not converge")
        ws.prepare_velocity()
        return ws.result(), ws.diagnostics(), count, initial, model, constraints
    finally:
        ws.close()


@pytest.mark.parametrize("kind", ["shared", "none", "empty", "fixed", "coincident"])
def test_numpy_path_preserves_candidate_and_records_each_sweep(kind):
    plain, old, old_count, *_ = run_material(trace=False, kind=kind)
    final, diag, count, *_ = run_material(kind=kind)
    assert count == old_count
    np.testing.assert_array_equal(final.position, plain.position)
    np.testing.assert_array_equal(final.velocity, plain.velocity)
    np.testing.assert_array_equal(diag["distance_multipliers"], old["distance_multipliers"])
    assert "material_path" not in old
    path = diag["material_path"]
    assert path["schema"] == "xpbd-distance-path-v1"
    assert len(path["columns"]) == 18
    assert len(path["frames"]) == count
    assert all(
        f.dtype == np.float64 and f.shape == (0 if kind in ("none", "empty") else 2, 18)
        for f in path["frames"]
    )
    assert path["h"] == 0.1


@pytest.mark.parametrize("value", [None, 0, 1, "yes", []])
@pytest.mark.parametrize("factory", [ClothProgram, ClothParticipant])
def test_trace_flag_is_not_truthiness(value, factory):
    model, _, constraints = material_case()
    with pytest.raises(TypeError, match="bool"):
        factory(model, constraints, record_material_path=value)


def test_numpy_bool_flag_and_signature_are_explicit():
    model, _, constraints = material_case()
    enabled = ClothProgram(model, constraints, record_material_path=np.bool_(True))
    disabled = ClothProgram(model, constraints)
    assert enabled.problem_signature()["record_material_path"] is True
    assert disabled.problem_signature()["record_material_path"] is False


def test_single_projection_records_exact_ordered_values():
    x = np.array([[0.0, 0, 0], [2.0, 0, 0]])
    weights, alpha, multipliers = np.array([1.0, 2.0]), np.array([0.5]), np.array([0.4])
    constraints = DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.array([0.005]))
    trace = np.full((1, 18), np.nan)
    project_distances(x, weights, constraints, alpha, multipliers, trace=trace)
    dl = -1.2 / 3.5
    expected = [0.0, 0, 0, 2.0, 0, 0, -dl, 0, 0, 2 + 2 * dl, 0, 0, 0.4, 0.4 + dl, -1, 0, 0, dl]
    np.testing.assert_allclose(trace[0], expected, rtol=0, atol=3e-16)
    np.testing.assert_array_equal(trace[0, 6:9], x[0])
    np.testing.assert_array_equal(trace[0, 9:12], x[1])


def test_zero_denominator_records_no_mutation_and_coincident_direction():
    x, lam = np.zeros((2, 3)), np.array([0.3])
    trace = np.full((1, 18), np.nan)
    constraints = DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.zeros(1))
    project_distances(x, np.zeros(2), constraints, np.zeros(1), lam, trace=trace)
    np.testing.assert_array_equal(trace[0], [0] * 12 + [0.3, 0.3, 0, 0, 0, 0])
    np.testing.assert_array_equal(lam, [0.3])


@pytest.mark.parametrize(
    "bad",
    [np.zeros((1, 17)), np.zeros((1, 18), np.float32), [[0.0] * 18], np.zeros((1, 36))[:, ::2]],
)
def test_invalid_trace_target_does_not_modify_state(bad):
    x, lam = np.array([[0.0, 0, 0], [2.0, 0, 0]]), np.zeros(1)
    original = x.copy()
    constraints = DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.zeros(1))
    with pytest.raises((ValueError, TypeError), match="trace"):
        project_distances(x, np.ones(2), constraints, np.zeros(1), lam, trace=bad)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(lam, [0])


def test_trace_cannot_alias_physical_inputs_or_be_readonly():
    backing = np.zeros(18)
    x = backing[:6].reshape(2, 3)
    x[1, 0] = 2
    trace, lam = backing.reshape(1, 18), np.zeros(1)
    original = backing.copy()
    constraints = DistanceConstraints(np.array([[0, 1]]), np.ones(1), np.zeros(1))
    with pytest.raises(ValueError, match="trace"):
        project_distances(x, np.ones(2), constraints, np.zeros(1), lam, trace=trace)
    np.testing.assert_array_equal(backing, original)
    target = np.zeros((1, 18))
    target.flags.writeable = False
    with pytest.raises(ValueError, match="trace"):
        project_distances(x.copy(), np.ones(2), constraints, np.zeros(1), lam, trace=target)


def test_path_snapshots_are_owned_and_runtime_helper_forwards_flag():
    model, initial, constraints = material_case()
    runtime = make_cloth_runtime(
        model, constraints, record_material_path=True, settings=StepSettings(substeps=1)
    )
    from phymeo import SceneInputs, SceneState

    with runtime:
        first = runtime.step(
            SceneState({"cloth": initial}), SceneInputs({"cloth": StepInputs.zeros(3)}), 0.1
        )
        assert first.report.accepted
        path = first.report.substeps[0].diagnostics["material_path"]
        saved = path["frames"][0].copy()
        with pytest.raises(ValueError):
            path["frames"][0][:] = 9
        second = runtime.step(
            SceneState({"cloth": initial}), SceneInputs({"cloth": StepInputs.zeros(3)}), 0.1
        )
        assert second.report.accepted
    np.testing.assert_array_equal(path["frames"][0], saved)
    np.testing.assert_array_equal(
        second.state.domains["cloth"].position, first.state.domains["cloth"].position
    )


def audit_material_path(mode):
    final, diag, _, initial, model, constraints = run_material(mode)
    path = diag["material_path"]
    tol = 2e-6 if mode != "numpy" else 2e-14
    position = path["predicted_position"].copy()
    lam = np.zeros(2)
    load = np.zeros_like(position)
    for frame in path["frames"]:
        for edge, (i, j) in enumerate(constraints.indices):
            row = frame[edge]
            np.testing.assert_allclose(row[:3], position[i], rtol=0, atol=tol)
            np.testing.assert_allclose(row[3:6], position[j], rtol=0, atol=tol)
            delta = row[:3] - row[3:6]
            length = np.linalg.norm(delta)
            direction = delta / length if length > 0 else np.array([1.0, 0, 0])
            np.testing.assert_allclose(row[14:17], direction, rtol=0, atol=tol)
            np.testing.assert_allclose(row[12], lam[edge], rtol=0, atol=tol)
            dl = -(length - constraints.rest_length[edge] + path["alpha"][edge] * row[12]) / (
                model.inverse_mass[i] + model.inverse_mass[j] + path["alpha"][edge]
            )
            np.testing.assert_allclose(row[17], dl, rtol=0, atol=tol)
            np.testing.assert_allclose(row[13], row[12] + row[17], rtol=0, atol=tol)
            correction = row[17] * row[14:17]
            np.testing.assert_allclose(
                row[6:9], row[:3] + model.inverse_mass[i] * correction, rtol=0, atol=tol
            )
            np.testing.assert_allclose(
                row[9:12], row[3:6] - model.inverse_mass[j] * correction, rtol=0, atol=tol
            )
            position[i], position[j], lam[edge] = row[6:9], row[9:12], row[13]
            load[i] += correction
            load[j] -= correction
    np.testing.assert_allclose(final.position, position, rtol=0, atol=tol)
    np.testing.assert_allclose(diag["distance_multipliers"], lam, rtol=0, atol=tol)
    np.testing.assert_allclose(
        final.velocity, path["predicted_velocity"] + load / path["h"], rtol=0, atol=10 * tol
    )
    np.testing.assert_allclose(
        np.sum(final.velocity, axis=0), np.sum(initial.velocity, axis=0), rtol=0, atol=10 * tol
    )
    final_load = np.zeros_like(position)
    for edge, (i, j) in enumerate(constraints.indices):
        delta = position[i] - position[j]
        force = lam[edge] * delta / np.linalg.norm(delta)
        final_load[i] += force
        final_load[j] -= force
    assert np.max(np.abs((final_load - load) / path["h"])) > 0.05

    def energy(state):
        lengths = np.linalg.norm(
            state.position[constraints.indices[:, 0]] - state.position[constraints.indices[:, 1]],
            axis=1,
        )
        return 0.5 * np.sum(state.velocity**2), np.sum(
            (lengths - constraints.rest_length) ** 2 / (2 * constraints.compliance)
        )

    k0, u0 = energy(initial)
    k1, u1 = energy(final)
    assert k1 > k0 and u1 < u0 and k1 + u1 < k0 + u0


def test_path_matches_incremental_equations_but_final_lambda_is_insufficient():
    audit_material_path("numpy")


def check_rotating_edge(mode, h):
    model = ParticleModel(np.ones(2), np.zeros(3))
    state = ParticleState(
        np.array([[-1.0, 0, 0], [1.0, 0, 0]]), np.array([[0.0, -1, 0], [0.0, 1, 0]])
    )
    constraints = DistanceConstraints(np.array([[0, 1]]), np.array([2.0]), np.zeros(1))
    factory = (
        ClothParticipant(model, constraints, record_material_path=True)
        if mode == "numpy"
        else TyxClothParticipant(model, constraints, mode, record_material_path=True)
    )
    ws = factory.create_workspace(
        "cloth",
        state,
        StepInputs.zeros(2),
        h,
        StepSettings(substeps=1),
        BackendContext(factory.capabilities.backend),
    )
    try:
        ws.predict()
        ws.position_sweep()
        first = ws.diagnostics()["material_path"]
        other = ws.diagnostics()["material_path"]
        assert not np.shares_memory(first["frames"][0], other["frames"][0])
        saved = first["frames"][0].copy()
        ws.position_sweep()
        assert len(first["frames"]) == 1 and len(ws.diagnostics()["material_path"]["frames"]) == 2
        np.testing.assert_array_equal(first["frames"][0], saved)
        ws.prepare_velocity()
        final = ws.result()
        angular = np.cross(final.position, final.velocity).sum(axis=0)[2]
        np.testing.assert_allclose(
            angular, 2 / np.sqrt(1 + h * h), rtol=0, atol=2e-6 if mode != "numpy" else 1e-14
        )
        assert angular < 2
    finally:
        ws.close()


@pytest.mark.parametrize("h", [0.1, 0.05, 0.025])
def test_trace_retains_original_rotating_edge_momentum_loss(h):
    check_rotating_edge("numpy", h)


def check_scene_failure(mode, failure):
    from phymeo import (
        DomainBinding,
        HistoryField,
        SceneInputs,
        SceneModel,
        SceneState,
        SceneStepper,
        SolverFailure,
    )

    model, state, constraints = material_case()
    factory = (
        ClothParticipant(model, constraints, record_material_path=True)
        if mode == "numpy"
        else TyxClothParticipant(model, constraints, mode, record_material_path=True)
    )
    scene = SceneModel((DomainBinding("cloth", "particles", model),))

    class Program:
        capabilities = replace(factory.capabilities, history="explicit")
        history_fields = (HistoryField("count", (1,), np.int64),)
        fail = failure
        closed = 0
        created = 0

        def problem_signature(self):
            return factory.problem_signature()

        def create_workspace(self, scene, state, inputs, h, settings, context):
            native = factory.create_workspace(
                "cloth", state.domains["cloth"], inputs.domains["cloth"], h, settings, context
            )
            count = state.history["count"].copy()
            self.created += 1
            owner = self

            class Workspace:
                def __getattr__(self, name):
                    return getattr(native, name)

                def prepare_velocity(self):
                    native.prepare_velocity()
                    if owner.fail == "velocity" and count[0] == 1:
                        assert len(native.diagnostics()["material_path"]["frames"]) > 0
                        raise SolverFailure("late traced candidate")

                def result(self):
                    return SceneState({"cloth": native.result()}, {"count": count + 1})

                def close(self):
                    native.close()
                    owner.closed += 1
                    if owner.fail == "close" and count[0] == 1:
                        raise SolverFailure("late traced close")

            return Workspace()

        def close(self):
            pass

    program = Program()
    initial = SceneState({"cloth": state}, {"count": np.zeros(1, np.int64)})
    inputs = SceneInputs({"cloth": StepInputs.zeros(3)})
    settings = StepSettings(substeps=2, tolerance=2e-5 if mode != "numpy" else 1e-8)
    with SceneStepper(
        scene, program, settings, BackendContext(factory.capabilities.backend)
    ) as runtime:
        failed = runtime.step(initial, inputs, 0.2)
        assert not failed.report.accepted and len(failed.report.substeps) == 2
        np.testing.assert_array_equal(failed.state.domains["cloth"].position, state.position)
        np.testing.assert_array_equal(failed.state.domains["cloth"].velocity, state.velocity)
        np.testing.assert_array_equal(failed.state.history["count"], [0])
        assert program.created == program.closed == 2
        saved = failed.report.substeps[0].diagnostics["material_path"]["frames"][0].copy()
        program.fail = None
        retry = runtime.step(initial, inputs, 0.2)
    fresh = Program()
    fresh.fail = None
    with SceneStepper(
        scene, fresh, settings, BackendContext(factory.capabilities.backend)
    ) as runtime:
        expected = runtime.step(initial, inputs, 0.2)
    assert retry.report.accepted and expected.report.accepted
    np.testing.assert_array_equal(
        retry.state.domains["cloth"].position, expected.state.domains["cloth"].position
    )
    np.testing.assert_array_equal(
        retry.state.domains["cloth"].velocity, expected.state.domains["cloth"].velocity
    )
    np.testing.assert_array_equal(retry.state.history["count"], [2])
    np.testing.assert_array_equal(
        failed.report.substeps[0].diagnostics["material_path"]["frames"][0], saved
    )
    for actual, want in zip(retry.report.substeps, expected.report.substeps, strict=True):
        a, b = actual.diagnostics["material_path"], want.diagnostics["material_path"]
        assert len(a["frames"]) == actual.position_iterations
        for fa, fb in zip(a["frames"], b["frames"], strict=True):
            np.testing.assert_array_equal(fa, fb)
    assert program.closed == program.created == 4


@pytest.mark.parametrize("failure", ["velocity", "close"])
def test_traced_later_substep_failure_rolls_back_and_retry_is_fresh(failure):
    check_scene_failure("numpy", failure)
