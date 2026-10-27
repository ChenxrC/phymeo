"""Physical consumers and failure injection exercise the generic staged transaction."""

import importlib
from dataclasses import dataclass, replace

import numpy as np
import pytest

from phymeo.core.capabilities import SolverFailure
from phymeo.core.model import ParticleModel
from phymeo.core.rigid import RigidBodyInputs, RigidBodyModel, RigidBodyState
from phymeo.core.scene import (
    BackendContext,
    DomainBinding,
    HistoryField,
    SceneInputs,
    SceneModel,
    SceneState,
    StageMetrics,
    StepSettings,
)
from phymeo.core.state import ParticleState, StepInputs
from phymeo.core.verification import VerificationReport
from phymeo.runtime.registry import StateFamily, StateRegistry


def api():
    return importlib.import_module("phymeo.runtime.scene_stepper")


def contracts():
    return importlib.import_module("phymeo.runtime.program")


def free():
    return importlib.import_module("phymeo.solvers.free_scene").FreeSceneProgram()


def mixed(dtype=np.float64):
    p = ParticleModel(np.array([0.5], dtype), np.array([0, -2, 0], dtype))
    b = RigidBodyModel(np.array([0.25], dtype), np.full((1, 3), 2, dtype), np.zeros(3, dtype))
    scene = SceneModel(
        (DomainBinding("dust", "particles", p), DomainBinding("body", "rigid_bodies", b))
    )
    state = SceneState(
        {
            "dust": ParticleState(np.zeros((1, 3), dtype), np.zeros((1, 3), dtype)),
            "body": RigidBodyState(
                np.zeros((1, 3), dtype),
                np.array([[0, 0, 0, 1]], dtype),
                np.zeros((1, 3), dtype),
                np.zeros((1, 3), dtype),
            ),
        }
    )
    inputs = SceneInputs(
        {
            "dust": StepInputs(np.array([[4, 0, 0]], dtype)),
            "body": RigidBodyInputs(np.array([[4, 0, 0]], dtype), np.array([[0, 0, 2]], dtype)),
        }
    )
    return scene, state, inputs


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_free_mixed_scene_matches_analytic_acceleration_and_torque(dtype):
    scene, state, inputs = mixed(dtype)
    runner = api().SceneStepper(
        scene, free(), StepSettings(substeps=4, record_substeps=True), BackendContext(dtype=dtype)
    )
    result = runner.step(state, inputs, 0.4)
    assert result.report.accepted and len(result.substep_states) == 4
    # Semi-implicit x = a h^2 n(n+1)/2, v = a n h; isotropic I gives alpha_z=1.
    np.testing.assert_allclose(result.state.domains["dust"].position, [[0.2, -0.2, 0]], atol=1e-7)
    np.testing.assert_allclose(result.state.domains["dust"].velocity, [[0.8, -0.8, 0]], atol=1e-7)
    body = result.state.domains["body"]
    np.testing.assert_allclose(body.position, [[0.1, 0, 0]], atol=1e-7)
    np.testing.assert_allclose(body.angular_velocity, [[0, 0, 0.4]], atol=1e-7)
    np.testing.assert_allclose(body.orientation, [[0, 0, np.sin(0.05), np.cos(0.05)]], atol=1e-7)
    np.testing.assert_array_equal(state.domains["dust"].position, np.zeros((1, 3)))
    result.state.domains["dust"].position[:] = 20
    assert result.substep_states[-1].domains["dust"].position[0, 0] < 1
    assert all(r.position_iterations == r.velocity_iterations == 1 for r in result.report.substeps)
    assert all(v >= 0 for r in result.report.substeps for v in r.timings.values())


@dataclass(frozen=True)
class ScalarModel:
    mass: float = 2.0


@dataclass
class ScalarState:
    q: np.ndarray
    v: np.ndarray
    plastic: np.ndarray


@dataclass
class ScalarInputs:
    tau: np.ndarray


def scalar_case():
    def validate(model, state, inputs):
        if any(
            getattr(state, f).shape != (1,) for f in ("q", "v", "plastic")
        ) or inputs.tau.shape != (1,):
            raise ValueError("scalar shape")

    registry = StateRegistry()
    registry.register(
        StateFamily(
            "scalar",
            ScalarModel,
            ScalarState,
            ScalarInputs,
            ("q", "v", "plastic"),
            ("tau",),
            "q",
            validate,
        )
    )
    scene = SceneModel((DomainBinding("osc", "scalar", ScalarModel()),))
    state = SceneState(
        {"osc": ScalarState(np.array([1.0]), np.array([0.0]), np.array([3.0]))},
        {"count": np.array([0], np.int64)},
    )
    inputs = SceneInputs({"osc": ScalarInputs(np.array([2.0]))})
    return scene, state, inputs, registry


class ScalarProgram:
    """Separate real scalar dynamics, with an explicit count used as failure location."""

    def __init__(self, fault=None):
        self.capabilities = contracts().ProgramCapabilities(
            "numpy", {"float64"}, {"scalar"}, history="explicit"
        )
        self.history_fields = (HistoryField("count", (1,), np.int64),)
        self.fault, self.closed, self.created = fault, 0, 0
        self.signature = {"law": "scalar-force-v1"}
        self.callback = None

    def problem_signature(self):
        return self.signature

    def close(self):
        pass

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.created += 1
        return ScalarWorkspace(self, scene, state, inputs, h)


class ScalarWorkspace:
    def __init__(self, program, scene, state, inputs, h):
        self.p, self.scene, self.state, self.inputs, self.h = program, scene, state, inputs, h
        self.late = state.history["count"][0] == 1
        self.trace = []

    def predict(self):
        self.trace.append("predict")
        if self.p.callback:
            self.p.callback()
        value = self.state.domains["osc"]
        value.v[:] += self.inputs.domains["osc"].tau / self.scene.domains[0].model.mass * self.h
        value.q[:] += value.v * self.h
        self.state.history["count"][:] += 1
        self.inputs.domains["osc"].tau[:] = 999  # per-attempt inputs must be owned
        if self.late and self.p.fault == "predict":
            raise SolverFailure("late prediction")
        if self.p.fault == "program_error":
            raise RuntimeError("program bug")

    def position_sweep(self):
        self.trace.append("position")

    def position_metrics(self):
        if self.p.fault == "bad_metrics":
            return {"residual": 0}
        return StageMetrics({"scalar": 1 if self.late and self.p.fault == "position_budget" else 0})

    def prepare_velocity(self):
        self.trace.append("prepare")

    def velocity_sweep(self):
        self.trace.append("velocity")

    def velocity_metrics(self):
        return StageMetrics({"scalar": 1 if self.late and self.p.fault == "velocity_budget" else 0})

    def result(self):
        if self.late:
            if self.p.fault == "bad_history":
                return SceneState(self.state.domains, {"count": np.zeros(2, np.int64)})
            if self.p.fault == "bad_candidate":
                self.state.domains["osc"].plastic = np.zeros(2)
            if self.p.fault == "nan_candidate":
                self.state.domains["osc"].q[:] = np.nan
        return self.state

    def diagnostics(self):
        return {"trace": tuple(self.trace), "count": self.state.history["count"]}

    def close(self):
        self.p.closed += 1
        self.state.history["count"][:] = 999  # result must already be snapshotted
        if self.late and self.p.fault == "close":
            raise SolverFailure("late close")


def scalar_runner(program=None, **kwargs):
    scene, state, inputs, registry = scalar_case()
    program = ScalarProgram() if program is None else program
    runner = api().SceneStepper(
        scene,
        program,
        StepSettings(
            substeps=2, position_iterations=2, velocity_iterations=2, record_substeps=True
        ),
        BackendContext(),
        registry,
        **kwargs,
    )
    return runner, program, state, inputs


def test_registered_q_v_program_preserves_additional_state_and_history_across_calls():
    runner, program, state, inputs = scalar_runner()
    first = runner.step(state, inputs, 0.2)
    second = runner.step(first.state, inputs, 0.2)
    assert first.report.accepted and second.report.accepted and program.closed == 4
    np.testing.assert_allclose(first.state.domains["osc"].q, [1.03])
    np.testing.assert_allclose(second.state.domains["osc"].q, [1.10])
    np.testing.assert_array_equal(second.state.domains["osc"].plastic, [3])
    np.testing.assert_array_equal(second.state.history["count"], [4])
    np.testing.assert_array_equal(state.history["count"], [0])
    np.testing.assert_array_equal(inputs.domains["osc"].tau, [2])
    assert first.report.substeps[0].diagnostics["trace"] == (
        "predict",
        "position",
        "prepare",
        "velocity",
    )
    assert first.report.substeps[0].diagnostics["count"][0] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "predict",
        "position_budget",
        "velocity_budget",
        "bad_candidate",
        "nan_candidate",
        "bad_history",
        "close",
        "bad_metrics",
    ],
)
def test_failed_attempt_rolls_back_all_domains_history_and_snapshots_then_can_retry(fault):
    runner, program, state, inputs = scalar_runner(ScalarProgram(fault))
    failed = runner.step(state, inputs, 0.2)
    assert not failed.report.accepted and failed.report.reason and failed.substep_states == ()
    np.testing.assert_array_equal(failed.state.domains["osc"].q, [1])
    np.testing.assert_array_equal(failed.state.domains["osc"].v, [0])
    np.testing.assert_array_equal(failed.state.history["count"], [0])
    assert program.created == program.closed
    program.fault = None
    assert runner.step(state, inputs, 0.2).report.accepted


def test_programming_exception_preserves_traceback_and_closes_workspace():
    runner, program, state, inputs = scalar_runner(ScalarProgram("program_error"))
    with pytest.raises(RuntimeError, match="program bug"):
        runner.step(state, inputs, 0.2)
    assert program.closed == 1 and state.history["count"][0] == 0
    program.fault = None
    assert runner.step(state, inputs, 0.2).report.accepted


@pytest.mark.parametrize("change", ["signature", "schema", "capabilities", "model"])
def test_binding_changes_are_rejected_before_creating_workspace(change):
    runner, program, state, inputs = scalar_runner()
    if change == "signature":
        program.signature["law"] = "different"
    elif change == "schema":
        program.history_fields = (HistoryField("count", (2,), np.int64),)
    elif change == "capabilities":
        program.capabilities = replace(program.capabilities, backend="another")
    else:
        object.__setattr__(runner.scene.domains[0].model, "mass", 3.0)
    with pytest.raises(ValueError):
        runner.step(state, inputs, 0.2)
    assert program.created == 0


@pytest.mark.parametrize(
    "change", ["hidden", "wrong_backend", "wrong_dtype", "missing_family", "learned"]
)
def test_unsupported_program_cannot_bind(change):
    program = ScalarProgram()
    if change == "hidden":
        with pytest.raises(ValueError):
            replace(program.capabilities, history="hidden")
        return
    if change == "wrong_backend":
        program.capabilities = replace(program.capabilities, backend="cuda")
    elif change == "wrong_dtype":
        program.capabilities = replace(program.capabilities, dtypes={"float32"})
    elif change == "missing_family":
        program.capabilities = replace(program.capabilities, families={"particles"})
    else:
        program.capabilities = replace(program.capabilities, provider_kind="learned")
    with pytest.raises(ValueError):
        scalar_runner(program)


class Verifier:
    def __init__(self, result):
        self.result = result

    def problem_signature(self):
        return {"law": "scalar-force-v1"}

    def verify(self, scene, before, inputs, h, after):
        # A verifier can damage its copies, but cannot bless a different returned state.
        assert inputs.domains["osc"].tau[0] == 2
        before.history["count"][:] = 700
        after.history["count"][:] = 800
        after.domains["osc"].q[:] = 900
        return self.result


@pytest.mark.parametrize(
    "report",
    [
        VerificationReport(False, reason="physics"),
        VerificationReport(True, max_residual=1),
        VerificationReport(True, max_residual=np.nan),
        VerificationReport("yes"),
        None,
    ],
)
def test_learned_candidate_requires_independent_valid_physics_report(report):
    program = ScalarProgram()
    program.capabilities = replace(program.capabilities, provider_kind="learned")
    runner, _, state, inputs = scalar_runner(program, verifier=Verifier(report))
    result = runner.step(state, inputs, 0.2)
    assert not result.report.accepted and result.state.history["count"][0] == 0


def test_verifier_copies_cannot_rewrite_accepted_candidate():
    runner, _, state, inputs = scalar_runner(verifier=Verifier(VerificationReport(True)))
    result = runner.step(state, inputs, 0.2)
    assert result.report.accepted and result.state.history["count"][0] == 2
    np.testing.assert_allclose(result.state.domains["osc"].q, [1.03])


def test_reentrant_and_closed_runner_reject_without_poisoning_later_calls():
    runner, program, state, inputs = scalar_runner()
    program.callback = lambda: runner.step(state, inputs, 0.2)
    with pytest.raises(RuntimeError, match="active"):
        runner.step(state, inputs, 0.2)
    program.callback = None
    assert runner.step(state, inputs, 0.2).report.accepted
    runner.close()
    runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner.step(state, inputs, 0.2)


@pytest.mark.parametrize("dt", [0, -1, True, float("nan"), 1e-320])
def test_invalid_or_unrepresentable_dt_never_starts_program(dt):
    scene, state, inputs = mixed(np.float32)
    runner = api().SceneStepper(scene, free(), context=BackendContext(dtype=np.float32))
    with pytest.raises(ValueError):
        runner.step(state, inputs, dt)


def test_real_point_exchange_program_uses_unmodified_scene_scheduler():
    from phymeo.coupling.ports import ParticlePointPort, exchange_impulses

    class ExchangeProgram(type(free())):
        def problem_signature(self):
            return {"law": "prescribed-paired-impulse-v1", "force": 2.0}

        def create_workspace(self, scene, state, inputs, h, settings, context):
            workspace = super().create_workspace(scene, state, inputs, h, settings, context)

            def apply():
                ports = [
                    ParticlePointPort(d.model, workspace.state.domains[d.name], [0], domain=d.name)
                    for d in scene.domains
                ]
                result = exchange_impulses(*ports, np.array([[2 * h, 0, 0]]))
                workspace.state = SceneState(result.states)

            workspace.velocity_sweep = apply
            return workspace

    domains = tuple(
        DomainBinding(name, "particles", ParticleModel(np.array([inverse]), np.zeros(3)))
        for name, inverse in (("left", 1.0), ("right", 0.5))
    )
    scene = SceneModel(domains)
    state = SceneState({d.name: ParticleState(np.zeros((1, 3)), np.zeros((1, 3))) for d in domains})
    inputs = SceneInputs({d.name: StepInputs.zeros(1) for d in domains})
    with api().SceneStepper(scene, ExchangeProgram(), StepSettings(substeps=1)) as runner:
        result = runner.step(state, inputs, 0.1)
    assert result.report.accepted
    left, right = (result.state.domains[n] for n in ("left", "right"))
    np.testing.assert_allclose(left.velocity, [[0.2, 0, 0]])
    np.testing.assert_allclose(right.velocity, [[-0.1, 0, 0]])
    np.testing.assert_allclose(left.velocity + 2 * right.velocity, np.zeros((1, 3)))


def test_independent_residual_is_retained_in_report():
    runner, _, state, inputs = scalar_runner(
        verifier=Verifier(VerificationReport(True, max_residual=1e-8, max_constraint_error=0.01))
    )
    result = runner.step(state, inputs, 0.2)
    assert result.report.accepted
    assert result.report.max_residual == 1e-8
    assert result.report.substeps[0].verification.max_constraint_error == 0.01


def test_bad_verifier_signature_and_changed_signature_are_rejected():
    verifier = Verifier(VerificationReport(True))
    verifier.problem_signature = lambda: {"wrong": True}
    with pytest.raises(ValueError, match="verifier"):
        scalar_runner(verifier=verifier)
    verifier = Verifier(VerificationReport(True))
    runner, program, state, inputs = scalar_runner(verifier=verifier)
    verifier.problem_signature = lambda: {"wrong": True}
    with pytest.raises(ValueError, match="verifier"):
        runner.step(state, inputs, 0.2)
    assert program.created == 0


@pytest.mark.parametrize("fault", ["fixed", "quaternion", "angular_nan", "second_domain"])
def test_mixed_failure_reverts_first_domain_too(fault):
    class BadProgram(type(free())):
        def create_workspace(self, scene, state, inputs, h, settings, context):
            workspace = super().create_workspace(scene, state, inputs, h, settings, context)
            original = workspace.result

            def corrupt():
                result = original()
                body = result.domains["body"]
                if fault == "quaternion":
                    body.orientation[:] = 0
                elif fault == "angular_nan":
                    body.angular_velocity[:] = np.nan
                elif fault == "fixed":
                    body.position[:] = 1
                else:
                    body.angular_velocity = np.zeros((2, 3))
                return result

            workspace.result = corrupt
            return workspace

    scene, state, inputs = mixed()
    if fault == "fixed":
        fixed = RigidBodyModel(np.array([0.0]), np.zeros((1, 3)), np.zeros(3))
        scene = SceneModel((scene.domains[0], DomainBinding("body", "rigid_bodies", fixed)))
    result = api().SceneStepper(scene, BadProgram()).step(state, inputs, 0.1)
    assert not result.report.accepted
    np.testing.assert_array_equal(
        result.state.domains["dust"].position, state.domains["dust"].position
    )
    np.testing.assert_array_equal(
        result.state.domains["body"].orientation, state.domains["body"].orientation
    )


def test_schema_and_metrics_corruption_after_bind_do_not_succeed():
    program = ScalarProgram()
    runner, _, state, inputs = scalar_runner(program)
    program.callback = lambda: setattr(program, "history_fields", ())
    with pytest.raises(ValueError, match="schema"):
        runner.step(state, inputs, 0.2)
    assert program.closed == 1 and state.history["count"][0] == 0


def test_partial_workspace_construction_retains_original_history():
    program = ScalarProgram()
    runner, _, state, inputs = scalar_runner(program)

    def fail(scene, candidate, forces, *args):
        candidate.history["count"][:] = 42
        forces.domains["osc"].tau[:] = 99
        raise SolverFailure("partial allocation cleaned by constructor")

    program.create_workspace = fail
    result = runner.step(state, inputs, 0.2)
    assert not result.report.accepted and result.substep_states == ()
    assert result.state.history["count"][0] == state.history["count"][0] == 0
    assert inputs.domains["osc"].tau[0] == 2
