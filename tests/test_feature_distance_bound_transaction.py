"""Host certificates consume real scene candidates, within generic transactions."""

from contextlib import ExitStack
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from test_feature_distance_bound import certificate, true_distance_squared

import phymeo as ph


class GeometryProgram:
    """Test-side composition of existing physical participants; no new runtime."""

    def __init__(self, mode="numpy", fault=None):
        self.mode, self.fault = mode, fault
        self.created = self.closed = 0
        self.native = []
        self.capabilities = ph.ProgramCapabilities(
            "numpy" if mode == "numpy" else "tyx:" + mode,
            {"float64"},
            {"particles"},
            history="explicit",
        )
        self.history_fields = (
            ph.HistoryField("tick", (1,), np.int64),
            ph.HistoryField("control", (2,), np.float64),
        )

    def problem_signature(self):
        return {"law": "independent-probe-plate-free-motion-v1"}

    def create_workspace(self, scene, state, inputs, h, settings, context):
        self.created += 1
        with ExitStack() as resources:
            natives = {}
            for domain in scene.domains:
                factory = (
                    ph.ClothParticipant(domain.model, None)
                    if self.mode == "numpy"
                    else ph.TyxClothParticipant(domain.model, None, self.mode)
                )
                native = factory.create_workspace(
                    domain.name,
                    state.domains[domain.name],
                    inputs.domains[domain.name],
                    h,
                    settings,
                    context,
                )
                resources.callback(native.close)
                natives[domain.name] = native
            owned = resources.pop_all()
        return GeometryWorkspace(self, state, natives, owned)

    def close(self):
        pass


class GeometryWorkspace:
    def __init__(self, owner, state, natives, resources):
        self.owner, self.before, self.natives, self.resources = owner, state, natives, resources

    def predict(self):
        for native in self.natives.values():
            native.predict()

    def position_sweep(self):
        for native in self.natives.values():
            native.position_sweep()

    def position_metrics(self):
        return ph.StageMetrics()

    def prepare_velocity(self):
        for native in self.natives.values():
            native.prepare_velocity()

    def velocity_sweep(self):
        for native in self.natives.values():
            native.velocity_sweep()

    def velocity_metrics(self):
        return ph.StageMetrics()

    def result(self):
        return ph.SceneState(
            {name: native.result() for name, native in self.natives.items()},
            {
                "tick": self.before.history["tick"] + 1,
                "control": self.before.history["control"] + np.array([0.25, -0.5]),
            },
        )

    def diagnostics(self):
        data = {name: native.diagnostics() for name, native in self.natives.items()}
        self.owner.native.append(data)
        return data

    def close(self):
        self.resources.close()
        self.owner.closed += 1
        if self.owner.fault == "close" and self.before.history["tick"][0] == 5:
            raise ph.SolverFailure("late geometry resource close")


class GeometryVerifier:
    def __init__(self, program):
        self.program, self.certified = program, []

    def problem_signature(self):
        return self.program.problem_signature()

    def verify(self, scene, before, inputs, h, after):
        point, plate = after.domains["probe"].position, after.domains["plate"].position
        # Explicitly host FP64 geometry on actual candidate state, never CUDA geometry.
        assert point.dtype == plate.dtype == np.float64
        result = certificate(point, plate, [1.0], [0.5, 0.25, 0.25])
        squared = true_distance_squared(point, plate)
        assert Fraction(result.lower_m) ** 2 <= squared <= Fraction(result.upper_m) ** 2
        self.certified.append((after.history["tick"][0], point.copy(), plate.copy(), result))
        if self.program.fault == "certificate" and after.history["tick"][0] == 6:
            # Exercise the real certificate's range rejection at the last candidate.
            maximum = np.finfo(float).max
            overflow_point = point.copy()
            overflow_point[0, 0] = maximum
            overflow_plate = plate.copy()
            overflow_plate[:, 0] = -maximum
            certificate(overflow_point, overflow_plate, [1.0], [0.5, 0.25, 0.25])
        return ph.VerificationReport(True)


def case():
    scene = ph.SceneModel(
        tuple(
            ph.DomainBinding(name, "particles", ph.ParticleModel(np.ones(n), np.zeros(3)))
            for name, n in [("probe", 1), ("plate", 3)]
        )
    )
    initial = ph.SceneState(
        {
            "probe": ph.ParticleState(np.array([[0.5, 0.5, 3.0]]), np.array([[0.0, 0.0, 0.5]])),
            "plate": ph.ParticleState(
                np.array([[0.0, 0, 0], [2.0, 0, 0], [0.0, 2, 0]]), np.tile([0.0, 0.0, -0.5], (3, 1))
            ),
        },
        {"tick": np.array([4], np.int64), "control": np.array([2.0, -3.0])},
    )
    inputs = ph.SceneInputs({"probe": ph.StepInputs.zeros(1), "plate": ph.StepInputs.zeros(3)})
    inputs.domains["probe"].force[0, 2] = 0.5
    return scene, initial, inputs


def assert_state_equal(actual, expected):
    assert set(actual.domains) == set(expected.domains)
    for name in actual.domains:
        np.testing.assert_array_equal(
            actual.domains[name].position, expected.domains[name].position
        )
        np.testing.assert_array_equal(
            actual.domains[name].velocity, expected.domains[name].velocity
        )
    assert set(actual.history) == set(expected.history)
    for name in actual.history:
        np.testing.assert_array_equal(actual.history[name], expected.history[name])


def run_transaction(mode, fault):
    scene, initial, inputs = case()
    original = replace(
        initial,
        domains={k: v.copy() for k, v in initial.domains.items()},
        history={k: v.copy() for k, v in initial.history.items()},
    )
    forces = {k: v.force.copy() for k, v in inputs.domains.items()}
    settings = ph.StepSettings(substeps=2, record_substeps=True)
    program = GeometryProgram(mode, fault)
    verifier = GeometryVerifier(program)
    context = ph.BackendContext(program.capabilities.backend)
    with ph.SceneStepper(scene, program, settings, context, verifier=verifier) as runtime:
        failed = runtime.step(initial, inputs, 0.5)
        assert not failed.report.accepted
        assert len(failed.report.substeps) == 2
        assert failed.report.substeps[0].accepted
        assert (
            "represent" in failed.report.reason
            if fault == "certificate"
            else "late geometry resource close" in failed.report.reason
        )
        assert_state_equal(failed.state, original)
        assert_state_equal(initial, original)
        assert failed.substep_states == ()
        assert program.created == program.closed == 2
        assert [row[0] for row in verifier.certified] == [5, 6]
        assert verifier.certified[-1][1][0, 2] > initial.domains["probe"].position[0, 2]
        assert verifier.certified[-1][2][0, 2] < initial.domains["plate"].position[0, 2]
        program.fault = None
        retry = runtime.step(initial, inputs, 0.5)
    fresh_program = GeometryProgram(mode)
    fresh_verifier = GeometryVerifier(fresh_program)
    with ph.SceneStepper(
        scene, fresh_program, settings, context, verifier=fresh_verifier
    ) as runtime:
        fresh = runtime.step(initial, inputs, 0.5)
    assert retry.report.accepted and fresh.report.accepted
    assert_state_equal(retry.state, fresh.state)
    assert_state_equal(initial, original)
    for name in inputs.domains:
        np.testing.assert_array_equal(inputs.domains[name].force, forces[name])
    np.testing.assert_array_equal(retry.state.history["tick"], [6])
    np.testing.assert_array_equal(retry.state.history["control"], [2.5, -4.0])
    for a, b in zip(retry.substep_states, fresh.substep_states, strict=True):
        assert_state_equal(a, b)
    for a, b in zip(verifier.certified[-2:], fresh_verifier.certified, strict=True):
        assert a[3] == b[3]
    assert program.created == program.closed == 4
    # Binary-exact h=.25 and force=.5: independent semi-implicit answers.
    np.testing.assert_array_equal(retry.state.domains["probe"].position, [[0.5, 0.5, 3.34375]])
    np.testing.assert_array_equal(retry.state.domains["probe"].velocity, [[0.0, 0.0, 0.75]])
    np.testing.assert_array_equal(retry.state.domains["plate"].position[:, 2], [-0.25] * 3)
    assert verifier.certified[-1][3].lower_m == verifier.certified[-1][3].upper_m == 3.59375
    if mode != "numpy":
        # These statistics are produced by the real tyx participant engines.
        for diagnostic in program.native:
            for domain in diagnostic.values():
                execution = domain["execution"]
                assert execution["completed_launches"] > 0
                assert execution["mode"] == mode
        assert retry.report.backend == "tyx:" + mode


@pytest.mark.parametrize("fault", ["certificate", "close"])
def test_last_substep_host_certificate_transaction_rollback_and_retry(fault):
    run_transaction("numpy", fault)


@pytest.mark.parametrize("fault", ["certificate", "close"])
def test_actual_tyx_snapshot_host_certificate_and_transaction(tyx_mode, fault):
    run_transaction(tyx_mode, fault)
