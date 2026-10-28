"""Thin verifier on actual native candidate endpoints, with outer rollback."""

from dataclasses import replace

import numpy as np
import pytest
from test_feature_distance_bound_transaction import GeometryProgram, assert_state_equal

import phymeo as ph


class SweepVerifier:
    def __init__(self, program, budget=128):
        self.program, self.budget, self.results = program, budget, []

    def problem_signature(self):
        return self.program.problem_signature()

    def verify(self, scene, before, inputs, h, after):
        names = [domain.name for domain in scene.domains]
        a0, b0 = (before.domains[name].position for name in names)
        a1, b1 = (after.domains[name].position for name in names)
        assert a0.dtype == a1.dtype == b0.dtype == b1.dtype == np.float64
        result = ph.linear_feature_sweep(
            a0, a1, b0, b1, clearance_m=0.25, max_intervals=self.budget
        )
        self.results.append((result, a0.copy(), a1.copy(), b0.copy(), b1.copy()))
        return ph.VerificationReport(result.status == "certified", reason=result.status)


def case(kind):
    if kind == "unresolved":
        # Spec's rotating affine rod, extrapolated back one substep. Its
        # distance²=(1-t)²+t² >= .5, but t=0..1 has no fixed separator.
        positions = ([[0, 0, -0.25], [0, 0, 0.25]], [[0, -5, 0], [4, 3, 0]])
        velocities = (np.zeros((2, 3)), [[2, 6, 0], [-6, -2, 0]])
        names = ("first", "second")
    else:
        positions = ([[0.5, 0.5, 3]], [[0, 0, 0], [2, 0, 0], [0, 2, 0]])
        velocities = ([[0, 0, -4 if kind == "violation" else 0.5]], [[0, 0, 0.5]] * 3)
        names = ("probe", "plate")
    scene = ph.SceneModel(
        tuple(
            ph.DomainBinding(name, "particles", ph.ParticleModel(np.ones(len(p)), np.zeros(3)))
            for name, p in zip(names, positions)
        )
    )
    initial = ph.SceneState(
        {
            name: ph.ParticleState(np.array(p, float), np.array(v, float))
            for name, p, v in zip(names, positions, velocities)
        },
        {"tick": np.array([4], np.int64), "control": np.array([2.0, -3.0])},
    )
    inputs = ph.SceneInputs(
        {name: ph.StepInputs.zeros(len(p)) for name, p in zip(names, positions)}
    )
    return scene, initial, inputs


def run_case(mode, kind):
    scene, initial, inputs = case(kind)
    original = replace(
        initial,
        domains={k: v.copy() for k, v in initial.domains.items()},
        history={k: v.copy() for k, v in initial.history.items()},
    )
    forces = {name: value.force.copy() for name, value in inputs.domains.items()}
    program = GeometryProgram(mode, "close" if kind == "close" else None)
    verifier = SweepVerifier(program, 1 if kind == "unresolved" else 128)
    settings = ph.StepSettings(substeps=2, record_substeps=True)
    context = ph.BackendContext(program.capabilities.backend)
    with ph.SceneStepper(scene, program, settings, context, verifier=verifier) as runtime:
        result = runtime.step(initial, inputs, 1.0)
        assert len(verifier.results) == 2
        assert verifier.results[0][0].status == "certified"
        assert verifier.results[-1][0].status == (
            kind if kind in ("violation", "unresolved") else "certified"
        )
        assert len(result.report.substeps) == 2 and result.report.substeps[0].accepted
        if kind == "violation":
            # Independent semi-implicit endpoint oracle: h=.5, no forces.
            np.testing.assert_array_equal(verifier.results[0][2], [[0.5, 0.5, 1.0]])
            np.testing.assert_array_equal(verifier.results[1][2], [[0.5, 0.5, -1.0]])
            np.testing.assert_array_equal(verifier.results[0][4][:, 2], [0.25] * 3)
            np.testing.assert_array_equal(verifier.results[1][4][:, 2], [0.5] * 3)
        elif kind == "unresolved":
            np.testing.assert_array_equal(
                verifier.results[1][3], [[1.0, -2.0, 0.0], [1.0, 2.0, 0.0]]
            )
            np.testing.assert_array_equal(
                verifier.results[1][4], [[2.0, 1.0, 0.0], [-2.0, 1.0, 0.0]]
            )
        if kind != "safe":
            assert not result.report.accepted
            assert (
                kind if kind != "close" else "late geometry resource close"
            ) in result.report.reason
            assert result.substep_states == ()
            assert_state_equal(result.state, original)
        else:
            assert result.report.accepted
        assert_state_equal(initial, original)
        assert program.created == program.closed == 2
        program.fault = None
        retry = runtime.step(initial, inputs, 1.0)
    fresh_program = GeometryProgram(mode)
    fresh_verifier = SweepVerifier(fresh_program, verifier.budget)
    with ph.SceneStepper(
        scene, fresh_program, settings, context, verifier=fresh_verifier
    ) as runtime:
        fresh = runtime.step(initial, inputs, 1.0)
    assert (
        retry.report.accepted == fresh.report.accepted == (kind not in ("violation", "unresolved"))
    )
    assert_state_equal(retry.state, fresh.state)
    assert_state_equal(initial, original)
    for name, force in forces.items():
        np.testing.assert_array_equal(inputs.domains[name].force, force)
    if retry.report.accepted:
        np.testing.assert_array_equal(retry.state.history["tick"], [6])
        np.testing.assert_array_equal(retry.state.history["control"], [2.5, -4.0])
        np.testing.assert_array_equal(retry.state.domains["probe"].position, [[0.5, 0.5, 3.5]])
        np.testing.assert_array_equal(retry.state.domains["plate"].position[:, 2], [0.5] * 3)
    for row, fresh_row in zip(verifier.results[-2:], fresh_verifier.results, strict=True):
        assert row[0] == fresh_row[0]
        for actual, expected in zip(row[1:], fresh_row[1:], strict=True):
            np.testing.assert_array_equal(actual, expected)
    assert program.created == program.closed == 4
    if mode != "numpy":
        assert retry.report.backend == "tyx:" + mode
        for diagnostic in program.native:
            for domain in diagnostic.values():
                assert domain["execution"]["completed_launches"] > 0
                assert domain["execution"]["mode"] == mode


@pytest.mark.parametrize("kind", ["safe", "violation", "unresolved", "close"])
def test_actual_candidates_sweep_and_outer_rollback(kind):
    run_case("numpy", kind)


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ["safe", "violation"])
def test_actual_tyx_endpoints_host_sweep(tyx_mode, kind):
    run_case(tyx_mode, kind)
