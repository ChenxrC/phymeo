"""Actual particle adapter transport and bounded terminal boundary controls."""

import numpy as np
import pytest
from test_point_participant_program import (
    observed_program,
    particle_fixture,
    terminal_requests,
    terminal_trial,
)

from phymeo.core.capabilities import SolverFailure
from phymeo.core.scene import fingerprint
from phymeo.runtime.scene_stepper import SceneStepper
from phymeo.runtime.terminal_impulse import TerminalImpulseProgram

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("material", [False, True])
def test_actual_point_adapter_bound_stages_and_terminal_receipts(tyx_mode, material):
    before, _, _, result = terminal_trial(tyx_mode, material=material, history=True)
    assert result.report.accepted
    after = result.state.domains["cloth"]
    np.testing.assert_allclose(
        after.position[:, 0],
        [3 / 64, 63 / 64] if material else [5 / 64, 31 / 32],
        rtol=0,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        after.velocity[:, 0], [9 / 16, 1 / 16] if material else [11 / 16, 0.0], rtol=0, atol=2e-5
    )
    assert np.array_equal(after.position, after.position.astype(np.float32).astype(np.float64))
    assert np.array_equal(after.velocity, after.velocity.astype(np.float32).astype(np.float64))
    receipt = result.report.substeps[0].diagnostics["terminal_impulse"]
    assert receipt["applications"] == 1
    assert receipt["post_execution_delta"]["cloth"]["completed_launches"] > 0
    assert (
        receipt["post_execution_delta"]["cloth"]["completed_launches"]
        == receipt["post_execution_delta"]["cloth"]["launches"]
    )
    stats = result.report.substeps[0].diagnostics["participants"]["cloth"]["execution"]
    assert stats["mode"] == tyx_mode and stats["completed_launches"] == stats["launches"] > 0
    if tyx_mode == "cuda":
        assert stats["uploads"] > 0 and stats["downloads"] > 0
    assert before.history["counter"].tobytes() == result.state.history["counter"].tobytes()


def test_actual_point_adapter_terminal_failure_retries_from_original(tyx_mode):
    scene, state, inputs, plain, settings, context = particle_fixture(tyx_mode, history=True)

    def fail(operation):
        if operation == "apply_velocity_impulses":
            raise SolverFailure("injected native terminal failure after operation")

    events = []
    base = observed_program(plain, events, {"cloth": fail})
    wrapper = TerminalImpulseProgram(base, terminal_requests(), registry=base.registry)
    original = fingerprint((state, inputs))
    with SceneStepper(scene, wrapper, settings, context, base.registry) as runtime:
        failed = runtime.step(state, inputs, 0.25)
    assert not failed.report.accepted and fingerprint(failed.state) == fingerprint(state)
    assert fingerprint((state, inputs)) == original
    assert events.count(("cloth", "create")) == events.count(("cloth", "close"))
    base.participants["cloth"].hook = None
    with SceneStepper(scene, wrapper, settings, context, base.registry) as runtime:
        retry = runtime.step(state, inputs, 0.25)
    _, _, _, fresh = terminal_trial(tyx_mode, history=True)
    assert retry.report.accepted and fresh.report.accepted
    assert fingerprint(retry.state) == fingerprint(fresh.state)


def test_actual_point_adapter_bounded_distance_terminal(tyx_mode):
    """One distance substep plus one terminal batch; the bounded cpu_cpp node."""
    _, _, _, result = terminal_trial(tyx_mode, material=True)
    assert result.report.accepted
    receipt = result.report.substeps[0].diagnostics["terminal_impulse"]
    assert receipt["applications"] == 1
    np.testing.assert_allclose(
        result.state.domains["cloth"].velocity[:, 0], [9 / 16, 1 / 16], rtol=0, atol=2e-5
    )
    assert receipt["post_execution_delta"]["cloth"]["completed_launches"] > 0
