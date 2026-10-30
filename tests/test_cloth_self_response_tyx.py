"""Actual tyx S2-L controls; explicit --require-tyx is mandatory at validation.

All assertions consume the same physics runs as the oracle and saved evidence.
The bounded C++ node is selected separately; no backend fallback is accepted.
"""

import numpy as np
import pytest

from test_cloth_self_physics import fixture, run_window, assert_a, assert_b, assert_c

pytestmark = pytest.mark.tyx


def require_native_receipts(records, mode, *, terminal_nonzero):
    for index, record in enumerate(records):
        trial = record["replay"]["trial"]
        sub = trial.result.report.substeps[0]
        execution = sub.diagnostics["participants"]["cloth"]["execution"]
        assert execution["mode"] == mode
        assert execution["compute_dtype"] == "float32"
        assert execution["completed_launches"] == execution["launches"] > 0
        if mode == "cuda":
            assert execution["uploads"] > 0 and execution["downloads"] > 0
        terminal = record["replay"]["terminal"]
        assert terminal["applications"] == 1
        delta = terminal["post_execution_delta"]["cloth"]
        expected = 2 if terminal_nonzero and index == len(records)-1 else 0
        # One apply_node_loads and one axpy call in the existing public native
        # point operation. No additional response probe or second application.
        assert delta["completed_launches"] == delta["launches"] == expected
        for state in (record["selected_pre"], record["final_state"]):
            for value in (state.domains["cloth"].position, state.domains["cloth"].velocity):
                np.testing.assert_array_equal(value, value.astype(np.float32).astype(np.float64))
        assert record["native_fallback"] is False


@pytest.mark.parametrize("h", [1/64, 1/128, 1/256])
@pytest.mark.parametrize("material", [False, True])
@pytest.mark.parametrize("rebuild", [0, 1])
@pytest.mark.parametrize("variant", ["identity", "rigid", "exchange_cycle"])
def test_actual_a_full_pressure_window(tyx_mode, h, material, rebuild, variant, tmp_path):
    data = fixture("A", material=material, variant=variant)
    result = run_window(data, h, tmp_path, backend="tyx:"+tyx_mode)
    assert_a(data, result, h)
    require_native_receipts(result[2], tyx_mode, terminal_nonzero=False)


@pytest.mark.parametrize("h", [1/16, 1/32, 1/64])
@pytest.mark.parametrize("u,t,slip,delta", [(35/64, -1/4, 0., -3-105/512), (35/16, -1/2, 35/32, -3-315/128)])
@pytest.mark.parametrize("rebuild", [0, 1])
@pytest.mark.parametrize("variant", ["identity", "rigid", "exchange_cycle"])
def test_actual_b_terminal_stick_slip(tyx_mode, h, u, t, slip, delta, rebuild, variant, tmp_path):
    data = fixture("B", u=u, variant=variant)
    result = run_window(data, h, tmp_path, backend="tyx:"+tyx_mode)
    assert_b(data, result, t, slip, delta)
    require_native_receipts(result[2], tyx_mode, terminal_nonzero=True)


@pytest.mark.parametrize("rebuild", [0, 1])
@pytest.mark.parametrize("variant", ["identity", "rigid", "exchange_cycle"])
def test_actual_c_shared_base(tyx_mode, rebuild, variant, tmp_path):
    data = fixture("C", variant=variant)
    result = run_window(data, 1/64, tmp_path, backend="tyx:"+tyx_mode)
    assert_c(data, result)
    require_native_receipts(result[2], tyx_mode, terminal_nonzero=False)


def test_actual_bounded_cpp_terminal_control(tyx_mode, tmp_path):
    data = fixture("B", u=35/64)
    result = run_window(data, 1/16, tmp_path, backend="tyx:"+tyx_mode)
    assert_b(data, result, -1/4, 0., -3-105/512)
    require_native_receipts(result[2], tyx_mode, terminal_nonzero=True)


def test_actual_terminal_failure_preserves_original_and_retries(tyx_mode, monkeypatch, tmp_path):
    from phymeo.core.capabilities import SolverFailure
    from phymeo.core.scene import fingerprint
    from test_cloth_self_physics import configured, save_evidence
    runtime, program, state, inputs = configured(fixture("B"), 1/16, backend="tyx:"+tyx_mode)
    factory = program.base.participants["cloth"]
    create = factory.create_workspace
    events = []
    class FailAfterTerminal:
        def __init__(self, native):
            self.native = native
        def __getattr__(self, name):
            return getattr(self.native, name)
        def apply_velocity_impulses(self, *args):
            self.native.apply_velocity_impulses(*args)
            events.append("applied")
            raise SolverFailure("injected after actual public native terminal application")
        def close(self):
            self.native.close()
            events.append("closed")
    def wrapped(*args):
        events.append("created")
        return FailAfterTerminal(create(*args))
    monkeypatch.setattr(factory, "create_workspace", wrapped)
    original = fingerprint((state, inputs))
    failed = runtime.step(state, inputs, 1/16)
    save_evidence(tmp_path/"native-failure.json", result=failed, attempts=program.last_attempt_receipts, inputs=inputs, label="native terminal failure")
    assert not failed.report.accepted and "actual public native" in failed.report.reason
    assert events.count("applied") == 1
    assert events.count("created") == events.count("closed")
    assert fingerprint((state, inputs)) == original and fingerprint(failed.state) == fingerprint(state)
    monkeypatch.setattr(factory, "create_workspace", create)
    retry = runtime.step(state, inputs, 1/16)
    save_evidence(tmp_path/"native-retry.json", result=retry, attempts=program.last_attempt_receipts, inputs=inputs, label="native retry")
    runtime.close()
    fresh_runtime, fresh_program, fresh_state, fresh_inputs = configured(fixture("B"), 1/16, backend="tyx:"+tyx_mode)
    try:
        fresh = fresh_runtime.step(fresh_state, fresh_inputs, 1/16)
        save_evidence(tmp_path/"native-fresh.json", result=fresh, attempts=fresh_program.last_attempt_receipts, inputs=fresh_inputs, label="native fresh")
        assert retry.report.accepted and fresh.report.accepted
        assert fingerprint(retry.state) == fingerprint(fresh.state)
    finally:
        fresh_runtime.close()
