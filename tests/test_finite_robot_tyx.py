"""Actual selected TYX backend, FP32 compute and FP64 native storage controls."""

import numpy as np
import pytest
from test_finite_robot_contact import build, platform_window, step
from test_finite_robot_physics import rotor_expected, rotor_window
from test_terminal_impulse import ObservedProgram, ObservedWorkspace

import phymeo as ph

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("chain", [False, True])
def test_actual_platform_and_material(tyx_mode, chain):
    case = build(tyx_mode, chain=chain)
    result = step(case)
    d = result.report.substeps[0].diagnostics["dynamic_robot_finite_contact"]
    expected_p = 3 / 7 if chain else 1 / 3
    np.testing.assert_allclose(
        d["held_impulses"], [expected_p] + ([0.0] if chain else []), atol=2e-5, rtol=0
    )
    np.testing.assert_allclose(
        d["terminal_impulses"], [1 / 3] + ([0.0] if chain else []), atol=2e-5, rtol=0
    )
    np.testing.assert_allclose(
        result.state.domains["robot"].q, [-3 / 56 if chain else -1 / 24], atol=2e-5, rtol=0
    )
    np.testing.assert_allclose(
        result.state.domains["robot"].v, [-8 / 21 if chain else -1 / 3], atol=2e-5, rtol=0
    )
    assert result.report.backend == "tyx:" + tyx_mode
    assert d["terminal_ledger"]["ledger"]["compute_precision"] == "float32"
    assert result.report.substeps[0].diagnostics["execution"]["completed_launches"] > 0


@pytest.mark.parametrize("gain", [False, True])
def test_actual_rotor_and_energy_rejection(tyx_mode, gain):
    case = build(tyx_mode, rotor=True, gain=gain)
    result = step(case, 1.0, not gain)
    if gain:
        assert "energy" in result.report.reason
        return
    theta, V, ell, u, eta, R = rotor_expected()
    d = result.report.substeps[0].diagnostics["dynamic_robot_finite_contact"]
    np.testing.assert_allclose(d["held_impulses"], [1 / 16], atol=2e-5, rtol=0)
    np.testing.assert_allclose(d["terminal_impulses"], [eta], atol=2e-5, rtol=0)
    np.testing.assert_allclose(result.state.domains["robot"].q, [0.25 + theta], atol=2e-5, rtol=0)
    np.testing.assert_allclose(
        result.state.domains["robot"].v, [theta - ell * eta], atol=2e-5, rtol=0
    )
    assert d["axis_balance"]["residual_nms"] <= 1e-5
    assert d["ccd"][0].status == "certified"


def test_actual_finite_lifecycle(tyx_mode):
    platform_window(tyx_mode, 0.25)


@pytest.mark.parametrize("count", [2, 4])
def test_actual_rotor_short_refinement(tyx_mode, count):
    first = rotor_window(tyx_mode, count)
    second = rotor_window(tyx_mode, count)
    np.testing.assert_array_equal(first.domains["robot"].q, second.domains["robot"].q)
    np.testing.assert_array_equal(first.domains["nodes"].position, second.domains["nodes"].position)


def test_actual_native_storage_and_counts(tyx_mode):
    case = build(tyx_mode, base_transform=ObservedProgram)
    result = step(case)
    d = result.report.substeps[0].diagnostics["dynamic_robot_finite_contact"]
    observed = case[3].base
    assert len(observed.workspaces) == d["held_replays"] + d["terminal_replays"]
    assert all(
        w.counts["terminal"] == w.counts["predict"] == w.counts["close"] == 1
        for w in observed.workspaces
    )
    for state in (d["held_state"], result.state):
        for name, fields in (("robot", ("q", "v")), ("nodes", ("position", "velocity"))):
            for field in fields:
                value = getattr(state.domains[name], field)
                assert value.dtype == np.float64
                np.testing.assert_array_equal(value, value.astype(np.float32).astype(np.float64))
    assert d["terminal_ledger"]["applications"] == 1
    assert (
        sum(
            v.get("completed_launches", 0)
            for v in d["terminal_ledger"]["post_execution_delta"].values()
        )
        > 0
    )


@pytest.mark.parametrize("domain", ["robot", "nodes"])
def test_actual_one_ulp_host_native_change_rejects(tyx_mode, domain):
    class Program(ObservedProgram):
        def create_workspace(self, *args):
            ws = Workspace(self, self.base.create_workspace(*args), len(self.workspaces) + 1)
            self.workspaces.append(ws)
            return ws

    class Workspace(ObservedWorkspace):
        def result(self):
            state = self.native.result()
            original = state.domains[domain]
            v = (original.v if domain == "robot" else original.velocity).copy()
            if np.any(v):
                index = tuple(np.argwhere(v != 0)[0])
                v[index] = np.nextafter(v[index], np.inf)
                changed = (
                    ph.RobotState(original.q, v)
                    if domain == "robot"
                    else ph.ParticleState(original.position, v)
                )
                return ph.SceneState({**state.domains, domain: changed}, state.history)
            return state

    result = step(build(tyx_mode, base_transform=Program), accepted=False)
    assert "FP32" in result.report.reason
