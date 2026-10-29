"""Actual native mode, launch, FP32 and automatic-law checks."""

import numpy as np
import pytest
from test_dynamic_robot_friction_contact import build, diagnostics, step
from test_dynamic_robot_friction_physics import cross_case, lifecycle, window

pytestmark = pytest.mark.tyx


def assert_native(result, mode):
    sub = result.report.substeps[-1]
    d = diagnostics(result)
    assert d["native_backend"] == "tyx:" + mode
    assert d["terminal_evidence"]["ledger"]["compute_precision"] == "float32"
    assert sub.diagnostics["execution"]["completed_launches"] > 0
    assert sub.diagnostics["execution"]["compute_dtype"] == "float32"
    for state in (d["held_state"], result.state):
        for name, fields in (("robot", ("q", "v")), ("nodes", ("position", "velocity"))):
            for field in fields:
                a = getattr(state.domains[name], field)
                assert a.dtype == np.float64
                np.testing.assert_array_equal(a, a.astype(np.float32).astype(np.float64))
    np.testing.assert_allclose(
        d["input_integrals"]["nodes"]["force"],
        d["held_world"].reshape(-1, 3)
        if len(d["row_keys"]) == len(result.state.domains["nodes"].position)
        else d["held_world"].sum(axis=0, keepdims=True),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("mu", [2.0, 0.25])
def test_actual_support(tyx_mode, mu):
    result = step(build(tyx_mode, mu=mu))
    assert_native(result, tyx_mode)
    x, v = (0.0, 0.0) if mu == 2.0 else (3 / 64, 3 / 16)
    np.testing.assert_allclose(
        result.state.domains["nodes"].position, [[x, 0, 0]], atol=2e-6, rtol=0
    )
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity, [[v, 0, 0]], atol=2e-6, rtol=0
    )


def test_actual_material_impact(tyx_mode):
    result = step(build(tyx_mode, mu=0.5, chain=True))
    assert_native(result, tyx_mode)
    np.testing.assert_allclose(
        result.state.domains["nodes"].position,
        [[1 / 4, -3 / 56, 0], [1 / 4, 51 / 56, 0]],
        atol=2e-6,
        rtol=0,
    )
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity,
        [[13 / 21, -8 / 21, 0], [1, -6 / 7, 0]],
        atol=2e-6,
        rtol=0,
    )
    assert (
        sum(
            v.get("completed_launches", 0)
            for v in diagnostics(result)["terminal_evidence"]["post_execution_delta"].values()
        )
        > 0
    )


def test_actual_two_planes_and_lifecycle(tyx_mode):
    result = step(cross_case(tyx_mode), 1.0)
    assert_native(result, tyx_mode)
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity, [[-17 / 25, 56 / 25, 0]], atol=2e-5, rtol=0
    )
    for result in lifecycle(tyx_mode):
        assert_native(result, tyx_mode)


@pytest.mark.parametrize("mu", [2.0, 0.25])
@pytest.mark.parametrize("h", [1 / 16, 1 / 32])
def test_actual_half_second_window(tyx_mode, mu, h):
    result, _ = window(tyx_mode, mu, h, 0.5)
    assert_native(result, tyx_mode)


@pytest.mark.parametrize("domain", ["robot", "nodes"])
def test_actual_fp64_nextafter_native_pre_state_is_rejected(tyx_mode, domain):
    import phymeo as ph

    class Program:
        def __init__(self, base):
            self.base = base

        def __getattr__(self, name):
            return getattr(self.base, name)

        def create_workspace(self, *args):
            return Workspace(self.base.create_workspace(*args))

        def close(self):
            self.base.close()

    class Workspace:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

        def result(self):
            result = self.native.result()
            state = result.domains[domain]
            velocity = state.v.copy() if domain == "robot" else state.velocity.copy()
            if np.any(velocity != 0):
                index = tuple(np.argwhere(velocity != 0)[0])
                velocity[index] = np.nextafter(velocity[index], np.inf)
                changed = (
                    ph.RobotState(state.q, velocity)
                    if domain == "robot"
                    else ph.ParticleState(state.position, velocity)
                )
                return ph.SceneState({**result.domains, domain: changed}, result.history)
            return result

    case = build(tyx_mode, base_transform=Program)
    case[1].domains["nodes"].position[0, 1] = 1.0
    case[2].domains["robot"].effort[0] = 0.25
    with ph.SceneStepper(case[0], case[3], case[4], case[5], case[3].registry) as runtime:
        result = runtime.step(case[1], case[2], 0.25)
    assert not result.report.accepted, (
        "FP64 display perturbations cannot pass as actual FP32 output"
    )
    assert "FP32" in result.report.reason or "float32" in result.report.reason


def test_actual_all_native_trials_counted_once(tyx_mode):
    from test_terminal_impulse import ObservedProgram, ObservedWorkspace

    class Program(ObservedProgram):
        def create_workspace(self, *args):
            w = Workspace(self, self.base.create_workspace(*args), len(self.workspaces) + 1)
            self.workspaces.append(w)
            return w

    class Workspace(ObservedWorkspace):
        def close(self):
            stats = [p.engine.stats() for p in self.native.participants.values()]
            self.actual = {
                key: sum(s[key] for s in stats)
                for key in ("launches", "completed_launches", "uploads", "downloads")
            }
            super().close()

    c = build(tyx_mode, mu=0.5, chain=True, base_transform=Program)
    result = step(c)
    d = diagnostics(result)
    assert len(c[3].base.workspaces) == d["held_replays"] + d["terminal_replays"]
    actual = result.report.substeps[0].diagnostics["execution"]
    for key in ("launches", "completed_launches", "uploads", "downloads"):
        assert actual[key] == sum(w.actual.get(key, 0) for w in c[3].base.workspaces)
    assert all(
        w.counts["predict"] == w.counts["terminal"] == w.counts["close"] == 1
        for w in c[3].base.workspaces
    )
