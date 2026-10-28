"""Real FP32 compute / FP64 storage for automatic two-stage normal contact."""

import numpy as np
import pytest
from test_dynamic_robot_normal_contact import build, step
from test_terminal_impulse_tyx import exact_reference

pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("chain", [False, True])
def test_actual_automatic_two_stage(tyx_mode, chain):
    case = build(tyx_mode, chain)
    assert case[3].capabilities.features == frozenset({"dynamic_robot_normal_split"})
    result = step(case)
    d = result.report.substeps[0].diagnostics["dynamic_robot_normal_contact"]
    x, q, vm, rv, vp, rp = exact_reference(chain)
    np.testing.assert_allclose(
        result.state.domains["nodes"].position[:, 1], list(map(float, x)), atol=2e-6, rtol=0
    )
    np.testing.assert_allclose(
        result.state.domains["nodes"].velocity[:, 1], list(map(float, vp)), atol=2e-6, rtol=0
    )
    np.testing.assert_allclose(result.state.domains["robot"].q, [float(q)], atol=2e-6, rtol=0)
    np.testing.assert_allclose(result.state.domains["robot"].v, [float(rp)], atol=2e-6, rtol=0)
    np.testing.assert_allclose(
        d["held_state"].domains["nodes"].velocity[:, 1], list(map(float, vm)), atol=2e-6, rtol=0
    )
    assert d["native_backend"] == "tyx:" + tyx_mode
    assert d["terminal_evidence"]["ledger"]["compute_precision"] == "float32"
    assert d["held_replays"] > 1 and d["terminal_replays"] > 1
    execution = result.report.substeps[0].diagnostics["execution"]
    assert execution["completed_launches"] > 0
    assert (
        sum(
            v.get("completed_launches", 0)
            for v in d["terminal_evidence"]["post_execution_delta"].values()
        )
        > 0
    )
    np.testing.assert_allclose(
        d["input_integrals"]["robot"]["effort"], [-d["held_impulses"][0]], atol=1e-12
    )


def test_actual_lifecycle_and_fixed_support(tyx_mode):
    from test_dynamic_robot_normal_physics import lifecycle, support

    lifecycle(tyx_mode)
    support(tyx_mode, 1 / 16)
    support(tyx_mode, 1 / 32)


def test_actual_finite_material_timestep_window(tyx_mode):
    from test_dynamic_robot_normal_physics import chain_window

    x, v = chain_window(tyx_mode, 1 / 32)
    assert x <= 0.013 and v <= 0.015


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


def test_actual_three_free_toi_endpoint_windows(tyx_mode):
    from test_dynamic_robot_normal_physics import impact

    for h in (0.25, 0.125, 0.0625):
        impact(tyx_mode, h)
