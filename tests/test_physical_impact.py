"""Opt-in impact assessment; physical criteria do not replace numerical gates."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from _impact_cases import CASES, DT, STEPS, command, contacts, impact_route, observe
from _solver_comparison import INTEGRATION, OBSERVATIONS, options


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
@pytest.mark.parametrize("name", CASES)
def test_physical_impact(name, nworld):
    route = os.environ.get("PHYMEO_IMPACT_ROUTE")
    if route is None:
        pytest.skip("explicit PHYMEO_IMPACT_ROUTE required")
    assert route in ("warp", "physolve", "runtime")
    root = Path(os.environ["PHYMEO_IMPACT_CASES"])
    output = Path(os.environ["PHYMEO_IMPACT_OUTPUT"])
    protocol = json.loads((root / "protocol.json").read_text())
    for path, digest in protocol["files"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    case = root / name
    controls = dict(np.load(case / "controls.npz"))
    assert protocol["steps"][name] == STEPS[name] and protocol["dt"][name] == DT[name]
    records, failures = [], []
    states, contact_count, penetration, actions = [], [], [], []
    prefix = output / f"b1k-{route}-{name}-n{nworld}"
    phase, hard_failure = "prepare", None

    def check(label, actual, limit):
        row = dict(check=label, value=float(actual), limit=float(limit), passed=bool(actual <= limit))
        records.append(row)
        if not row["passed"]:
            failures.append(row)

    print(f"IMPACT_BEGIN {route} {name} N={nworld}", flush=True)
    try:
        with impact_route(route, nworld, name, case,
                          os.environ["PHYMEO_PHYSOLVE_SOURCES"]) as sim:
            opt = options(sim.model)
            assert not any(isinstance(v, dict) and "unhandled_type" in v for v in opt.values())
            Path(str(prefix) + "-options.json").write_text(json.dumps(opt, indent=2))
            initial = {k: getattr(sim.data, k).numpy().copy()
                       for k in dict.fromkeys(INTEGRATION + OBSERVATIONS)}
            current = observe(sim)
            states.append(current)
            for step in range(STEPS[name]):
                phase = f"step{step + 1}"
                previous_time = current["time"].copy()
                force = command(name, nworld, step, current["qpos"], current["qvel"], controls)
                sim.step(force)
                current = observe(sim)
                states.append(current)
                actions.append(force)
                if (step + 1) % 10 == 0:
                    counts, depths = contacts(sim, nworld)
                    contact_count.append(counts)
                    penetration.append(depths)
                # Per-step local ULP check detects reset/missing advancement;
                # cumulative budget separately allows FP32 time accumulation.
                increment = current["time"] - previous_time
                local_budget = 4 * np.maximum(np.spacing(abs(current["time"])),
                                               np.spacing(np.float32(DT[name])))
                assert np.all(increment > 0)
                assert np.all(abs(increment - DT[name]) <= local_budget)
                check_time = abs(current["time"] - (step + 1) * DT[name])
                assert np.max(check_time) <= max(1e-6, (step + 1)**2 * DT[name] * 1.2e-7)
            trajectory = {k: np.stack([s[k] for s in states]) for k in current}
            phase = "physical_criteria"
            q, v = trajectory["qpos"], trajectory["qvel"]
            if name.startswith("fall"):
                steps = np.arange(STEPS[name] + 1)[:, None]
                expected = 2 - 9.81 * DT[name]**2 * steps * (steps + 1) / 2
                check("Euler_position_m", np.max(abs(q[:, :, 2] - expected)), 3e-5)
                check("velocity_m_per_s", np.max(abs(v[:, :, 2] + 9.81 * DT[name] * steps)), 2e-5)
                check("lateral_motion_m", np.max(abs(q[:, :, :2])), 1e-6)
            elif name == "slider":
                expected_q, expected_v = np.zeros((nworld, 1)), np.zeros((nworld, 1))
                errors = []
                for i, force in enumerate(actions):
                    expected_v += DT[name] * force.astype(float) / 2
                    expected_q += DT[name] * expected_v
                    errors.append(max(np.max(abs(q[i + 1] - expected_q)),
                                      np.max(abs(v[i + 1] - expected_v))))
                check("constant_force_discrete_q_v", max(errors), 1e-6)
            elif name == "balance":
                check("gravity_balance_angle_rad", np.max(abs(q)), 1e-4)
                check("gravity_balance_speed_rad_per_s", np.max(abs(v)), .001)
            elif name == "support":
                # Existing P2PE physical reference: final half-second after 3.5s.
                start = round(3.5 / DT[name]) + 1
                check("settled_speed_m_per_s", np.linalg.norm(v[start:, :, :3], axis=-1).max(), .001)
                check("penetration_m", max(0, .1 - q[start:, :, 2].min()), .002)
                support = trajectory["qfrc_constraint"][start:, :, :3].mean(axis=0)
                check("support_force_error_N", np.max(abs(support - [0, 0, 19.62])), .1962)
                check("support_lateral_force_N", np.linalg.norm(support[:, :2], axis=-1).max(), .1962)
                assert np.all(np.asarray(contact_count)[-50:] > 0)
            else:
                # Impact labels, not new acceptance tolerances, are evaluated
                # offline against CPU and other routes from the complete trace.
                check("finite_trajectory", 0, 0)
            # Check real Runtime reset separately from native diagnostic restore.
            before = {k: getattr(sim.data, k).numpy().copy()
                      for k in dict.fromkeys(INTEGRATION + OBSERVATIONS)}
            mask = np.arange(nworld) % 2 == 0
            phase = "masked_reset"
            sim.reset(mask)
            sim.complete()
            for key in dict.fromkeys(INTEGRATION + OBSERVATIONS):
                value = getattr(sim.data, key).numpy()
                np.testing.assert_array_equal(value[~mask], before[key][~mask], err_msg=key)
                if key in INTEGRATION:
                    np.testing.assert_array_equal(value[mask], initial[key][mask], err_msg=key)
            phase = "full_reset"
            sim.reset()
            sim.complete()
            for key in INTEGRATION:
                np.testing.assert_array_equal(getattr(sim.data, key).numpy(), initial[key], err_msg=key)
            phase = "close"
    except Exception as error:
        hard_failure = dict(phase=phase, type=type(error).__name__, message=str(error))
        raise
    finally:
        if states:
            np.savez(str(prefix) + "-trace.npz",
                     **{k: np.stack([s[k] for s in states]) for k in states[0]},
                     actions=np.asarray(actions), contact_count=np.asarray(contact_count),
                     penetration=np.asarray(penetration))
        Path(str(prefix) + "-result.json").write_text(json.dumps(
            dict(route=route, case=name, nworld=nworld, steps_completed=len(actions),
                 checks=records, failures=failures, hard_failure=hard_failure,
                 passed=hard_failure is None and not failures and len(actions) == STEPS[name]),
            indent=2))
    assert not failures, json.dumps(failures)
