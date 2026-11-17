"""Optional installed-route comparison; explicit selection never silently skips."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from _solver_comparison import INTEGRATION, OBSERVATIONS, open_route, options, snapshot


@pytest.mark.cuda
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_panda_native_repeatability(nworld):
    route = os.environ.get("PHYMEO_COMPARISON_ROUTE")
    if route is None:
        pytest.skip("explicit PHYMEO_COMPARISON_ROUTE required for installed comparison")
    assert route in ("warp", "physolve", "runtime")
    case = Path(os.environ["PHYMEO_COMPARISON_CASE"])
    evidence = Path(os.environ["PHYMEO_COMPARISON_OUTPUT"])
    evidence.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((case / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((case / name).read_bytes()).hexdigest() == digest
    assert manifest["comparison_atol"] == 1e-6 and manifest["comparison_rtol"] == 0
    assert manifest["repetitions"] == 3 and manifest["steps"] == 6
    force = np.load(case / f"forces-n{nworld}.npy")
    failures, comparisons, runs = [], [], []
    core = ("qpos", "qvel", "time", "ctrl", "qfrc_applied")

    def check(phase, actual, expected, fields=OBSERVATIONS, exact=()):
        for field in fields:
            a, b = actual[field], expected[field]
            assert a.shape == b.shape and a.dtype == b.dtype
            assert np.isfinite(a).all() and np.isfinite(b).all()
            equal = bool(np.array_equal(a, b))
            passed = equal if field in exact else bool(np.allclose(a, b, atol=1e-6, rtol=0))
            row = dict(phase=phase, field=field, passed=passed, exact=equal,
                       exact_required=field in exact,
                       max_abs=float(np.max(np.abs(a.astype(float) - b.astype(float))))
                       if a.size else 0.0)
            comparisons.append(row)
            if not passed:
                failures.append(row)

    def save(run, phase, instance):
        values = snapshot(instance)
        np.savez(evidence / f"b1j-{route}-n{nworld}-r{run}-{phase}.npz", **values)
        return values

    for run in range(3):
        print(f"COMPARISON_BEGIN {route} N={nworld} repetition={run}", flush=True)
        states = {}
        with open_route(route, nworld, case, os.environ.get("PHYMEO_PHYSOLVE_SOURCES")) as sim:
            opt = options(sim.model)
            assert not any(isinstance(v, dict) and "unhandled_type" in v for v in opt.values())
            (evidence / f"b1j-{route}-n{nworld}-r{run}-options.json").write_text(
                json.dumps(opt, indent=2) + "\n")
            states["initial"] = save(run, "initial", sim)
            for step in range(6):
                sim.step(force)
                states[f"step{step + 1}"] = save(run, f"step{step + 1}", sim)
            assert np.all(np.any(states["step6"]["qvel"] != states["initial"]["qvel"], axis=1))
            np.testing.assert_allclose(states["step6"]["time"], .002 * 6, atol=1e-6, rtol=0)
            sim.forward()
            states["forward"] = save(run, "forward", sim)
            check(f"r{run}/forward_state", states["forward"], states["step6"],
                  fields=("qpos", "qvel", "time"), exact=("qpos", "qvel", "time"))
            mask = np.arange(nworld) % 2 == 0
            sim.reset(mask)
            states["masked"] = save(run, "masked", sim)
            check(f"r{run}/masked_unselected",
                  {k: states["masked"][k][~mask] for k in OBSERVATIONS},
                  {k: states["forward"][k][~mask] for k in OBSERVATIONS}, exact=OBSERVATIONS)
            check(f"r{run}/masked_selected",
                  {k: states["masked"][k][mask] for k in OBSERVATIONS},
                  {k: states["initial"][k][mask] for k in OBSERVATIONS}, exact=core)
            check(f"r{run}/masked_integration",
                  {k: states["masked"][k][mask] for k in INTEGRATION},
                  {k: states["initial"][k][mask] for k in INTEGRATION},
                  fields=INTEGRATION, exact=INTEGRATION)
            sim.reset()
            states["reset"] = save(run, "reset", sim)
            check(f"r{run}/full_reset", states["reset"], states["initial"], exact=core)
            check(f"r{run}/full_integration", states["reset"], states["initial"],
                  fields=INTEGRATION, exact=INTEGRATION)
            for step in range(6):
                sim.step(force)
                replay = save(run, f"replay{step + 1}", sim)
            states["replay"] = replay
            np.savez(evidence / f"b1j-{route}-n{nworld}-r{run}-replay.npz", **replay)
            check(f"r{run}/replay", states["replay"], states["step6"],
                  fields=("qpos", "qvel", "time"), exact=("qpos", "qvel", "time"))
        if runs:
            for phase in states:
                check(f"r{run}-r0/{phase}", states[phase], runs[0][phase])
        runs.append(states)
    report = dict(route=route, nworld=nworld, comparisons=comparisons, failures=failures)
    (evidence / f"b1j-{route}-n{nworld}-results.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print("COMPARISON_RESULT " + json.dumps(dict(route=route, nworld=nworld,
                                                failed_fields=len(failures))), flush=True)
    assert not failures, json.dumps(failures, indent=2)
