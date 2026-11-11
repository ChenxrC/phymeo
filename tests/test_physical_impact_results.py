"""Keep old numeric assertions separate from new physical impact reporting."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from _impact_cases import CASES


def evidence():
    path = os.environ.get("PHYMEO_IMPACT_OUTPUT")
    if path is None:
        pytest.skip("explicit PHYMEO_IMPACT_OUTPUT required")
    return Path(path)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_impact_actual_options(name, nworld):
    root = evidence()
    values = [json.loads((root / f"b1k-{route}-{name}-n{nworld}-options.json").read_text())
              for route in ("warp", "physolve", "runtime")]
    assert all(value == values[0] for value in values)
    assert not any(isinstance(v, dict) and "unhandled_type" in v for v in values[0].values())


@pytest.mark.parametrize("route", ["warp", "physolve", "runtime"])
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_freefall_convergence(route, nworld):
    root = evidence()
    errors = []
    for name in ("fall", "fall_half"):
        prefix = root / f"b1k-{route}-{name}-n{nworld}"
        assert json.loads(Path(str(prefix) + "-result.json").read_text())["passed"]
        with np.load(str(prefix) + "-trace.npz") as data:
            errors.append(abs(data["qpos"][-1, :, 2] - (2 - .5 * 9.81 * .2**2)))
    ratio = errors[1] / errors[0]
    assert np.all((ratio > .4) & (ratio < .6)), ratio.tolist()


@pytest.mark.parametrize("left,right", [("warp", "physolve"), ("physolve", "runtime")])
@pytest.mark.parametrize("nworld", [1, 8, 64])
@pytest.mark.parametrize("name", ["panda_stress", "panda_motion"])
def test_panda_trajectory_numeric_gate(left, right, nworld, name):
    root = evidence()
    traces = []
    opts = []
    for route in (left, right):
        prefix = root / f"b1k-{route}-{name}-n{nworld}"
        assert json.loads(Path(str(prefix) + "-result.json").read_text())["passed"]
        opts.append(json.loads(Path(str(prefix) + "-options.json").read_text()))
        with np.load(str(prefix) + "-trace.npz") as data:
            traces.append({key: data[key] for key in ("qpos", "qvel", "qacc", "actions")})
    assert opts[0] == opts[1]
    if name == "panda_stress":
        np.testing.assert_array_equal(traces[0]["actions"], traces[1]["actions"])
    records = []
    for key in ("qpos", "qvel", "qacc"):
        x, y = (trace[key] for trace in traces)
        assert x.shape == y.shape and x.dtype == y.dtype
        assert np.isfinite(x).all() and np.isfinite(y).all()
        records.append(dict(field=key, max_abs=float(np.max(abs(x - y))),
                            passed=bool(np.allclose(x, y, atol=1e-6, rtol=0))))
    (root / f"b1k-numeric-{left}-{right}-{name}-n{nworld}.json").write_text(
        json.dumps(records, indent=2))
    assert all(row["passed"] for row in records), records
