"""Offline cross-route checks of explicitly captured installed-route evidence."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from _solver_comparison import OBSERVATIONS

PHASES = ("initial", *(f"step{i}" for i in range(1, 7)),
          "forward", "masked", "reset", "replay")


@pytest.mark.parametrize("left,right", [("warp", "physolve"), ("physolve", "runtime")])
@pytest.mark.parametrize("nworld", [1, 8, 64])
def test_panda_cross_route_evidence(left, right, nworld):
    directory = os.environ.get("PHYMEO_COMPARISON_RESULTS")
    if directory is None:
        pytest.skip("explicit PHYMEO_COMPARISON_RESULTS required")
    root = Path(directory)
    records, failures = [], []
    for run in range(3):
        aopt = json.loads((root / f"b1j-{left}-n{nworld}-r{run}-options.json").read_text())
        bopt = json.loads((root / f"b1j-{right}-n{nworld}-r{run}-options.json").read_text())
        assert not any(isinstance(v, dict) and "unhandled_type" in v for v in aopt.values())
        assert aopt == bopt, "actual GPU options must match before numerical comparison"
        for phase in PHASES:
            with np.load(root / f"b1j-{left}-n{nworld}-r{run}-{phase}.npz") as a:
                with np.load(root / f"b1j-{right}-n{nworld}-r{run}-{phase}.npz") as b:
                    for field in OBSERVATIONS:
                        x, y = a[field], b[field]
                        assert x.shape == y.shape and x.dtype == y.dtype
                        assert np.isfinite(x).all() and np.isfinite(y).all()
                        row = dict(run=run, phase=phase, field=field,
                                   exact=bool(np.array_equal(x, y)),
                                   passed=bool(np.allclose(x, y, atol=1e-6, rtol=0)),
                                   max_abs=float(np.max(np.abs(x.astype(float) - y.astype(float))))
                                   if x.size else 0.0)
                        records.append(row)
                        if not row["passed"]:
                            failures.append(row)
    report = dict(left=left, right=right, nworld=nworld, comparisons=records, failures=failures)
    (root / f"b1j-cross-{left}-{right}-n{nworld}.json").write_text(
        json.dumps(report, indent=2) + "\n")
    assert not failures, json.dumps(failures, indent=2)
