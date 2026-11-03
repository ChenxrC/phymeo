"""Future multi-hinge CCD gets a positive analytic target, alongside current static FK controls."""

import json
import math
import runpy
from pathlib import Path

import pytest

REFERENCE = Path(__file__).parent / "reference_newton" / "robot" / "two_hinge_ellipse.py"


def test_mirrored_two_hinge_pose_and_whole_interval_controls():
    reference = runpy.run_path(str(REFERENCE))
    result = reference["study_phy"]()
    first = result["first_t"]
    s = math.sin(math.pi * first)
    assert 2.5 - s / 2 - 2 * s**2 == pytest.approx(1 / 64, abs=2e-15)
    for sign in (1, -1):
        run = result["signs"][str(sign)]
        target, control = reference["targets"](sign)
        assert [row["occupied"][target] for row in run["rows"]] == [
            False, False, True, True, True, False, False
        ]
        assert min(row["distance_gaps_m"][control] for row in run["rows"]) == pytest.approx(7 / 8)
        assert run["rows"][3]["distance_gaps_m"][target] == pytest.approx(-1 / 8)
        assert max(row["center_error_m"] for row in run["rows"]) < 1e-12
        assert run["chord_status"] == "separated"
    print(json.dumps(result, allow_nan=False))
