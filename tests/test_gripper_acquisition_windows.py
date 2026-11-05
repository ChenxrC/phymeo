"""Frozen acquisition experiments; physical failures remain ordinary failed gates."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "case",
    [
        "panda-sphere",
        "mimic-box",
        "mimic-zero-friction",
        "mimic-overload",
    ],
)
def test_frozen_acquisition_lift_and_release_windows(case):
    root = Path(__file__).parents[1]
    functions = runpy.run_path(str(root / "examples" / "robot_force_acquisition.py"))
    scene, controller, info = functions["make_case"](case)
    with scene:
        result = functions["run_case"](scene, controller, info)
    print({key: value for key, value in result.items() if key not in {"trace", "events"}})
    assert result["execution_failure"] is None, result["execution_failure"]
    assert result["accepted_steps"] > 0 and result["time_s"] > 0
    assert result["expected_physical_holding"] == (case in {"panda-sphere", "mimic-box"})
    assert all(result["checks"].values()), result["checks"]
