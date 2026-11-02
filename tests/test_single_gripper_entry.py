"""Public example selects a single real robot and writes a reusable compressed trace."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def load_example():
    path = Path(__file__).resolve().parents[1] / "examples/compliant_cloth.py"
    spec = importlib.util.spec_from_file_location("single_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "mu", "accepted"), [("single-grip", 0.9, True), ("single-slip", 0.0, False)]
)
def test_cli_selects_fixed_single_arm_controls_and_compression(
    tmp_path, monkeypatch, name, mu, accepted
):
    mod = load_example()
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(
            case=name,
            completed=accepted,
            failure=None if accepted else "gripper lost the sheet to the table",
            failure_phase=None if accepted else "lift_A",
            time_s=0.52,
        )

    monkeypatch.setattr(mod, "run_single_gripper", run)
    out = tmp_path / "trajectory.json.gz"
    monkeypatch.setattr(sys, "argv", ["compliant_cloth.py", "--case", name, "--output", str(out)])
    assert mod.main() == 0
    with gzip.open(out, "rt", encoding="utf8") as src:
        data = json.load(src)
    assert data["completed"] == accepted
    (opts,) = calls
    assert opts["grip_friction"] == mu and opts["initial_tilt_deg"] == 2.0
    assert opts["finger_pad_radius_m"] == 0.002 and opts["finger_backing_clearance_m"] == 0.000125


def test_single_no_grip_does_not_count_a_kernel_failure_as_slip(tmp_path, monkeypatch):
    mod = load_example()
    monkeypatch.setattr(
        mod,
        "run_single_gripper",
        lambda **_: dict(
            case="single-slip",
            completed=False,
            failure_phase="lift_A",
            failure="backend failed",
            time_s=0.0,
        ),
    )
    monkeypatch.setattr(
        sys, "argv", ["example", "--case", "single-slip", "--output", str(tmp_path / "f.json")]
    )
    assert mod.main() == 1
