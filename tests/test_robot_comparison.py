"""The optional external comparison must fail closed and preserve model semantics."""

import importlib.util
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from test_robot_kinematics import branched_model

SCRIPT = Path(__file__).parents[1] / "scripts/compare_robot_mujoco.py"


def comparison():
    spec = importlib.util.spec_from_file_location("robot_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oracle_xml_preserves_non_depth_first_tree_and_explicit_inertia():
    model = branched_model()
    xml = comparison().model_xml(model, dt=1 / 2000)
    root = ET.fromstring(xml)
    for i, parent in enumerate(model.parents):
        body = root.find(f".//body[@name='link{i}']")
        assert body is not None
        if parent != -1:
            assert root.find(f".//body[@name='link{parent}']/body[@name='link{i}']") is body
        np.testing.assert_allclose(
            np.fromstring(body.attrib["quat"], sep=" "), model.joint_orientation[i][[3, 0, 1, 2]]
        )
        inertia = body.find("inertial")
        np.testing.assert_allclose(
            np.fromstring(inertia.attrib["diaginertia"], sep=" "), model.inertia_diagonal[i]
        )
    assert root.find("option").attrib["integrator"] == "Euler"
    assert root.find("option/flag").attrib["contact"] == "disable"


@pytest.mark.parametrize(
    "actual", [np.array([np.nan]), np.array([np.inf]), np.array([1.0]), np.array([0.0, 0.0])]
)
def test_comparison_never_accepts_bad_result(actual):
    with pytest.raises(AssertionError):
        comparison().checked_error(actual, np.array([0.0]))


def test_comparison_requires_the_pinned_optional_dependency(tmp_path):
    # Core environment deliberately has no MuJoCo; the command must not skip/pass.
    if importlib.util.find_spec("mujoco") is not None:
        pytest.skip("this failure-mode test requires a core-only environment")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(tmp_path / "oracle")],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "optional MuJoCo 3.3.7 environment is required" in (result.stdout + result.stderr)
    assert not (tmp_path / "oracle" / "comparison.json").exists()
