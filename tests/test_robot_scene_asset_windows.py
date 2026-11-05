"""Real controlled preload/release through the same public multiobject scene."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("kind", ["panda", "mimic"])
def test_two_urdf_topologies_hold_release_and_support_two_live_bodies(kind):
    root = Path(__file__).parents[1]
    functions = runpy.run_path(str(root / "examples" / "urdf_multiobject_contact.py"))
    source = root / "tests" / "reference_newton" / "robot" / "mimic_jaws.urdf"
    scene, info = functions["make_case"](kind, source)
    with scene:
        result = functions["run_case"](scene, info)
    print({k: v for k, v in result.items() if k not in {"trace", "events"}})
    assert all(result["checks"].values()), result["checks"]
