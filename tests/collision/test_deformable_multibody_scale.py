"""M-B1 capacity/mixed-layout qualification; four predeclared backend nodes.

S8 keeps original A physics in four independent worlds (7,31,43,97), with
8 participants total. M3 keeps original A body 19 in world7 and both bodies in
world31. Each case executes four actual uninterrupted steps and all original
per-body, material, source and continuous-reference gates. No product changes.
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "deformable_multibody_scale_consumer",
    Path(__file__).parents[1] / "installed" / "check_deformable_multibody_scale.py",
)
case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(case)
pytestmark = pytest.mark.tyx


@pytest.mark.parametrize("identity", ("S8", "M3"))
def test_full_capacity_and_mixed_worlds_continue_original_physics(identity, tyx_mode, tmp_path):
    case.base.initialize()
    with case.base.saved(tmp_path / (identity + ".jsonl.gz")) as journal:
        result = case.run_case(identity, tyx_mode, journal)
    expected = (16, 32) if identity == "S8" else (8, 12)
    assert (result["worldsteps"], result["participant_steps"]) == expected
    assert result["steps"] == 4
