"""Fixed D4-N windows and independent stream/gate negative controls.

The four complete physical windows are not shortened for pytest. Synthetic
statistics below test only the observation validator, never physical success.
"""

import gzip
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest


@lru_cache(maxsize=1)
def runner():
    directory = Path(__file__).parents[2] / "examples"
    for name, filename in (
        ("deformable_normal_load_cases", "deformable_normal_load_cases.py"),
        ("deformable_impulse_audit", "deformable_impulse_audit.py"),
        ("d4_normal_load_runner", "run_deformable_normal_loads.py"),
    ):
        path = directory / filename
        existing = sys.modules.get(name)
        if existing is not None:
            assert Path(existing.__file__).resolve() == path.resolve()
            continue
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    result = sys.modules["d4_normal_load_runner"]
    result.load_reference()
    return result


def test_fixed_declarations_and_control_boundaries():
    reference = runner().ref
    total = 0
    for name in reference.NAMES:
        case = reference.make_case(name)
        total += case["steps"]
        np.testing.assert_array_equal(
            case["node"].initial_state.position,
            ((0, 0, 0), (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)),
        )
        np.testing.assert_array_equal(case["weights"], (2, 0, 0, 0, 0))
        np.testing.assert_array_equal(case["faces"], ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)))
        np.testing.assert_array_equal(case["edges"], ((0, 1), (0, 2), (0, 3), (0, 4)))
        assert case["mass"] == 2 and case["inertia"] == 0.25
        assert case["initial_root"].q[2] == (41 / 128 if name == "n2_capsule" else 25 / 128)
        np.testing.assert_array_equal(case["initial_root"].v, (0, 0, -1 / 16, 0, 0, 0))
        assert case["steps"] * case["h"] == 3
        assert np.all(case["compliances"] == (2 if name == "n3_soft" else 1) / 16384)
        lo, hi = int(1 / case["h"]), int((9 / 8) / case["h"])
        for index, force in (
            (0, -16),
            (lo - 1, -16),
            (lo, 16),
            (hi - 1, 16),
            (hi, -16),
            (case["steps"] - 1, -16),
        ):
            np.testing.assert_array_equal(
                reference.load_at_step(case, index), (0, 0, force, 0, 0, 0)
            )
        assert int(reference.EVENT_DURATION / case["h"]) == (16 if name == "n1_fine" else 8)
    assert total == 15360


@pytest.mark.parametrize("name", ("n1_sphere", "n2_capsule"))
def test_independent_initial_reference_has_all_four_material_owners(name):
    reference = runner().ref
    case = reference.make_case(name)
    initial = reference.initial_reference(case)
    force, wrench, rows = reference.reference_load(case, *initial)
    assert len(rows) == 4 and len({row["identity"] for row in rows}) == 4
    for row in rows:
        assert row["gap"] == 9 / 128
        np.testing.assert_array_equal(row["bary"], (1, 0, 0))
        np.testing.assert_array_equal(row["normal"], (0, 0, -1))
        assert row["side"] == 25 / 128
    np.testing.assert_array_equal(force, 0)
    np.testing.assert_array_equal(wrench, 0)
    external = reference.load_at_step(case, 0)
    continued, _ = reference.continue_reference(case, initial, external)
    x, v, q, speed = continued
    np.testing.assert_allclose(x, initial[0], atol=1e-14, rtol=0)
    np.testing.assert_allclose(v, initial[1], atol=1e-12, rtol=0)
    expected = initial[3].copy()
    expected[2] -= 8 * case["h"]
    np.testing.assert_allclose(speed, expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(q[:3], initial[2][:3] + case["h"] * expected[:3], atol=1e-14, rtol=0)


def synthetic_gate_state(module):
    """Validator-only fabricated counters; this is never a physical fixture."""
    case = module.ref.make_case("n1_sphere")
    observed = module.Observations(case)
    observed.steps = case["steps"]
    observed.minimum_gap, observed.minimum_side, observed.peak = 1 / 32, 5 / 32, 16
    times = {
        "initial_outside": (0, 1 / 128),
        "entry": (0.25, 0.25 + 1 / 128),
        "release": (1, 1 + 1 / 128),
        "recontact": (2, 2 + 1 / 128),
    }
    for name, (begin, end) in times.items():
        observed.events[name].update(
            maximum=observed.minimum, qualified_start=begin, qualified_end=end
        )
    for row in observed.windows:
        count = int((row["end"] - row["begin"]) / case["h"])
        row.update(count=count, force_sum=16 * count, rigid_speed2=0.0, node_speed2=0.0)
    return observed


@pytest.mark.parametrize("bad", ("missing-recontact", "rigid-rms", "center-rms", "mean-force"))
def test_window_statistics_cannot_replace_missing_physical_observations(bad):
    module = runner()
    observed = synthetic_gate_state(module)
    observed.finish()  # Only establishes the validator-negative starting state.
    if bad == "missing-recontact":
        observed.events["recontact"]["maximum"] = 0
    elif bad == "rigid-rms":
        observed.windows[1]["rigid_speed2"] = 0.2**2 * observed.windows[1]["count"]
    elif bad == "center-rms":
        observed.windows[1]["node_speed2"] = 0.2**2 * observed.windows[1]["count"]
    else:
        observed.windows[1]["force_sum"] = 32 * observed.windows[1]["count"]
    with pytest.raises(AssertionError):
        observed.finish()


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("n1_sphere", "n2_capsule", "n3_soft", "n1_fine"))
def test_complete_supported_membrane_window(name, tyx_mode, tmp_path):
    summary = runner().run_window(name, tyx_mode, tmp_path / "window")
    assert summary["status"] == "PASS"
    assert summary["progress"]["steps_checked"] == (6144 if name == "n1_fine" else 3072)
    assert summary["progress"]["worldsteps"] == summary["planned_steps"]
    assert summary["progress"]["close_returned"] == 1
    assert summary["trace"]["logical_bytes"] > 0


@pytest.mark.tyx
def test_post_return_oracle_failure_keeps_actual_committed_prefix(tyx_mode, tmp_path, monkeypatch):
    module = runner()
    original = module.audit_step

    def after_real_audit(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("D4 test-only observer failed after real accepted step")

    output = tmp_path / "observer-failure"
    with monkeypatch.context() as patch:
        patch.setattr(module, "audit_step", after_real_audit)
        with pytest.raises(RuntimeError, match="observer failed after real accepted"):
            module.run_window("n1_sphere", tyx_mode, output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert summary["progress"]["steps_attempted"] == summary["progress"]["steps_returned"] == 1
    assert summary["progress"]["steps_checked"] == 0
    assert summary["progress"]["worldsteps"] == 1
    assert summary["progress"]["close_returned"] == 1
    with gzip.open(output / "normal-load.jsonl.gz", "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert [row["ordinal"] for row in records] == list(range(len(records)))
    events = [row["event"] for row in records]
    assert events.index("step-attempt") < events.index("step-return") < events.index("failure")
    returned = next(row for row in records if row["event"] == "step-return")
    receipt = dict(returned["payload"]["mapping_entries"])["receipt"]
    assert "query" in receipt and "node_receipt" in receipt and "rigid_publication" in receipt
    state = dict(receipt["states"]["mapping_entries"])[7]
    assert state["node"]["tick"] == state["rigid"]["tick"] == 1
    # The physics committed before this observer failure. No rollback is claimed.
