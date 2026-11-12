"""D6-B declarations and observer contracts; no pytest test launches a 3 s run.

Fabricated phase/statistics records exercise validators only, never establish
physical qualification. Full windows require the explicit public runner CLI.
"""
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@lru_cache(maxsize=1)
def runner():
    directory = Path(__file__).parents[2] / "examples"
    for name, filename in (
        ("deformable_normal_load_cases", "deformable_normal_load_cases.py"),
        ("deformable_impulse_audit", "deformable_impulse_audit.py"),
        ("deformable_damped_load_cases", "deformable_damped_load_cases.py"),
        ("d6b_damped_load_runner", "run_deformable_damped_loads.py"),
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
    result = sys.modules["d6b_damped_load_runner"]
    result.load_reference()
    return result


def test_only_explicit_radial_dissipation_changes_the_four_declarations():
    module = runner()
    reference = module.ref
    assert reference.NAMES == (
        "d6b_n1_sphere_c4096", "d6b_n2_capsule_c4096",
        "d6b_n3_soft_c4096", "d6b_n1_fine_c4096",
    )
    assert reference.COEFFICIENT == 4096
    steps = 0
    for name, original_name in zip(reference.NAMES, reference.base.NAMES, strict=True):
        new = reference.make_case(name)
        old = reference.base.make_case(original_name)
        steps += new["steps"]
        assert new["base_name"] == original_name and old["node"].dissipation is None
        assert new["steps"] * new["h"] == 3
        left, right = reference.base.manifest(new), reference.base.manifest(old)
        # The original public manifest is otherwise byte-for-byte unchanged.
        left = dict(left, name=original_name, node=old["node"])
        assert module.encode(left) == module.encode(right)
        for field in ("model", "initial_state", "constraints", "point_ids", "surfaces"):
            assert module.encode(getattr(new["node"], field)) == module.encode(getattr(old["node"], field))
        phase = new["node"].dissipation
        np.testing.assert_array_equal(phase.ordered_point_ids, old["ids"][old["edges"]])
        np.testing.assert_array_equal(phase.coefficients_ns_per_m, (4096,) * 4)
        assert not phase.ordered_point_ids.flags.writeable
        assert not phase.coefficients_ns_per_m.flags.writeable
        for step in (0, int(1 / new["h"]) - 1, int(1 / new["h"]), int(9 / 8 / new["h"]), new["steps"] - 1):
            np.testing.assert_array_equal(reference.load_at_step(new, step), reference.base.load_at_step(old, step))
    assert steps == 15360
    with pytest.raises(ValueError, match="fixed damped"):
        reference.make_case("n1_sphere")


def test_independent_damped_reference_keeps_flat_null_mode_and_uses_its_own_state():
    reference = runner().ref
    case = reference.make_case(reference.NAMES[0])
    initial = reference.initial_reference(case)
    velocity = np.zeros((5, 3))
    velocity[0, 2] = -1 / 16
    np.testing.assert_array_equal(reference.ordered_velocity(case["node"], initial[0], velocity, case["h"]), velocity)
    bent = initial[0].copy()
    bent[0, 2] = -1 / 8
    damped = reference.ordered_velocity(case["node"], bent, velocity, case["h"])
    assert np.linalg.norm(damped[0]) < np.linalg.norm(velocity[0])
    np.testing.assert_array_equal(damped[1:], 0)
    # This reference input is its own state, never a returned product state.
    state = (bent, damped, initial[2], initial[3])
    external = reference.load_at_step(case, 0)
    force, wrench, _ = reference.reference_load(case, *state)
    y, reconstructed, lam = reference.material_step(case, bent, damped, force, case["h"])
    expected_velocity = reference.ordered_velocity(case["node"], y, reconstructed, case["h"])
    next_state, audit = reference.continue_reference(case, state, external)
    np.testing.assert_array_equal(next_state[0], y)
    np.testing.assert_array_equal(next_state[1], expected_velocity)
    np.testing.assert_array_equal(audit["reconstructed_velocity"], reconstructed)
    np.testing.assert_array_equal(audit["multipliers"], lam)
    expected_q, expected_v = reference.rigid_step(case, state[2], state[3], wrench, case["h"], external)
    np.testing.assert_array_equal(next_state[2], expected_q)
    np.testing.assert_array_equal(next_state[3], expected_v)


def phase_record():
    """Exact one-edge validator fixture, not a physical run or copied output."""
    def own(value, dtype=float):
        return np.frombuffer(np.asarray(value, dtype).tobytes(), dtype=dtype).reshape(np.shape(value))
    spec = SimpleNamespace(
        point_ids=own((11, 23), np.int64),
        constraints=SimpleNamespace(indices=own(((0, 1),), np.int64)),
        model=SimpleNamespace(inverse_mass=own((1, 0), np.float32)),
        dissipation=SimpleNamespace(coefficients_ns_per_m=own((1,))),
    )
    trace = np.zeros((1, 32), np.float32)
    trace[0, :6] = (1, 0, 0, 1, -0.5, 0.5)
    trace[0, 6:9] = (1, 0, 0)
    trace[0, 12:15] = (0.5, 0, 0)
    trace[0, 18:28] = (1, 1, 1, 0, 1, 2, -0.5, 0, 0, 1)
    phase = SimpleNamespace(
        schema="distance-dissipation-step-v1",
        ordered_point_ids=own(((11, 23),), np.int64), coefficients_ns_per_m=own((1,)),
        reconstructed_velocity=own(((1, 0, 0), (0, 0, 0)), np.float32),
        dissipated_velocity=own(((0.5, 0, 0), (0, 0, 0)), np.float32),
        trace=own(trace, np.float32), node_impulse=own(((-0.5, 0, 0), (0.5, 0, 0))),
        node_velocity_impulse=own(((-0.5, 0, 0), (0, 0, 0))),
        kinetic_change=own((-0.375,)), ideal_work=own((-0.375,)), rounding_work=own((0,)),
    )
    return spec, np.array(((1, 0, 0), (0, 0, 0)), np.float32), phase


@pytest.mark.parametrize("bad", ("chain", "coefficient", "fixed-impulse", "energy"))
def test_damping_observer_rejects_broken_phase_evidence(bad):
    module = runner()
    spec, position, phase = phase_record()
    actual = module.audit_dissipation(spec, position, phase, 1)
    np.testing.assert_array_equal(actual["fixed_reaction_impulse"], ((-0.5, 0, 0),))
    field = {"chain": "trace", "coefficient": "coefficients_ns_per_m", "fixed-impulse": "node_impulse", "energy": "kinetic_change"}[bad]
    changed = getattr(phase, field).copy()
    if bad == "chain":
        changed[0, 6] = 2
    elif bad == "fixed-impulse":
        changed[1, 0] = 0
    else:
        changed.flat[0] += 0.25
    frozen = np.frombuffer(changed.tobytes(), dtype=changed.dtype).reshape(changed.shape)
    setattr(phase, field, frozen)
    with pytest.raises(AssertionError):
        module.audit_dissipation(spec, position, phase, 1)


def gate_record(module):
    """Validator-only synthetic counters, never counted as physical evidence."""
    observed = module.Observations(module.ref.make_case(module.ref.NAMES[0]))
    observed.steps = observed.case["steps"]
    observed.minimum_gap, observed.minimum_side, observed.peak = 1 / 32, 5 / 32, 16
    for name, start in (("initial_outside", 0), ("entry", 0.25), ("release", 1), ("recontact", 2)):
        observed.events[name].update(maximum=observed.minimum, qualified_start=start, qualified_end=start + 1 / 128)
    for row in observed.windows:
        n = int((row["end"] - row["begin"]) / observed.case["h"])
        row.update(count=n, force_sum=16 * n, rigid_speed2=0.0, node_speed2=0.0)
    return observed


@pytest.mark.parametrize("bad", ("recontact", "rigid-rms", "center-rms", "force", "truncated", "window-count"))
def test_damped_window_retains_every_original_functional_gate(bad):
    observed = gate_record(runner())
    observed.finish()
    if bad == "recontact":
        observed.events["recontact"]["maximum"] = 0
    elif bad == "truncated":
        observed.steps -= 1
    elif bad == "window-count":
        observed.windows[1]["count"] -= 1
    else:
        field, value = {"rigid-rms": ("rigid_speed2", 0.2**2), "center-rms": ("node_speed2", 0.2**2), "force": ("force_sum", 32)}[bad]
        observed.windows[1][field] = value * observed.windows[1]["count"]
    with pytest.raises(AssertionError):
        observed.finish()


def test_new_runner_verifies_unchanged_original_dependency_bytes():
    module = runner()
    result = module.reference_provenance()
    assert set(result) == {"deformable_damped_load_cases", "deformable_normal_load_cases", "deformable_impulse_audit"}
    for value in result.values():
        assert len(value["sha256"]) == 64 and value["bytes"] > 0
    original = module.EXPECTED_DEPENDENCIES["deformable_normal_load_cases"]
    module.EXPECTED_DEPENDENCIES["deformable_normal_load_cases"] = "0" * 64
    try:
        with pytest.raises(AssertionError, match="dependency bytes"):
            module.reference_provenance()
    finally:
        module.EXPECTED_DEPENDENCIES["deformable_normal_load_cases"] = original

@pytest.mark.tyx
def test_late_observer_failure_keeps_one_real_damped_return(tyx_mode, tmp_path, monkeypatch):
    import gzip
    import json

    module = runner()
    original = module.audit_step
    visits = []

    def after_real_audit(*args, **kwargs):
        original(*args, **kwargs)
        visits.append(args[3])
        raise RuntimeError("D6-B observer failed after first actual return")

    output = tmp_path / "one-return-prefix"
    with monkeypatch.context() as patch:
        patch.setattr(module, "audit_step", after_real_audit)
        with pytest.raises(RuntimeError, match="observer failed after first"):
            module.run_window(module.ref.NAMES[0], tyx_mode, output)
    assert len(visits) == 1
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert summary["progress"]["steps_returned"] == summary["progress"]["worldsteps"] == 1
    assert summary["progress"]["steps_checked"] == 0
    assert summary["progress"]["close_returned"] == 1
    with gzip.open(output / "damped-normal-load.jsonl.gz", "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    assert [r["ordinal"] for r in records] == list(range(len(records)))
    events = [r["event"] for r in records]
    assert events.index("step-attempt") < events.index("step-return") < events.index("damping-phase-audit") < events.index("failure")
    returned = next(r for r in records if r["event"] == "step-return")
    receipt = dict(returned["payload"]["mapping_entries"])["receipt"]
    node = dict(receipt["node_receipt"]["worlds"]["mapping_entries"])[7]
    assert node["dissipation"]["trace"]["shape"] == [4, 32]
    state = dict(receipt["states"]["mapping_entries"])[7]
    assert state["node"]["tick"] == state["rigid"]["tick"] == 1
    # The observer fails after common physical publication; no rollback claim.