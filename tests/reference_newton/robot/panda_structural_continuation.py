"""Compare a changed structural response at the first common step and over the retained window."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from phymeo.builders.panda import load_panda
from phymeo.core.robot import RobotState
from phymeo.dynamics.robot_dynamics import robot_dynamics_terms


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def compare(old, current, spec, native):
    dt = spec["dt_s"]
    a, b = old["trace"], current["trace"]
    assert len(a) == len(b) == spec["steps"] == 3600
    first = current["first_structural_contact"]
    i = first["step_index"]
    assert i == (1340 if native else 1339) and first["shape"] == "panda_link1_sc_0"
    for key in ("t", "q", "v", "cube", "xyzw", "velocity", "omega"):
        np.testing.assert_array_equal([r[key] for r in a[:i]], [r[key] for r in b[:i]])
    (c,) = b[i]["structural_contacts"]
    P = np.array(c["force_on_payload_N"]) * dt
    observed = spec["mass_kg"] * (np.array(b[i]["velocity"]) - a[i]["velocity"])
    np.testing.assert_allclose(P, observed, atol=8e-7 if native else 1e-12, rtol=0)
    M = robot_dynamics_terms(
        load_panda().model, RobotState(np.array(b[i - 1]["q"]), np.array(b[i - 1]["v"]))
    )[3]
    delta_v = np.array(b[i]["v"]) - a[i]["v"]
    expected = dt * np.linalg.solve(M, b[i]["structural_joint_load"])
    np.testing.assert_allclose(delta_v, expected, atol=2e-7 if native else 1e-11, rtol=1e-3)
    assert np.linalg.norm(expected) > 0 and np.linalg.norm(
        delta_v - expected
    ) < 0.15 * np.linalg.norm(expected)
    contacts = [r for r in b if r["structural_contacts"]]
    impulses = np.zeros(3)
    losses = 0.0
    events = {}
    deepest = 0.0
    for row in contacts:
        for hit in row["structural_contacts"]:
            events[hit["shape"]] = events.get(hit["shape"], 0) + 1
            impulses += dt * np.array(hit["force_on_payload_N"])
            deepest = min(deepest, hit["gap_m"])
            losses += dt * hit["pair_power_W"]
    return dict(
        prefix_steps_bitwise=i,
        t_before_s=first["t_before_s"],
        t_after_s=b[i]["t"],
        first_contact=c,
        expected_extra_payload_impulse_Ns=P.tolist(),
        actual_extra_payload_impulse_Ns=observed.tolist(),
        extra_joint_velocity=delta_v.tolist(),
        expected_extra_joint_velocity=expected.tolist(),
        max_first_joint_error=float(np.max(np.abs(delta_v - expected))),
        earliest_step_end_s=contacts[0]["t"],
        last_step_end_s=contacts[-1]["t"],
        contact_steps=len(contacts),
        active_rows_by_shape=events,
        force_impulse_Ns=impulses.tolist(),
        pair_work_J=losses,
        min_gap_m=deepest,
        final_position=b[-1]["cube"],
        old_final_position=a[-1]["cube"],
        final_velocity=b[-1]["velocity"],
        old_final_velocity=a[-1]["velocity"],
        first_unresolved=current["first_unresolved_structure"],
        carried=current["carried_during_transfer"],
        success=current["success"],
    )


if __name__ == "__main__":
    c = argparse.ArgumentParser(description=__doc__)
    for key in ("old-inputs", "new-inputs", "old-motion", "new-motion", "out"):
        c.add_argument("--" + key, type=Path, required=True)
    args = c.parse_args()
    original, new = read(args.old_inputs), read(args.new_inputs)
    assert (
        hashlib.sha256(args.old_inputs.read_bytes()).hexdigest()
        == "d49f2f212dd38f309dc2bc8044149f5705242a2e035f9673b349727346d72e7a"
    )
    assert new.pop("structural_material") == [6000.0, 16.0, 0.35, 0.04] and new == original
    before, after = read(args.old_motion), read(args.new_motion)
    result = compare(before, after, new, "engine" in after)
    with args.out.open("x", encoding="utf8") as f:
        json.dump(result, f, allow_nan=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "first_contact"}, allow_nan=False))
