"""Change only the tangential magnitude on already observed Newton contact rows."""

import argparse
import json
from pathlib import Path

import numpy as np


def change_tangent(row, slip_m_s):
    normal = np.array(row["normal_only"], dtype=float)
    original = np.array(row["wrench_on_payload"], dtype=float)
    tangent = original - normal
    measured = np.linalg.norm(tangent[:3])
    speed = row["vt_m_s"]
    coefficient = row["material"]["mu"]
    requested = coefficient * row["normal_N"] * speed / max(speed, slip_m_s)
    if measured == 0:
        if requested > 1e-10:
            raise ValueError("the saved witness has no measurable tangent direction")
        return normal, 0.0, 0.0
    # The torque about the same COM is linear in force at this unchanged action point.
    return normal + tangent * (requested / measured), -measured * speed, -requested * speed


def compare(saved, spec):
    if not spec["pad_material"][3] == spec["table_material"][3] > 0:
        raise ValueError("this frozen comparison requires the same slip speed for pad and table")
    slip = spec["pad_material"][3]
    actual = []
    for record in saved["records"]:
        native = np.array(record["aggregate_kernel_wrench"])
        normal = np.array(record["zero_mu_kernel_wrench"])
        rows = [change_tangent(row, slip) for row in record["contacts"]]
        converted = np.sum([r[0] for r in rows], axis=0) if rows else np.zeros(6)
        capacity = sum(c["material"]["mu"] * c["normal_N"] for c in record["contacts"])
        assert np.linalg.norm(converted[:3] - normal[:3]) <= capacity + 1e-5
        actual.append(
            dict(
                t=record["t"],
                native=native.tolist(),
                same_normal=normal.tolist(),
                sampled_tangent_only=converted.tolist(),
                sum_muN=capacity,
                native_friction_power_W=sum(r[1] for r in rows),
                replaced_friction_power_W=sum(r[2] for r in rows),
            )
        )
    return actual


if __name__ == "__main__":
    import hashlib

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    row = dict(
        normal_only=[0, 0, 2, 0.5, 0, 0],
        wrench_on_payload=[-4, 0, 2, 0.5, 0, 1],
        normal_N=2.0,
        vt_m_s=0.02,
        material=dict(mu=0.5),
    )
    changed, old_power, new_power = change_tangent(row, 0.04)
    np.testing.assert_allclose(changed, [-0.5, 0, 2, 0.5, 0, 0.125])
    np.testing.assert_allclose([old_power, new_power], [-0.08, -0.01])
    saved = json.loads(args.rows.read_text(encoding="utf8"))
    assert saved["input_sha256"] == hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    result = compare(saved, json.loads(args.inputs.read_text(encoding="utf8")))
    with args.out.open("x", encoding="utf8") as out:
        json.dump(result, out, allow_nan=False)
    step4, first_finger = result[3], result[-1]
    assert (
        np.linalg.norm(step4["native"][:2]) > 2
        and np.linalg.norm(step4["sampled_tangent_only"][:2]) < 0.01
    )
    assert first_finger["native"][2] > 50 and abs(first_finger["sampled_tangent_only"][2]) < 1
    print(json.dumps(dict(step4=step4, first_finger=first_finger), allow_nan=False))
