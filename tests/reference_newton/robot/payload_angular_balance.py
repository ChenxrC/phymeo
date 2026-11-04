"""Check angular impulse about the COM of a saved uniform box or solid sphere."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np


def vector(value, size):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f"expected a finite {size}-vector")
    return a


def angular_momentum(xyzw, omega, principal):
    q = vector(xyzw, 4)
    norm = np.linalg.norm(q)
    if abs(norm - 1) > 2e-5:
        raise ValueError("published orientation is not unit length")
    xyz = q[:3] / norm
    w = q[3] / norm
    velocity = vector(omega, 3)
    # Conjugate rotation maps world omega into the principal body axes.
    local = velocity + 2 * np.cross(xyz, np.cross(xyz, velocity) - w * velocity)
    spin = principal * local
    return spin + 2 * np.cross(xyz, np.cross(xyz, spin) + w * spin)


def balance(spec, saved):
    trace = saved["trace"]
    dt = float(spec["dt_s"])
    mass = float(spec["mass_kg"])
    size = vector(spec["full_size_m"], 3)
    if not (np.isfinite(dt) and dt > 0 and np.isfinite(mass) and mass > 0 and min(size) > 0):
        raise ValueError("invalid physical inputs")
    if not trace or len(trace) != spec["steps"] or saved["steps"] != spec["steps"]:
        raise ValueError("the angular impulse audit requires every original integration step")
    if saved["cube_mass_kg"] != mass or not np.array_equal(saved["cube_full_size_m"], size):
        raise ValueError("trajectory payload differs from its input")
    geometry = spec.get("payload_geometry", {"kind": "box"})
    if saved.get("payload_geometry", {"kind": "box"}) != geometry:
        raise ValueError("trajectory geometry differs from its input")
    if geometry["kind"] == "sphere":
        radius = float(geometry["radius_m"])
        if not np.isfinite(radius) or radius <= 0 or not np.all(size == 2 * radius):
            raise ValueError("sphere diameters and the declared radius disagree")
        principal = np.full(3, (2 / 5) * mass * radius**2)
    elif geometry["kind"] == "box":
        principal = mass / 12 * (np.roll(size, 1) ** 2 + np.roll(size, 2) ** 2)
    else:
        raise ValueError("unsupported payload inertia")
    np.testing.assert_allclose(
        principal, vector(spec["inertia_diagonal_kg_m2"], 3), rtol=1e-12, atol=0
    )
    initial = angular_momentum(spec["body_xyzw"], spec["initial_omega_rad_s"], principal)
    before = initial
    impulse_sum = np.zeros(3)
    total_defect = total_applied = omitted_defect = 0.0
    worst = None
    for i, row in enumerate(trace, 1):
        if abs(row["t"] - i * dt) > 1e-10 or not np.isfinite(row["t"]):
            raise ValueError("missing or misaligned time sample")
        after = angular_momentum(row["xyzw"], row["omega"], principal)
        impulse = dt * vector(row["applied_payload_wrench"], 6)[3:]
        delta = after - before
        error = float(np.linalg.norm(delta - impulse))
        if worst is None or error > worst["error_Nms"]:
            worst = dict(
                step=i,
                time_s=row["t"],
                error_Nms=error,
                delta_L_Nms=delta.tolist(),
                applied_Nms=impulse.tolist(),
            )
        total_defect += error
        total_applied += np.linalg.norm(impulse)
        omitted_defect += np.linalg.norm(delta)
        impulse_sum += impulse
        before = after
    return dict(
        steps=len(trace),
        sum_applied_norm_Nms=float(total_applied),
        sum_step_error_norm_Nms=total_defect,
        sum_error_if_torque_omitted_Nms=float(omitted_defect),
        relative_step_error_sum=float(total_defect / total_applied) if total_applied else None,
        endpoint_error_Nms=(before - initial - impulse_sum).tolist(),
        worst_step=worst,
    )


def verify_units_and_frames():
    # A 12 kg 1 x 2 x 3 m box has principal moments (13,10,5) kg m^2.
    # Yaw maps its body Y to world -X. Rolling about world X keeps that moment 10.
    k = np.sqrt(0.5)
    np.testing.assert_allclose(
        angular_momentum([0, 0, k, k], [1, 0, 0], np.array([13, 10, 5])), [10, 0, 0], atol=1e-14
    )
    q = [0.5, -0.5, 0.5, 0.5]
    spec = dict(
        dt_s=0.25,
        steps=1,
        mass_kg=12.0,
        full_size_m=[1, 2, 3],
        inertia_diagonal_kg_m2=[13, 10, 5],
        body_xyzw=[0, 0, k, k],
        initial_omega_rad_s=[0.5, 0, 0],
    )
    row = dict(t=0.25, xyzw=q, omega=[1, 0, 0], applied_payload_wrench=[0, 0, 0, 20, 0, 0])
    data = dict(steps=1, cube_mass_kg=12.0, cube_full_size_m=[1, 2, 3], trace=[row])
    report = balance(spec, data)
    np.testing.assert_allclose(report["sum_step_error_norm_Nms"], 0, atol=1e-14)
    np.testing.assert_allclose(report["sum_error_if_torque_omitted_Nms"], 5, atol=1e-14)
    row.update(xyzw=[-v for v in q], applied_payload_wrench=[0, 0, 0, -20, 0, 0])
    np.testing.assert_allclose(balance(spec, data)["sum_step_error_norm_Nms"], 10, atol=1e-14)

    # A 5 kg sphere of radius 1 m has inertia 2 about any axis.
    spec.update(
        mass_kg=5,
        full_size_m=[2] * 3,
        inertia_diagonal_kg_m2=[2] * 3,
        payload_geometry={"kind": "sphere", "radius_m": 1},
    )
    data.update(cube_mass_kg=5, cube_full_size_m=[2] * 3, payload_geometry=spec["payload_geometry"])
    row["applied_payload_wrench"] = [0, 0, 0, 4, 0, 0]
    sphere = balance(spec, data)
    np.testing.assert_allclose(sphere["sum_step_error_norm_Nms"], 0, atol=1e-14)
    np.testing.assert_allclose(sphere["sum_error_if_torque_omitted_Nms"], 1, atol=1e-14)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    verify_units_and_frames()
    inputs = args.inputs.read_bytes()
    trajectory = args.trace.read_bytes()
    saved = json.loads(trajectory)
    digest = sha256(inputs).hexdigest()
    if saved["inputs_sha256"] != digest:
        raise ValueError("trajectory belongs to different inputs")
    report = balance(json.loads(inputs), saved)
    report.update(inputs_sha256=digest, trace_sha256=sha256(trajectory).hexdigest())
    with args.out.open("x", encoding="utf8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False))
    relative_error = report["relative_step_error_sum"]
    if relative_error is None:
        if report["sum_step_error_norm_Nms"]:
            raise ValueError("observed angular momentum changed without an applied torque")
    elif relative_error > 0.1:
        raise ValueError("angular impulse diagnostic exceeds its declared 10 percent threshold")


if __name__ == "__main__":
    main()
