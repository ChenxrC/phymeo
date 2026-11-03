"""Newton Featherstone world FREE: accepted public end states on an eccentric body."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from floating_root import _model


def _flight(newton, wp, model, *, initial_v, wrench, dt, count):
    from newton.solvers import SolverFeatherstone

    solver = SolverFeatherstone(model, angular_damping=0.0)
    state, candidate, control = model.state(), model.state(), model.control()
    state.joint_q.assign([0, 0, 0, 0, 0, 0, 1])
    state.joint_qd.assign(initial_v)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    angle, rows = 0.0, []
    for k in range(1, count + 1):
        state.clear_forces()
        control.joint_f.assign(wrench)
        solver.step(state, candidate, control, None, dt)
        state, candidate = candidate, state
        q, v = state.joint_q.numpy().astype(float), state.joint_qd.numpy().astype(float)
        bodies, twists = state.body_q.numpy(), state.body_qd.numpy()
        np.testing.assert_allclose(bodies[0], q, rtol=0, atol=2e-5)
        np.testing.assert_allclose(twists[0], v, rtol=0, atol=2e-5)
        # Inputs here have only z torque. Its principal moment is 1/2 kg m².
        linear_acc = np.asarray(wrench[:3]) / 2
        omega = initial_v[5] + 2 * wrench[5] * k * dt
        expected_v = np.r_[np.asarray(initial_v[:3]) + k * dt * linear_acc, 0, 0, omega]
        expected_center = (
            np.array([0.125, 0, 0])
            + k * dt * np.asarray(initial_v[:3])
            + dt**2 * k * (k + 1) / 2 * linear_acc
        )
        # Newton normalizes q + h*qdot. A principal-z spin therefore advances
        # by 2 atan(h*omega_new/2) per step, including with nonzero torque.
        angle += 2 * np.arctan(dt * omega / 2)
        expected_q = np.r_[
            expected_center - 0.125 * np.array([np.cos(angle), np.sin(angle), 0]),
            0,
            0,
            np.sin(angle / 2),
            np.cos(angle / 2),
        ]
        center = np.asarray(
            wp.transform_point(
                wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:])), wp.vec3(0.125, 0, 0)
            ),
            dtype=float,
        )
        np.testing.assert_allclose(center, expected_center, rtol=0, atol=2e-5)
        np.testing.assert_allclose(q, expected_q, rtol=0, atol=2e-5)
        np.testing.assert_allclose(v, expected_v, rtol=0, atol=2e-5)
        rows.append(
            dict(k=k, q_origin_xyzw=q.tolist(), v_com_omega=v.tolist(), com_m=center.tolist())
        )
    return rows


def main():
    import newton
    import warp as wp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    if version("newton") != "1.7.0":
        raise RuntimeError("The discrete free-root observations target Newton 1.7.0")
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    with wp.ScopedDevice(args.device):
        model = _model(newton, wp, args.device, False)
        coast = _flight(
            newton, wp, model, initial_v=[0, 0.25, 0, 0, 0, 2], wrench=[0] * 6, dt=1 / 32, count=16
        )
        # Net -2 N in z equals the 2 kg Phymeo case with gravity -1 m/s².
        forced = _flight(
            newton, wp, model, initial_v=[0] * 6, wrench=[4, -2, -2, 0, 0, 1], dt=1 / 16, count=8
        )
    print(
        json.dumps(
            dict(
                newton=version("newton"),
                warp=version("warp-lang"),
                device=str(model.device),
                coast=coast,
                forced=forced,
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
