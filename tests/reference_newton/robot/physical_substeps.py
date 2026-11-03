"""Newton 1.7 HOST observations for a physical 2 kg prismatic rail, without Phymeo imports."""

import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np


def _case(newton, delay=0, *, kp=0.0, kd=0.0, cap=100.0, stops=False):
    from newton.actuators import ClampingMaxEffort, DrivePD
    from newton.solvers import SolverFeatherstone

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=2.0, inertia=np.eye(3), lock_inertia=True)
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=(1.0, 0.0, 0.0),
        armature=0.0,
        damping=0.0,
        target_ke=0.0,
        target_kd=0.0,
        limit_lower=-0.25 if stops else -100.0,
        limit_upper=0.25 if stops else 100.0,
        limit_ke=2000.0 if stops else 0.0,
        limit_kd=5.0 if stops else 0.0,
    )
    builder.add_articulation([joint])
    builder.add_actuator(
        DrivePD,
        index=builder.joint_qd_start[joint],
        kp=kp,
        kd=kd,
        delay_steps=delay or None,
        clamping=[(ClampingMaxEffort, {"max_effort": cap})],
    )
    model = builder.finalize(device="cpu")
    actuator = model.actuators[0]
    return model, SolverFeatherstone(model, angular_damping=0.0), actuator


def _run(sim, commands, *, h, q0=0.0, v0=0.0, external=0.0):
    model, solver, drive = sim
    state, after = model.state(), model.state()
    state.joint_q.assign([q0])
    state.joint_qd.assign([v0])
    control = model.control()
    history, candidate = drive.state(), drive.state()
    rows = []
    for target, target_v, ff in commands:
        state.clear_forces()
        control.joint_target_q.assign([target])
        control.joint_target_qd.assign([target_v])
        control.joint_act.assign([ff])
        control.joint_f.assign([external])
        drive.step(state, control, history, candidate, dt=h)
        history, candidate = candidate, history
        applied = float(control.joint_f.numpy()[0])
        before_v = float(state.joint_qd.numpy()[0])
        solver.step(state, after, control, None, h)
        state, after = after, state
        v = float(state.joint_qd.numpy()[0])
        rows.append(
            dict(
                drive_n=applied - external,
                q_m=float(state.joint_q.numpy()[0]),
                v_m_s=v,
                inferred_stop_n=2 * (v - before_v) / h - applied,
            )
        )
    return rows


def main():
    import newton
    import warp as wp

    if version("newton") != "1.7.0":
        raise RuntimeError("This observation targets Newton 1.7.0")
    wp.config.kernel_cache_dir = str(Path(sys.prefix) / "warp-kernels")
    wp.init()
    with wp.ScopedDevice("cpu"):
        stops = []
        rail = _case(newton, stops=True)
        for side in (-1, 1):
            for toward, magnitude in ((0, 2), (0.1, 2.5), (-0.1, 1.5)):
                (row,) = _run(rail, [(0, 0, 0)], h=0.01, q0=side * 0.251, v0=side * toward)
                np.testing.assert_allclose(
                    row["inferred_stop_n"], -side * magnitude, rtol=0, atol=1e-4
                )
                stops.append(dict(side=side, toward_m_s=toward, **row))

        commands = [1, -2, 3, -4, 0]
        delays = []
        for d in (0, 1, 2):
            # One Actuator.step per actual solver step. Zero commands establish the
            # same rest prehistory as Phymeo before comparing signed feedforwards.
            sim = _case(newton, d)
            impulses = [0] * d + commands
            rows = _run(sim, [(0, 0, f) for f in impulses], h=0.125)[d:]
            wanted = ([0] * d + commands)[: len(commands)]
            np.testing.assert_allclose([r["drive_n"] for r in rows], wanted, rtol=0, atol=1e-6)
            v = np.cumsum(wanted) * 0.125 / 2
            np.testing.assert_allclose([r["v_m_s"] for r in rows], v, rtol=0, atol=1e-6)
            np.testing.assert_allclose(
                [r["q_m"] for r in rows], np.cumsum(v) * 0.125, rtol=0, atol=1e-6
            )
            delays.append(dict(delay_steps=d, rows=rows))
        (empty,) = _run(_case(newton, 2), [(0, 0, 1)], h=0.125)
        np.testing.assert_allclose(empty["drive_n"], 1.0, rtol=0, atol=1e-6)

        pd = []
        for cap in (10.0, 3.0):
            (row,) = _run(
                _case(newton, kp=8, kd=2, cap=cap), [(0.25, 0.5, 0.5)], h=0.125, external=1.0
            )
            want = min(3.5, cap)
            np.testing.assert_allclose(
                [row["drive_n"], row["v_m_s"], row["q_m"]],
                [want, (want + 1) / 16, (want + 1) / 128],
                rtol=0,
                atol=1e-6,
            )
            pd.append(dict(cap_n=cap, **row))
    print(
        json.dumps(
            dict(
                newton=version("newton"),
                warp=version("warp-lang"),
                device="cpu",
                stops=stops,
                primed_delays=delays,
                empty_delay_2=empty,
                explicit_pd=pd,
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
