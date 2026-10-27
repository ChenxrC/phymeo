"""Independent public-API oracle for the P3 CPU robot, in an isolated environment.

Run with .venv-mujoco after installing requirements/mujoco.lock. No upstream
solver implementation is imported into phymeo or translated into its algorithms.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import numpy as np

from phymeo.builders.robot import pendulum_chain
from phymeo.core.robot import RobotInputs, RobotState
from phymeo.dynamics.robot_dynamics import bias_forces, forward_dynamics, mass_matrix
from phymeo.dynamics.robot_kinematics import forward_kinematics
from phymeo.experiments.rigid_cloth import provenance, save_run, trajectory_digest
from phymeo.experiments.robot import run_pendulum


def model_xml(model, dt):
    def numbers(values):
        return " ".join(format(float(v), ".17g") for v in values)

    root = ET.Element("mujoco", model="phymeo-independent-oracle")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false")
    option = ET.SubElement(
        root,
        "option",
        timestep=format(dt, ".17g"),
        gravity=numbers(model.gravity),
        integrator="Euler",
    )
    ET.SubElement(option, "flag", contact="disable")
    world, bodies = ET.SubElement(root, "worldbody"), []
    for i, parent in enumerate(model.parents):
        body = ET.SubElement(
            world if parent == -1 else bodies[parent],
            "body",
            name=f"link{i}",
            pos=numbers(model.joint_position[i]),
            quat=numbers(model.joint_orientation[i][[3, 0, 1, 2]]),
        )
        ET.SubElement(
            body,
            "joint",
            name=f"joint{i}",
            type="hinge" if model.joint_types[i] == "revolute" else "slide",
            pos="0 0 0",
            axis=numbers(model.joint_axis[i]),
            limited="false",
            damping="0",
            armature="0",
            stiffness="0",
            frictionloss="0",
        )
        ET.SubElement(
            body,
            "inertial",
            mass=format(model.mass[i], ".17g"),
            pos=numbers(model.com[i]),
            quat="1 0 0 0",
            diaginertia=numbers(model.inertia_diagonal[i]),
        )
        bodies.append(body)
    return ET.tostring(root, encoding="unicode")


def checked_error(actual, expected, *, atol=5e-10, rtol=1e-9):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, "oracle shape mismatch"
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), "non-finite oracle result"
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return float(np.max(np.abs(actual - expected), initial=0))


def comparison_models():
    one, two, four = (pendulum_chain(n)[0] for n in (1, 2, 4))
    zero = four.joint_orientation.copy()
    zero[0] = [np.sin(0.15), 0, 0, np.cos(0.15)]
    zero[1] = [0, 0, np.sin(0.2), np.cos(0.2)]
    mixed = replace(
        four,
        parents=np.array([-1, 0, 0, 1]),
        joint_types=("revolute", "prismatic", "revolute", "revolute"),
        joint_orientation=zero,
        joint_axis=np.array([[0.0, 0, 1], [0.0, 1, 0], [1.0, 0, 0], [0.0, 0, 1]]),
        joint_position=np.array([[0.1, 0, 0], [0.0, -0.5, 0.1], [0.3, -0.1, 0], [0.0, -0.3, 0.2]]),
        com=np.array(
            [[0.1, -0.2, 0.03], [0.02, -0.1, 0.04], [0.0, -0.25, 0.05], [0.03, -0.12, -0.04]]
        ),
        inertia_diagonal=np.tile([0.02, 0.03, 0.04], (4, 1)),
    )
    return {"rod": one, "two_rods": two, "mixed_branch": mixed}


def oracle_mapping(mj_model, count):
    # MJCF compilation uses depth-first order; input trees need only topological
    # order. Named mapping is essential for siblings preceding grandchildren.
    joints = [mj_model.joint(f"joint{i}").id for i in range(count)]
    bodies = np.array([mj_model.body(f"link{i}").id for i in range(count)])
    return mj_model.jnt_qposadr[joints], mj_model.jnt_dofadr[joints], bodies


def compare_samples(mujoco, destination, arrays, *, samples):
    rng = np.random.default_rng(20261027)
    summaries = {}
    for case, model in comparison_models().items():
        xml = model_xml(model, 1 / 2000)
        (destination / f"{case}.xml").write_text(xml + "\n", encoding="utf-8")
        mj_model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(mj_model)
        qmap, vmap, bodies = oracle_mapping(mj_model, model.dof_count)
        count, rows, maxima = model.dof_count, {}, {}
        for _ in range(samples):
            state = RobotState(rng.uniform(-0.75, 0.75, count), rng.uniform(-0.5, 0.5, count))
            inputs = RobotInputs(
                rng.uniform(-0.5, 0.5, count),
                rng.uniform(-0.5, 0.5, (count, 3)),
                rng.uniform(-0.5, 0.5, (count, 3)),
            )
            data.qpos[qmap], data.qvel[vmap] = state.q, state.v
            data.qfrc_applied[vmap] = inputs.effort
            data.xfrc_applied[bodies, :3] = inputs.force
            data.xfrc_applied[bodies, 3:] = inputs.torque
            mujoco.mj_forward(mj_model, data)
            dense = np.empty((count, count))
            mujoco.mj_fullM(mj_model, dense, data.qM)
            geometry = forward_kinematics(model, state.q)
            checks = {
                "origin": (geometry.position, data.xpos[bodies]),
                "rotation": (geometry.rotation, data.xmat[bodies].reshape(count, 3, 3)),
                "com": (geometry.com_position, data.xipos[bodies]),
                "mass": (mass_matrix(model, state.q), dense[np.ix_(vmap, vmap)]),
                "bias": (bias_forces(model, state), data.qfrc_bias[vmap]),
                "qacc": (forward_dynamics(model, state, inputs), data.qacc[vmap]),
            }
            for name, (actual, expected) in checks.items():
                maxima[name] = max(maxima.get(name, 0.0), checked_error(actual, expected))
                rows.setdefault(f"phymeo_{name}", []).append(actual.copy())
                rows.setdefault(f"mujoco_{name}", []).append(expected.copy())
            for name, value in (
                ("q", state.q),
                ("v", state.v),
                ("effort", inputs.effort),
                ("force", inputs.force),
                ("torque", inputs.torque),
            ):
                rows.setdefault(name, []).append(value.copy())
        arrays.update({f"{case}_{name}": np.array(values) for name, values in rows.items()})
        summaries[case] = {"samples": samples, "dof": count, "max_abs_error": maxima}
    return summaries


def compare_trace(mujoco, destination, arrays, *, steps, source):
    first, second = run_pendulum(steps=steps), run_pendulum(steps=steps)
    for name in first.arrays:
        np.testing.assert_array_equal(first.arrays[name], second.arrays[name])
    save_run(first, destination / "phymeo-s1", source=source)
    model, state = pendulum_chain()
    mj_model = mujoco.MjModel.from_xml_string(model_xml(model, 1 / 2000))
    data = mujoco.MjData(mj_model)
    qmap, vmap, _ = oracle_mapping(mj_model, 2)
    data.qpos[qmap], data.qvel[vmap] = state.q, state.v
    q, v = np.empty((steps + 1, 2)), np.empty((steps + 1, 2))
    q[0], v[0] = state.q, state.v
    for step in range(1, steps + 1):
        mujoco.mj_step(mj_model, data)
        q[step], v[step] = data.qpos[qmap], data.qvel[vmap]
    errors = {
        "q": checked_error(first.arrays["q"], q, atol=1e-7, rtol=0),
        "v": checked_error(first.arrays["v"], v, atol=1e-7, rtol=0),
    }
    arrays.update(
        {
            "s1_mujoco_q": q,
            "s1_mujoco_v": v,
            "s1_phymeo_q": first.arrays["q"],
            "s1_phymeo_v": first.arrays["v"],
        }
    )
    return {
        "steps": steps,
        "dt": 1 / 2000,
        "max_abs_error": errors,
        "repeat_exact": True,
        "phymeo_trajectory_sha256": first.metrics["trajectory_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1 or args.steps < 1:
        parser.error("output must be new; samples and steps must be positive")
    try:
        import mujoco
    except ImportError as error:
        raise SystemExit(
            "optional MuJoCo 3.3.7 environment is required; install requirements/mujoco.lock"
        ) from error
    if mujoco.__version__ != "3.3.7":
        raise SystemExit("oracle requires exactly MuJoCo 3.3.7")
    source, arrays = provenance(), {}
    root = Path(__file__).resolve().parents[1]
    source.update(
        {
            "mujoco": mujoco.__version__,
            "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "oracle_lock_sha256": hashlib.sha256(
                (root / "requirements/mujoco.lock").read_bytes()
            ).hexdigest(),
        }
    )
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        samples = compare_samples(mujoco, args.output, arrays, samples=args.samples)
        trace = compare_trace(mujoco, args.output, arrays, steps=args.steps, source=source)
        np.savez_compressed(args.output / "comparison.npz", **arrays)
        report = {
            "status": "complete",
            "scope": "unconstrained fixed-base CPU dynamics; no contact or performance comparison",
            "provenance": source,
            "seed": 20261027,
            "sample_atol": 5e-10,
            "sample_rtol": 1e-9,
            "trace_atol": 1e-7,
            "cases": samples,
            "s1": trace,
            "arrays_sha256": trajectory_digest(arrays),
        }
        (args.output / "comparison.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps({"status": "failed", "error": str(error), "provenance": source}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        raise
    print(json.dumps({"cases": samples, "s1": trace}, indent=2))


if __name__ == "__main__":
    main()
