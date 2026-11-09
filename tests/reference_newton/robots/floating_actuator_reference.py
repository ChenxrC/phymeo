"""Independent MuJoCo M/bias/qacc reference for generic effective joint actuators.

Run only in the parent's serial execution slot, with the intended Phymeo source
on PYTHONPATH. This script performs no trajectory integration or fixture fitting.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from phymeo.core.rigid import RigidBodyModel
from phymeo.core.robot import RobotModel
from phymeo.dynamics.floating_robot import FloatingRobotModel, FloatingRobotState
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms


def cross_matrix(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def pair(kind, armature, damping):
    mj_kind = "hinge" if kind == "revolute" else "slide"
    xml = f"""<mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/>
    <worldbody><body name="root"><freejoint/>
    <inertial mass="2" pos=".1 .02 -.03" quat=".9238795325112867 0 0 .3826834323650898"
      diaginertia=".4 .5 .6"/>
    <body name="child" pos=".2 .1 -.1">
    <joint name="scalar" type="{mj_kind}" axis="0 .6 .8" armature="{armature}"
      damping="{damping}"/>
    <inertial mass="1.25" pos=".04 -.02 .03" diaginertia=".1 .12 .15"/>
    </body></body></worldbody></mujoco>"""
    mj = mujoco.MjModel.from_xml_string(xml)
    gravity = np.array([0.0, 0.0, -9.81])
    children = RobotModel(
        np.array([-1]),
        (kind,),
        np.array([[0.2, 0.1, -0.1]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.6, 0.8]]),
        np.array([1.25]),
        np.array([[0.04, -0.02, 0.03]]),
        np.array([[0.1, 0.12, 0.15]]),
        gravity,
    )
    model = FloatingRobotModel(
        RigidBodyModel(np.array([0.5]), np.array([[0.4, 0.5, 0.6]]), gravity),
        np.array([0.1, 0.02, -0.03]),
        children,
        root_inertia_frame_xyzw=np.array([0.0, 0.0, 0.3826834323650898, 0.9238795325112867]),
        joint_armature=np.array([armature]),
        joint_damping=np.array([damping]),
    )
    return mj, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for kind in ("revolute", "prismatic"):
        for label, armature, damping in (("zero", 0.0, 0.0), ("passive", 0.03, 0.05)):
            mj, model = pair(kind, armature, damping)
            for pose, orientation, rate in (
                ("identity-rest", [0.0, 0.0, 0.0, 1.0], np.zeros(7)),
                ("tilted-moving", [0.0, 0.6, 0.0, 0.8], np.array([.1, -.2, .03, .2, -.1, .4, -.3])),
            ):
                q = np.array([.2, -.1, .7, *orientation, .25])
                state = FloatingRobotState(q, rate)
                rotation = quaternion_matrix(state.q[None, 3:7])[0]
                offset = rotation @ model.root_com_m
                transform = np.eye(7)
                transform[:3, 3:6] = cross_matrix(offset)
                transform[3:6, 3:6] = rotation.T
                derivative = np.zeros(7)
                derivative[:3] = -np.cross(rate[3:6], np.cross(rate[3:6], offset))
                data = mujoco.MjData(mj)
                data.qpos[:3] = q[:3]
                data.qpos[3:7] = q[[6, 3, 4, 5]]
                data.qpos[7] = q[7]
                data.qvel[:] = transform @ rate
                data.qfrc_applied[6] = .7
                mujoco.mj_forward(mj, data)
                mass_native = np.zeros((7, 7))
                try:
                    mujoco.mj_fullM(mj, mass_native, data.qM)
                    full_m_api = "legacy-qM"
                except TypeError:
                    mujoco.mj_fullM(mj, data, mass_native)
                    full_m_api = "data"
                mass_expected = transform.T @ mass_native @ transform
                bias_expected = transform.T @ (
                    data.qfrc_bias - data.qfrc_passive + mass_native @ derivative
                )
                acceleration_expected = np.linalg.solve(transform, data.qacc - derivative)
                inputs = FloatingRobotInputs(np.array([.7]), np.zeros((2, 3)), np.zeros((2, 3)))
                actual = floating_dynamics_terms(model, state, inputs)
                np.testing.assert_allclose(actual.mass_matrix, mass_expected, rtol=1e-10, atol=1e-11)
                np.testing.assert_allclose(actual.bias, bias_expected, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(actual.acceleration, acceleration_expected, rtol=1e-9, atol=1e-10)
                rows.append({
                    "identity": f"{kind}-{label}-{pose}",
                    "q": q.tolist(),
                    "v": rate.tolist(),
                    "armature": armature,
                    "damping": damping,
                    "basis_transform": transform.tolist(),
                    "basis_derivative_velocity": derivative.tolist(),
                    "mujoco_mass": mass_expected.tolist(),
                    "phymeo_mass": actual.mass_matrix.tolist(),
                    "mujoco_bias": bias_expected.tolist(),
                    "phymeo_bias": actual.bias.tolist(),
                    "mujoco_acceleration": acceleration_expected.tolist(),
                    "phymeo_acceleration": actual.acceleration.tolist(),
                    "full_m_api": full_m_api,
                })
    report = {
        "schema": "floating-joint-actuator-mujoco-reference-v1",
        "mujoco_version": mujoco.__version__,
        "physics_steps": 0,
        "forward_calls": len(rows),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {len(rows)} independent M/bias/qacc cases, zero physics steps")


if __name__ == "__main__":
    main()
