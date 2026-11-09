"""Independent fixed-pose G1 inertia/acceleration check against MuJoCo.

Uses the declared primitive plant with armature/damping and zero dry friction.
No contact simulation, parameter fitting, policy inference or finite differences.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from phymeo.builders.floating_mjcf import load_floating_mjcf
from phymeo.dynamics.floating_robot import FloatingRobotState
from phymeo.dynamics.rotation import quaternion_matrix
from phymeo.solvers.floating_tree import (
    FloatingRobotInputs,
    floating_dynamics_terms,
    floating_kinetic_energy,
)


def skew(value):
    x, y, z = value
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def run(assets):
    path = assets / "g1_primitive_actuated.xml"
    asset = load_floating_mjcf(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    motion = np.load(assets / "dance1_subject2.npz", allow_pickle=False)
    joints = model.actuator_trnid[:, 0]
    policy_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in joints]
    mapping = np.array([policy_names.index(name) for name in asset.joint_names])
    root_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, asset.root_link)
    rows = []
    for frame in (0, 20, 200):
        data = mujoco.MjData(model)
        data.qpos[:3] = motion["body_pos_w"][frame, 0] + np.array([0, 0, 2.0])
        data.qpos[3:7] = motion["body_quat_w"][frame, 0]
        data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
        for index, joint in enumerate(joints):
            data.qpos[model.jnt_qposadr[joint]] = motion["joint_pos"][frame, index]
            data.qvel[model.jnt_dofadr[joint]] = motion["joint_vel"][frame, index]
        rotation = quaternion_matrix(data.qpos[None, [4, 5, 6, 3]])[0]
        omega = motion["body_ang_vel_w"][frame, 0].astype(float)
        data.qvel[:3] = motion["body_lin_vel_w"][frame, 0]
        data.qvel[3:6] = rotation.T @ omega
        data.ctrl[:] = 0.2 * np.sin(np.arange(29) + 1)
        mujoco.mj_forward(model, data)
        native_mass = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, native_mass, data.qM)
        transform = np.zeros((35, 35))
        root_offset = rotation @ model.body_ipos[root_body]
        transform[:3, :3] = np.eye(3)
        transform[:3, 3:6] = skew(root_offset)
        transform[3:6, 3:6] = rotation.T
        for asset_index, policy_index in enumerate(mapping):
            transform[model.jnt_dofadr[joints[policy_index]], 6 + asset_index] = 1
        q = np.r_[data.qpos[:3], data.qpos[[4, 5, 6, 3]], motion["joint_pos"][frame, mapping]]
        velocity = np.r_[
            data.qvel[:3] + np.cross(omega, root_offset), omega, motion["joint_vel"][frame, mapping]
        ]
        inputs = FloatingRobotInputs(data.ctrl[mapping], np.zeros((30, 3)), np.zeros((30, 3)))
        state = FloatingRobotState(q, velocity)
        terms = floating_dynamics_terms(asset.model, state, inputs)
        expected_mass = transform.T @ native_mass @ transform
        alpha = rotation @ data.qacc_smooth[3:6]
        acceleration = np.r_[
            data.qacc_smooth[:3]
            + np.cross(alpha, root_offset)
            + np.cross(omega, np.cross(omega, root_offset)),
            alpha,
            [data.qacc_smooth[model.jnt_dofadr[joints[index]]] for index in mapping],
        ]
        native_energy = float(0.5 * data.qvel @ native_mass @ data.qvel)
        energy = floating_kinetic_energy(asset.model, state)
        mass_error = float(np.max(np.abs(terms.mass_matrix - expected_mass)))
        acceleration_error = float(np.max(np.abs(terms.acceleration - acceleration)))
        energy_error = float(abs(energy - native_energy))
        assert mass_error < 1e-8, mass_error
        assert acceleration_error < 1e-7, acceleration_error
        assert energy_error < 1e-8, energy_error
        rows.append(
            {
                "frame": frame,
                "mass_max_abs_error": mass_error,
                "acceleration_max_abs_error": acceleration_error,
                "kinetic_abs_error": energy_error,
                "native_contacts_ignored": int(data.ncon),
            }
        )
    return {
        "schema": "g1-h1-dynamics-reference-v1",
        "source": str(path),
        "poses": rows,
        "physics_steps": 0,
        "dofs": 35,
        "dry_friction": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.assets)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
