"""Run the official G1 policy in an independent MuJoCo reference environment."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

ROOT = None


def rotation(quaternion):
    result = np.zeros(9)
    mujoco.mju_quat2Mat(result, np.asarray(quaternion, dtype=float))
    return result.reshape(3, 3)


def axis_rotation(axis, angle):
    quaternion = np.zeros(4)
    quaternion[0] = np.cos(angle / 2)
    quaternion[axis + 1] = np.sin(angle / 2)
    return rotation(quaternion)


def torso(root_quaternion, joints):
    return (
        rotation(root_quaternion)
        @ axis_rotation(2, joints[12])
        @ axis_rotation(0, joints[13])
        @ axis_rotation(1, joints[14])
    )


def yaw(matrix):
    angle = np.arctan2(matrix[1, 0], matrix[0, 0])
    return axis_rotation(2, angle)


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--scene", default="scene_g1.xml")
    parser.add_argument("--output", default="dance-reference-full.json")
    args = parser.parse_args()
    ROOT = args.assets.resolve()
    config = yaml.safe_load((ROOT / "deploy.yaml").read_text())
    motion = np.load(ROOT / "dance1_subject2.npz", allow_pickle=False)
    session = ort.InferenceSession(str(ROOT / "policy.onnx"), providers=["CPUExecutionProvider"])
    model = mujoco.MjModel.from_xml_path(str(ROOT / args.scene))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    joints = model.actuator_trnid[:, 0]
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    assert model.nu == 29 and len(set(qadr)) == 29
    data.qpos[:3] = motion["body_pos_w"][0, 0]
    data.qpos[3:7] = motion["body_quat_w"][0, 0]
    data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
    data.qpos[qadr] = motion["joint_pos"][0]
    data.qvel[:3] = motion["body_lin_vel_w"][0, 0]
    data.qvel[3:6] = rotation(data.qpos[3:7]).T @ motion["body_ang_vel_w"][0, 0]
    data.qvel[vadr] = motion["joint_vel"][0]
    mujoco.mj_forward(model, data)
    kp, kd = np.array(config["stiffness"]), np.array(config["damping"])
    action_config = config["actions"]["JointPositionAction"]
    scale, offset = np.array(action_config["scale"]), np.array(action_config["offset"])
    default = np.array(config["default_joint_pos"])
    alignment = (
        yaw(torso(data.qpos[3:7], data.qpos[qadr])) @ yaw(rotation(motion["body_quat_w"][0, 0])).T
    )
    action = np.zeros(29)
    target = data.qpos[qadr].copy()
    frames = []
    failed = None
    for tick in range(10000):
        if tick % 10 == 0:
            frame = min(int(round(data.time * 50)), len(motion["joint_pos"]) - 1)
            reference_q = motion["joint_pos"][frame]
            reference_v = motion["joint_vel"][frame]
            current_torso = torso(data.qpos[3:7], data.qpos[qadr])
            relative = (
                current_torso.T @ alignment @ torso(motion["body_quat_w"][frame, 0], reference_q)
            )
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, 1, velocity, 1)
            obs = np.concatenate(
                (
                    reference_q,
                    reference_v,
                    relative[:, :2].reshape(-1),
                    velocity[:3],
                    data.qpos[qadr] - default,
                    data.qvel[vadr],
                    action,
                )
            ).astype(np.float32)[None]
            assert obs.shape == (1, 154) and np.isfinite(obs).all()
            action = session.run(["actions"], {"obs": obs})[0][0]
            assert action.shape == (29,) and np.isfinite(action).all()
            target = offset + scale * action
            frames.append(
                {
                    "time": float(data.time),
                    "qpos": data.qpos.tolist(),
                    "qvel": data.qvel.tolist(),
                    "action": action.tolist(),
                    "joint_rmse": float(np.sqrt(np.mean((data.qpos[qadr] - reference_q) ** 2))),
                }
            )
        torque = kp * (target - data.qpos[qadr]) - kd * data.qvel[vadr]
        data.ctrl[:] = np.clip(
            torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        )
        mujoco.mj_step(model, data)
        if (
            not np.isfinite(data.qpos).all()
            or data.qpos[2] < 0.35
            or rotation(data.qpos[3:7])[2, 2] < np.cos(1.0)
        ):
            failed = {
                "time": float(data.time),
                "root_height": float(data.qpos[2]),
                "root_up_z": float(rotation(data.qpos[3:7])[2, 2]),
            }
            break
    report = {
        "schema": "g1-official-policy-mujoco-reference-v1",
        "backend": "MuJoCo independent reference",
        "phymeo_steps": 0,
        "physics_steps": tick + 1,
        "duration_s": float(data.time),
        "policy_hz": 50,
        "physics_hz": 500,
        "joint_count": 29,
        "failure": failed,
        "frames": frames,
    }
    report["scene"] = args.scene
    (ROOT / args.output).write_text(json.dumps(report))
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
