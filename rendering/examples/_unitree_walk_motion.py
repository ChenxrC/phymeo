"""Run Unitree's pinned G1 locomotion policy in CPU MuJoCo.

From the repository root:
    python rendering/examples/_unitree_walk_motion.py --fetch --steps 3000

Assets, policy, configuration and reference implementation come from
https://github.com/unitreerobotics/unitree_rl_gym at REVISION (BSD-3-Clause).
Only the policy and original MJCF are executed. The cached deployment source is
provided for comparison. A new UnitreeWalk instance starts a new episode.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REVISION = "276801e46c5d433564f24658bac64f254b7d2d4b"
REPOSITORY = "https://github.com/unitreerobotics/unitree_rl_gym.git"
SCENE = "resources/robots/g1_description/scene.xml"
POLICY = "deploy/pre_train/g1/motion.pt"
CONFIG = "deploy/deploy_mujoco/configs/g1.yaml"
POLICY_SHA256 = "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d"
DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "unitree-walk"

# git ls-tree -r REVISION, restricted to the referenced files.
BLOBS = {
    "LICENSE": "9ad8f6dbb0d51bf0eb345fbe02a616f484b3e0d2",
    "deploy/deploy_mujoco/configs/g1.yaml": "db18fdfa78fe142c4d23d4bd5bccfcc4101c55fe",
    "deploy/deploy_mujoco/deploy_mujoco.py": "94bfb411140bbffd4e4ff8573b42bca90735340c",
    "deploy/pre_train/g1/motion.pt": "91a042aa76f6f3ec0f47b6b8839ba9ce2ca060c9",
    "resources/robots/g1_description/g1_12dof.xml": "5f7354bca2cae8ef867c2b14bcd46c94f0f10fc3",
    "resources/robots/g1_description/meshes/head_link.STL": "2ee5fba15de3ccd4c31774bc7a66de7aeefab770",
    "resources/robots/g1_description/meshes/left_ankle_pitch_link.STL": "69de8490184afc698f633e34bb74e65351bd4689",
    "resources/robots/g1_description/meshes/left_ankle_roll_link.STL": "8864e9f98103190883d1ba027aba5469925f2bca",
    "resources/robots/g1_description/meshes/left_elbow_link.STL": "1a96d99ba469960173129084ab6dd3bf8a732a71",
    "resources/robots/g1_description/meshes/left_hip_pitch_link.STL": "5b751c767ecf03b29c707b19f1faf5367e8d33d2",
    "resources/robots/g1_description/meshes/left_hip_roll_link.STL": "778437ffe61731915fa4cb7cf2390e9cbe738c43",
    "resources/robots/g1_description/meshes/left_hip_yaw_link.STL": "383093ab96ed3a099ab79f5d6ed144fc0b7067a9",
    "resources/robots/g1_description/meshes/left_knee_link.STL": "f2e98e54e8f1980eb62ad25c7a838044c23bbadf",
    "resources/robots/g1_description/meshes/left_shoulder_pitch_link.STL": "e698311fb12d3f22f018b668c40dab667c39b8d3",
    "resources/robots/g1_description/meshes/left_shoulder_roll_link.STL": "80bca84aca41137f8270fb35e263dcc5d9658e43",
    "resources/robots/g1_description/meshes/left_shoulder_yaw_link.STL": "281e699055ffafe094f52a440547eb4b64f83bfa",
    "resources/robots/g1_description/meshes/left_wrist_roll_rubber_hand.STL": "8fa435b66cf9d81df34423c752452ea2c7d90479",
    "resources/robots/g1_description/meshes/logo_link.STL": "e9792098504b64b359fbffa983373e5e6fa1b50a",
    "resources/robots/g1_description/meshes/pelvis.STL": "691a779b9c9f4c97f48eaa9edf9eb336ca3b4411",
    "resources/robots/g1_description/meshes/pelvis_contour_link.STL": "42434339abf5a0a3816f890389e3e61af3b1e03c",
    "resources/robots/g1_description/meshes/right_ankle_pitch_link.STL": "e77d8a2fe1e5d56fac049833d254d6ffa4f6b350",
    "resources/robots/g1_description/meshes/right_ankle_roll_link.STL": "d4261dd7ccf39dd9c54cad37c1982f1d50e6fccc",
    "resources/robots/g1_description/meshes/right_elbow_link.STL": "f259e3812efb9985d6463c04d3e8a4b53793a699",
    "resources/robots/g1_description/meshes/right_hip_pitch_link.STL": "998a0a0f5c248b217cf82c5088d6d365d9654ecd",
    "resources/robots/g1_description/meshes/right_hip_roll_link.STL": "47b2eebdd36c7289679eebb9583c65a15514e194",
    "resources/robots/g1_description/meshes/right_hip_yaw_link.STL": "371856427c7cc31ae7463c83f00ef6652c72d672",
    "resources/robots/g1_description/meshes/right_knee_link.STL": "76d21a3d81e4e34520c5d6b3f28d7e43c1874364",
    "resources/robots/g1_description/meshes/right_shoulder_pitch_link.STL": "3f5b4ed47ade68e83f617ef3fd65cb810e9b0e07",
    "resources/robots/g1_description/meshes/right_shoulder_roll_link.STL": "179d61753d48b3a55165b81ded2c14e84deb81c2",
    "resources/robots/g1_description/meshes/right_shoulder_yaw_link.STL": "2ba6076a8853edac5e3e17cffd2c256318e88d92",
    "resources/robots/g1_description/meshes/right_wrist_roll_rubber_hand.STL": "c365aa9d8cafb8a96d2f58fe53a83727bb7427c8",
    "resources/robots/g1_description/meshes/torso_link_23dof_rev_1_0.STL": "079aa8124f279cd40a50d23c2f061dd82ccc16c6",
    "resources/robots/g1_description/scene.xml": "d0e3c0adcd6f31f87664ed18ce3020aab95ba6b7",
}


def _matches(path, contents):
    header = f"blob {len(contents)}\0".encode("ascii")
    digest = hashlib.sha1(header + contents).hexdigest()
    if path == POLICY and hashlib.sha256(contents).hexdigest() != POLICY_SHA256:
        return False
    return digest == BLOBS[path]


def _missing(root):
    return [
        name for name in BLOBS
        if not (root / name).is_file() or not _matches(name, (root / name).read_bytes())
    ]


def fetch_assets(root=DEFAULT_ROOT):
    """Fetch only the selected official Git objects and verify every byte before use."""
    root = Path(root).resolve()
    missing = _missing(root)
    if not missing:
        return root
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="g1-git-", dir=root) as work:
        # Bare, ephemeral object storage. No checkout or repository hooks.
        subprocess.run(["git", "init", "--quiet", "--bare", work], check=True)
        git = ["git", "-C", work]
        subprocess.run(
            [*git, "-c", "protocol.version=2", "fetch", "--no-tags", "--no-write-fetch-head",
             REPOSITORY, *(BLOBS[name] for name in missing)], check=True,
        )
        for name in missing:
            data = subprocess.check_output([*git, "cat-file", "blob", BLOBS[name]])
            if not _matches(name, data):
                raise ValueError(f"Unitree asset verification failed: {name}")
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                temporary = Path(temp.name)
                try:
                    temp.write(data)
                except BaseException:
                    temp.close()
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    return root


class UnitreeWalk:
    """Official 50 Hz G1 policy and 500 Hz leg torque loop, with a free pelvis."""

    def __init__(self, root=DEFAULT_ROOT):
        import mujoco
        import torch
        import yaml

        root = Path(root).resolve()
        missing = _missing(root)
        if missing:
            raise ValueError(f"Missing or changed {missing[0]}. Run {Path(__file__).name} --fetch")
        self.mj = mujoco
        self.torch = torch
        self.config = cfg = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
        self.model = m = mujoco.MjModel.from_xml_path(str(root / SCENE))
        m.opt.timestep = cfg["simulation_dt"]
        self.data = mujoco.MjData(m)
        self._pose = mujoco.MjData(m)
        self.policy = torch.jit.load(str(root / POLICY), map_location="cpu").eval()
        self.target = np.asarray(cfg["default_angles"], dtype=np.float32)
        self.default = self.target.copy()
        self.kp = np.asarray(cfg["kps"], dtype=np.float32)
        self.kd = np.asarray(cfg["kds"], dtype=np.float32)
        self.cmd_observation = np.multiply(cfg["cmd_init"], cfg["cmd_scale"], dtype=np.float32)
        self.action = np.zeros(cfg["num_actions"], dtype=np.float32)
        self.obs = np.zeros(cfg["num_obs"], dtype=np.float32)
        self.tick = 0
        if (m.nq, m.nv, m.nu) != (19, 18, 12):
            raise ValueError("Unexpected joint layout for official G1 12-DoF policy")
        for i, joint_id in enumerate(m.actuator_trnid[:, 0]):
            if m.jnt_qposadr[joint_id] != 7 + i or m.jnt_dofadr[joint_id] != 6 + i:
                raise ValueError(f"Unexpected joint order for G1 actuator {i}")
        self._floor = m.geom("floor").id
        self._foot_ids = {m.body(f"{side}_ankle_roll_link").id: i
                          for i, side in enumerate(("left", "right"))}
        self._wrench = np.empty(6)
        self._start = self.data.qpos[:3].copy()
        self._min_height = float(self._start[2])
        self._min_up = 1.0
        self._support_ticks = [0, 0]
        self._single_ticks = [0, 0]
        self._touchdowns = [0, 0]
        self._last_loaded = [False, False]
        self._foot_impulse = [0.0, 0.0]
        self._other_support_ticks = 0
        self._pose_tick = -1

    @property
    def time_s(self):
        return float(self.data.time)

    def kinematics(self):
        """Post-integration body/geom transforms. Does not change the dynamics data."""
        if self._pose_tick != self.tick:
            self._pose.qpos[:] = self.data.qpos
            self._pose.qvel[:] = self.data.qvel
            self._pose.time = self.data.time
            self.mj.mj_kinematics(self.model, self._pose)
            self._pose_tick = self.tick
        return self._pose

    def _update_policy(self):
        cfg, d = self.config, self.data
        w, x, y, z = d.qpos[3:7]
        self.obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]
        self.obs[3:6] = (2 * (w*y-z*x), -2 * (z*y+w*x), 1 - 2*(w*w+z*z))
        self.obs[6:9] = self.cmd_observation
        self.obs[9:21] = (d.qpos[7:] - self.default) * cfg["dof_pos_scale"]
        self.obs[21:33] = d.qvel[6:] * cfg["dof_vel_scale"]
        self.obs[33:45] = self.action
        phase = (self.tick * cfg["simulation_dt"] % 0.8) / 0.8
        self.obs[45:] = (np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase))
        with self.torch.inference_mode():
            output = self.policy(self.torch.from_numpy(self.obs).unsqueeze(0))
        self.action = output.detach().cpu().numpy().reshape(12)
        if not np.isfinite(self.action).all():
            raise ArithmeticError(f"G1 policy returned nonfinite action at {self.tick}")
        self.target = self.default + self.action * cfg["action_scale"]

    def _observe_step(self):
        m, d = self.model, self.data
        # Constraint forces belong to the just-completed MuJoCo solve; query before
        # any further dynamics call. The post-step pelvis is read directly from qpos.
        loads = [0.0, 0.0]
        other = 0.0
        for index in range(d.ncon):
            pair = d.contact[index]
            if pair.geom1 == self._floor:
                geom = pair.geom2
            elif pair.geom2 == self._floor:
                geom = pair.geom1
            else:
                continue
            self.mj.mj_contactForce(m, d, index, self._wrench)
            normal = max(0.0, float(self._wrench[0]))
            foot = self._foot_ids.get(m.geom_bodyid[geom])
            if foot is None:
                other += normal
            else:
                loads[foot] += normal

        dt = m.opt.timestep
        loaded = [load > 1.0 for load in loads]
        for i, active in enumerate(loaded):
            self._support_ticks[i] += int(active)
            self._single_ticks[i] += int(active and not loaded[1-i])
            self._touchdowns[i] += int(active and not self._last_loaded[i])
            self._foot_impulse[i] += loads[i] * dt
        self._last_loaded = loaded
        self._other_support_ticks += int(other > 1.0)
        x, y = d.qpos[4:6]
        self._min_height = min(self._min_height, float(d.qpos[2]))
        self._min_up = min(self._min_up, float(1 - 2*(x*x+y*y)))

    def step(self, n=1):
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative number of whole MuJoCo steps")
        cfg, d = self.config, self.data
        for _ in range(n):
            d.ctrl[:] = (self.target - d.qpos[7:]) * self.kp - d.qvel[6:] * self.kd
            self.mj.mj_step(self.model, d)
            self.tick += 1
            if self.tick % cfg["control_decimation"] == 0:
                self._update_policy()
            self._observe_step()
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            raise ArithmeticError(f"Nonfinite G1 state at step {self.tick}")
        return self.data

    def report(self):
        end = self.data.qpos[:3]
        dt = self.model.opt.timestep
        return {
            "source_revision": REVISION, "policy_sha256": POLICY_SHA256,
            "mujoco_version": self.mj.__version__, "torch_version": str(self.torch.__version__),
            "physics_steps": self.tick, "simulation_time_s": self.time_s,
            "step_s": dt, "policy_updates": self.tick // self.config["control_decimation"],
            "command_forward_mps": self.config["cmd_init"][0],
            "start_pelvis_xyz_m": self._start.tolist(), "end_pelvis_xyz_m": end.tolist(),
            "forward_displacement_m": float(end[0]-self._start[0]),
            "min_pelvis_height_m": self._min_height, "min_pelvis_up_dot_z": self._min_up,
            "left_right_loaded_seconds": [t*dt for t in self._support_ticks],
            "left_right_single_support_seconds": [t*dt for t in self._single_ticks],
            "left_right_touchdowns": self._touchdowns.copy(),
            "left_right_normal_impulse_Ns": self._foot_impulse.copy(),
            "nonfoot_floor_load_seconds": self._other_support_ticks * dt,
            "load_detection_N": 1.0,
        }

    def assert_walking(self):
        """Require forward travel, alternating physical support and an upright pelvis."""
        report = self.report()
        checks = {
            "trajectory at least 4 seconds": self.time_s >= 4.0 - 1e-8,
            "elapsed time matches integrated steps": math.isclose(
                self.time_s, self.tick * self.model.opt.timestep, abs_tol=1e-8),
            "forward motion": report["forward_displacement_m"] > .15 * self.time_s,
            "upright": self._min_height > .4 and self._min_up > .5,
            "both feet reloaded": min(self._touchdowns) >= 2,
            "each foot supported alone": min(report["left_right_single_support_seconds"]) > .1,
            "feet supplied all floor support above 1 N": self._other_support_ticks == 0,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise AssertionError(f"G1 walking checks failed: {', '.join(failed)}\n{report}")
        return report


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    cli.add_argument("--fetch", action="store_true", help="Download or repair official assets")
    cli.add_argument("--steps", type=int, default=0, help="Run continuous 2 ms physics steps")
    args = cli.parse_args()
    if args.fetch:
        fetch_assets(args.root)
    if args.steps:
        walk = UnitreeWalk(args.root)
        walk.step(args.steps)
        result = walk.assert_walking() if args.steps >= 2000 else walk.report()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
