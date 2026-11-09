"""Fixed-budget RSL-RL PPO on two real HOST RobotScene tasks.

AUTHORING_ONLY / UNEXECUTED. Optional dependencies: rsl-rl-lib==5.5.2,
Torch>=2.6, TensorDict>=.7. No TYX, fake GPU physics or custom PPO algorithm.
CLI: python -B examples/train_scene_ppo.py --output-dir <new directory>

The seed, plant, goals, budget, before/after evaluation and 20% cost-reduction
criterion are fixed below before execution. A missed target remains a failure;
this program never retries seeds, changes plant parameters or extends training.
"""

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import platform
import sys
import traceback
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import phymeo as ph
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand
from phymeo.training import ResidentVectorEnv
from phymeo.training.rsl_rl import RslRlVecEnv
from phymeo.training.scene_backend import SceneBatchBackend, SceneControl

SEED, WORLDS, ITERATIONS, ROLLOUT, HORIZON = 20261110, 16, 100, 24, 96
DT, EVAL_STEPS, IMPROVEMENT = 1 / 32, 192, .20
ROBOT = 10
TASKS = ("prismatic-reach", "revolute-velocity")
PPO = {
    "num_steps_per_env": ROLLOUT, "save_interval": 100, "logger": "tensorboard",
    "obs_groups": {"actor": ["policy"], "critic": ["policy"]}, "check_for_nan": True,
    "torch_compile_mode": None,
    "algorithm": {"class_name": "PPO", "learning_rate": .0003, "num_learning_epochs": 4,
                  "num_mini_batches": 4, "schedule": "fixed", "gamma": .99, "lam": .95,
                  "clip_param": .2, "entropy_coef": .01, "value_loss_coef": 1.,
                  "max_grad_norm": 1., "use_clipped_value_loss": True},
    "actor": {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "elu",
              "obs_normalization": False, "distribution_cfg": {
                  "class_name": "GaussianDistribution", "init_std": .4, "std_type": "log"}},
    "critic": {"class_name": "MLPModel", "hidden_dims": [32, 32], "activation": "elu",
               "obs_normalization": False},
}
PLAN = {"seed": SEED, "worlds": WORLDS, "iterations": ITERATIONS,
        "rollout_steps_per_world": ROLLOUT, "episode_steps": HORIZON, "dt_s": DT,
        "evaluation_steps": EVAL_STEPS, "cost_reduction_required": IMPROVEMENT,
        "tasks": TASKS, "ppo": PPO, "physics_backend": "HOST RobotScene",
        "policy_backend": "Torch CPU", "network_logging": False}


def serial(value):
    if isinstance(value, torch.Tensor):
        return serial(value.detach().cpu().numpy())
    if isinstance(value, SceneControl):
        return {"actions": serial(value.actions), "external": serial(value.external)}
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": value.shape,
                "bytes_hex": value.tobytes().hex(), "values": serial(value.tolist())}
    if isinstance(value, np.generic):
        return serial(value.item())
    if is_dataclass(value):
        return {field.name: serial(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_ieee_value": repr(value)}
    return value


def fact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(serial(value), stream, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def urdf(kind):
    joint, axis, limit = ("prismatic", "1 0 0", 16) if kind == TASKS[0] else ("revolute", "0 0 1", 32)
    return f'''<robot name="ppo_{joint}"><link name="base"/><link name="body">
<inertial><mass value="1"/><inertia ixx=".2" iyy=".2" izz=".2" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="physical_ball"><geometry><sphere radius="1"/></geometry></collision></link>
<joint name="controlled" type="{joint}"><parent link="base"/><child link="body"/>
<axis xyz="{axis}"/><limit lower="{-limit}" upper="{limit}" effort="2" velocity="32"/></joint></robot>'''


class SceneTask:
    def __init__(self, kind):
        self.kind = kind
        goals = [-.25, -.15, .15, .25] if kind == TASKS[0] else [-.75, -.5, .5, .75]
        self.goals = np.array([goals[i % 4] for i in range(WORLDS)])
        self.last_request = None
        self.last_transition = None

    def controls(self, actions, keys):
        effort = (2. if self.kind == TASKS[0] else 1.) * np.clip(actions[:, 0], -1, 1)
        controls = {key: SceneControl({ROBOT: RobotCommand([0.], feedforward=[effort[i]])})
                    for i, key in enumerate(keys)}
        self.last_request = {"actions": np.array(actions, copy=True), "effort": effort,
                             "controls": controls}
        return controls

    def coordinates(self, states, keys):
        q = np.array([states[key].robots[0, ROBOT].q[0] for key in keys])
        v = np.array([states[key].robots[0, ROBOT].v[0] for key in keys])
        return q, v

    def observe(self, states, keys):
        q, v = self.coordinates(states, keys)
        error = (q - self.goals) / .25 if self.kind == TASKS[0] else (v - self.goals) / .75
        return np.column_stack((error, v, self.goals))

    def cost(self, states, actions, keys):
        q, v = self.coordinates(states, keys)
        effort = (2. if self.kind == TASKS[0] else 1.) * np.clip(actions[:, 0], -1, 1)
        error = (q - self.goals) / .25 if self.kind == TASKS[0] else (v - self.goals) / .75
        return error**2 + (.05 * v**2 if self.kind == TASKS[0] else 0) + .001 * effort**2

    def evaluate(self, before, after, actions, keys):
        cost = self.cost(after, actions, keys)
        self.last_transition = {"before": before, "after": after, "actions": np.array(actions, copy=True),
                                "cost": cost}
        return -cost, np.zeros(len(keys), dtype=bool)


def make_environment(kind, source):
    def factory(world):
        builder = ContactSceneBuilder(gravity=(0, 0, 0))
        builder.add_urdf(ROBOT, source, robot_state=RobotState([0.], [0.]),
                         kp=[0.], kd=[0.], effort_limit=[2.])
        return builder.build(backend="host")

    task = SceneTask(kind)
    backend = SceneBatchBackend(tuple(range(WORLDS)), factory)
    env = ResidentVectorEnv(backend, task, backend.world_keys, action_size=1, dt=DT, horizon=HORIZON)
    try:
        adapter = RslRlVecEnv(env, device="cpu")
    except Exception:
        backend.close()
        raise
    return backend, task, adapter


def evaluate(policy, adapter):
    observations = adapter.reset()
    frames, actions, costs = [observations["policy"].cpu().numpy().copy()], [], []
    terminal_frames, final_masks, accepted_joint_qv = [], [], []
    with torch.inference_mode():
        for _ in range(EVAL_STEPS):
            action = policy(observations)
            observations, rewards, _, extras = adapter.step(action)
            actions.append(action.cpu().numpy().copy())
            costs.append(-rewards.cpu().numpy().copy())
            frames.append(observations["policy"].cpu().numpy().copy())
            terminal_frames.append(extras["final_observation"]["policy"].cpu().numpy().copy())
            final_masks.append(extras["final_mask"].cpu().numpy().copy())
            q, v = adapter.env.task.coordinates(adapter.env.task.last_transition["after"], adapter.env.world_keys)
            accepted_joint_qv.append(np.column_stack((q, v)))
    return {"mean_cost": float(np.mean(costs)), "observations": np.array(frames),
            "actions": np.array(actions), "costs": np.array(costs),
            "terminal_observations": np.array(terminal_frames), "final_masks": np.array(final_masks),
            "accepted_joint_qv": np.array(accepted_joint_qv)}


def run_task(kind, output, provenance):
    output.mkdir()
    source = output / "physical_task.urdf"
    source.write_text(urdf(kind), encoding="utf-8")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    record = {"task": kind, "plan": PLAN, "plant": fact(source), "urdf": urdf(kind),
              "status": "starting", "seed": SEED}
    task = reloaded_task = None
    try:
        backend, task, adapter = make_environment(kind, source)
        with backend:
            runner = OnPolicyRunner(adapter, copy.deepcopy(PPO), log_dir=str(output / "logs"), device="cpu")
            before = evaluate(runner.get_inference_policy(), adapter)
            record.update(before=before, stage="train", goals=task.goals)
            initial_actor = {key: value.detach().clone() for key, value in runner.alg.actor.state_dict().items()}
            adapter.reset()
            runner.learn(num_learning_iterations=ITERATIONS, init_at_random_ep_len=False)
            after = evaluate(runner.get_inference_policy(), adapter)
            changed = any(not torch.equal(value, runner.alg.actor.state_dict()[key]) for key, value in initial_actor.items())
            checkpoint = output / "trained.pt"
            infos = {"seed": SEED, "task": kind, "plan": PLAN, "sources": provenance,
                     "environment_resume": "fresh original worlds for evaluation; no mid-episode resume claim"}
            runner.save(str(checkpoint), infos=infos)
            record.update(after=after, actor_parameters_changed=changed, checkpoint=fact(checkpoint), stage="reload")
        reloaded_backend, reloaded_task, reloaded_adapter = make_environment(kind, source)
        with reloaded_backend:
            restored_runner = OnPolicyRunner(reloaded_adapter, copy.deepcopy(PPO), log_dir=None, device="cpu")
            restored_infos = restored_runner.load(str(checkpoint), map_location="cpu")
            if restored_infos != infos:
                raise AssertionError("checkpoint provenance did not round-trip")
            restored = evaluate(restored_runner.get_inference_policy(), reloaded_adapter)
            record["restored_evaluation"] = restored
            np.testing.assert_allclose(restored["observations"], after["observations"], rtol=0, atol=1e-10)
            np.testing.assert_allclose(restored["actions"], after["actions"], rtol=0, atol=1e-7)
            np.testing.assert_allclose(restored["costs"], after["costs"], rtol=0, atol=1e-10)
            np.testing.assert_allclose(restored["accepted_joint_qv"], after["accepted_joint_qv"], rtol=0, atol=1e-10)
            np.testing.assert_allclose(restored["terminal_observations"], after["terminal_observations"], rtol=0, atol=1e-10)
            np.testing.assert_array_equal(restored["final_masks"], after["final_masks"])
        if before["mean_cost"] <= 1e-12:
            raise AssertionError("initial deterministic cost is not a meaningful comparison")
        improvement = 1 - after["mean_cost"] / before["mean_cost"]
        record.update(status="passed" if changed and improvement >= IMPROVEMENT else "learning_target_missed",
                      stage="finished", relative_cost_reduction=improvement,
                      checkpoint_evaluation_reproduced=True)
        return {"task": kind, "status": record["status"], "before_cost": before["mean_cost"],
                "after_cost": after["mean_cost"], "relative_cost_reduction": improvement,
                "actor_parameters_changed": changed, "checkpoint": record["checkpoint"],
                "checkpoint_evaluation_reproduced": True}
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        if task is not None:
            record["last_request"] = task.last_request
            record["last_transition"] = task.last_transition
        if reloaded_task is not None:
            record["reload_last_request"] = reloaded_task.last_request
            record["reload_last_transition"] = reloaded_task.last_transition
        with gzip.open(output / "complete_result.json.gz", "xt", encoding="utf-8") as stream:
            json.dump(serial(record), stream, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "phymeo-rsl-ppo-host-scenes-v1", "plan": PLAN, "tasks": [],
              "status": "starting", "python": sys.version, "platform": platform.platform(),
              "versions": {name: version(name) for name in ("torch", "rsl-rl-lib", "tensordict", "numpy")}}
    try:
        if os.environ.get("WORLD_SIZE", "1") != "1":
            raise ValueError("this CPU example requires one training process")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        package = Path(ph.__file__).resolve().parent
        provenance = {name: fact(path) for name, path in {
            "example": __file__, "adapter": package / "training/rsl_rl.py",
            "environment": package / "training/resident_env.py",
            "scene_backend": package / "training/scene_backend.py",
        }.items()}
        report["sources"] = provenance
        write_json(output / "declared_plan.json", {"plan": PLAN, "sources": provenance, "versions": report["versions"]})
        for kind in TASKS:
            report["tasks"].append(run_task(kind, output / kind, provenance))
        report["status"] = "passed" if all(item["status"] == "passed" for item in report["tasks"]) else "learning_target_missed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output / "summary.json", report)
        print(json.dumps({"status": report["status"], "output": str(output), "tasks": report["tasks"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
