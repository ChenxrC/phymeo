"""Fixed N9 HOST contact PPO holdout using the original public scene/environment.

AUTHORING_ONLY / UNEXECUTED. One fixed seed and 100x24 budget; no search,
parameter fitting, state repair, GPU-PPO claim or G1 training claim.
Run: python -B examples/train_contact_ppo.py --output-dir <new directory>
"""

import argparse
import copy
import gzip
import json
import math
import os
import platform
import sys
import traceback
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

import phymeo as ph
import train_scene_ppo as original
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand
from phymeo.sensors.contact_wrench import measure_contact_wrench
from phymeo.training import ResidentVectorEnv, resident_env, rsl_rl, scene_backend
from phymeo.training.rsl_rl import RslRlVecEnv
from phymeo.training.scene_backend import SceneBatchBackend, SceneControl

SEED, WORLDS = original.SEED, original.WORLDS
ITERATIONS, ROLLOUT, HORIZON = original.ITERATIONS, original.ROLLOUT, original.HORIZON
EVAL_STEPS, IMPROVEMENT = original.EVAL_STEPS, original.IMPROVEMENT
DT, RADIUS, DEPTH, ROBOT, FLOOR = 1 / 128, 1 / 16, 0.003, 10, 20
MASSES, GOALS = (0.5, 1.0), (-0.004, -0.008, -0.012)
LAW = ph.RobotContactMaterial(1000, 8, 0)
PLAN = {
    "status": "N9_declared_before_execution",
    "seed": SEED,
    "worlds": WORLDS,
    "mass_by_world_kg": [MASSES[index % 2] for index in range(WORLDS)],
    "goal_by_world_m": [GOALS[index % 3] for index in range(WORLDS)],
    "q0_m": 0.0,
    "v0_by_world_m_s": [0.02 if index % 2 == 0 else -0.02 for index in range(WORLDS)],
    "sphere_radius_m": RADIUS,
    "initial_depth_m": DEPTH,
    "sphere_origin_z_m": RADIUS - DEPTH,
    "sphere_inertia": "uniform solid sphere: I=2*m*r^2/5",
    "floor_half_extents_m": (1.0, 1.0, 0.05),
    "floor_position_m": (0.0, 0.0, -0.05),
    "gravity_m_s2": (0.0, 0.0, -9.81),
    "contact": {
        "stiffness_n_m": 1000, "damping_ns_m": 8, "friction": 0,
        "slip_speed_m_s": LAW.slip_speed_m_s,
    },
    "dt_s": DT,
    "episode_steps": HORIZON,
    "iterations": ITERATIONS,
    "rollout_steps_per_world": ROLLOUT,
    "evaluation_steps": EVAL_STEPS,
    "action": "clip(raw_action,-1,1)*12 N feedforward, zero KP/KD",
    "physical_effort_cap_n": 16,
    "joint_limits_m": (-8, 8),
    "joint_velocity_limit_m_s": 128,
    "observation": ("actual_q/.02", "actual_v/.5", "goal/.02", "mass"),
    "cost": "((actual_q-goal)/.02)^2+.01*(actual_v/.5)^2+.001*actual_drive_effort^2",
    "required_reduction_overall_and_each_mass": IMPROVEMENT,
    "checkpoint_evaluation": "192 steps in newly constructed original worlds",
    "ppo": original.PPO,
    "physics_backend": "HOST public RobotScene",
    "policy_backend": "Torch CPU / RSL-RL 5.5.2",
    "network_logging": False,
    "reaction_consistency_atol": 1e-10,
    "reload_numeric_atol": 1e-10,
    "reload_action_atol": 1e-7,
}


def save_gzip(path, value):
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb") as compressed:
            compressed.write(json.dumps(original.serial(value), allow_nan=False).encode("utf-8"))
        raw.flush()
        os.fsync(raw.fileno())


def urdf(mass):
    inertia = 2 * mass * RADIUS**2 / 5
    return f'''<robot name="contact_ppo_{mass}"><link name="base"/><link name="body">
<inertial><mass value="{mass}"/><inertia ixx="{inertia}" iyy="{inertia}" izz="{inertia}"
 ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="physical_ball"><geometry><sphere radius="{RADIUS}"/></geometry></collision></link>
<joint name="vertical" type="prismatic"><parent link="base"/><child link="body"/>
<origin xyz="0 0 {RADIUS - DEPTH}"/><axis xyz="0 0 1"/>
<limit lower="-8" upper="8" effort="16" velocity="128"/></joint></robot>'''


class ContactTask:
    def __init__(self):
        self.masses = np.array(PLAN["mass_by_world_kg"])
        self.goals = np.array(PLAN["goal_by_world_m"])
        self.v0 = np.array(PLAN["v0_by_world_m_s"])
        self.sources = {}
        self.last_request = self.last_transition = None

    def controls(self, actions, keys):
        requested = 12 * np.clip(actions[:, 0], -1, 1)
        controls = {
            key: SceneControl({ROBOT: RobotCommand([0.0], feedforward=[requested[index]])})
            for index, key in enumerate(keys)
        }
        self.last_request = {
            "actions": np.array(actions, copy=True),
            "requested_effort_n": requested,
            "controls": controls,
        }
        return controls

    @staticmethod
    def coordinates(states, keys):
        return (
            np.array([states[key].robots[0, ROBOT].q[0] for key in keys]),
            np.array([states[key].robots[0, ROBOT].v[0] for key in keys]),
        )

    def observe(self, states, keys):
        q, v = self.coordinates(states, keys)
        return np.column_stack((q / 0.02, v / 0.5, self.goals / 0.02, self.masses))

    def actual_effort(self, states, keys):
        return np.array([states[key].robots[0, ROBOT].drive_effort[0] for key in keys])

    def cost(self, states, keys):
        q, v = self.coordinates(states, keys)
        actual = self.actual_effort(states, keys)
        return ((q - self.goals) / 0.02) ** 2 + 0.01 * (v / 0.5) ** 2 + 0.001 * actual**2

    def contact_reading(self, key, reading):
        source = self.sources[key]
        events = source["scene"].contact_events
        if (
            events.step_index != reading.step_number
            or events.commit_time_s != reading.time_s
            or events.sample_time_s != reading.load_time_s
        ):
            raise AssertionError("contact events and actual accepted observation differ in time")
        if not math.isclose(reading.time_s - reading.load_time_s, DT, rel_tol=0, abs_tol=1e-12):
            raise AssertionError("contact load must be sampled at entry t, accepted state at t+h")
        ball, ground = source["ball"], source["ground"]
        robot = measure_contact_wrench(reading.loads, ball, counterparts=ground)
        floor = measure_contact_wrench(reading.loads, ground[0], counterparts=(ball,))
        published_floor = reading.loads.inputs.rigid.force[source["floor_row"]]
        # All come from public actual load channels, never the command or a witness-only proxy.
        np.testing.assert_allclose(robot.force_n + floor.force_n, 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(robot.torque_nm + floor.torque_nm, 0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(floor.force_n, published_floor, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            reading.robots[0, ROBOT].contact_effort[0], robot.force_n[2], rtol=0, atol=1e-10
        )
        return {
            "sample_time_s": reading.load_time_s,
            "commit_time_s": reading.time_s,
            "step_number": reading.step_number,
            "robot_wrench": robot,
            "ground_wrench": floor,
            "published_ground_force_n": published_floor,
            "pair_loads": reading.loads.contacts,
            "contact_details": reading.contacts,
            "contact_events": events,
        }

    def evaluate(self, before, after, actions, keys):
        self.last_transition = {
            "before": before, "after": after, "actions": np.array(actions, copy=True),
            "contact": {},
        }
        cost = self.cost(after, keys)
        self.last_transition.update(cost=cost, actual_effort_n=self.actual_effort(after, keys))
        # Retain the actual accepted prefix if a public diagnostic itself fails.
        for key in keys:
            self.last_transition["contact"][key] = self.contact_reading(key, after[key])
        return -cost, np.zeros(len(keys), dtype=bool)


def make_environment(paths):
    task = ContactTask()

    def factory(world):
        builder = ContactSceneBuilder(gravity=(0, 0, -9.81))
        builder.add_urdf(
            ROBOT, paths[float(task.masses[world])],
            robot_state=RobotState([0.0], [task.v0[world]]),
            kp=[0.0], kd=[0.0], effort_limit=[16.0],
        )
        builder.add_box(FLOOR, (1, 1, 0.05), mass=None, position=(0, 0, -0.05))
        builder.pair(ROBOT, FLOOR, LAW)
        scene = builder.build(backend="host")
        try:
            task.sources[world] = {
                "scene": scene,
                "ball": scene.shape_key(ROBOT, "body", "physical_ball"),
                "ground": scene.body_keys[0, FLOOR],
                "floor_row": scene.rigid_rows[0, FLOOR],
            }
        except Exception:
            scene.close()
            raise
        return scene

    backend = SceneBatchBackend(tuple(range(WORLDS)), factory)
    try:
        env = ResidentVectorEnv(
            backend, task, backend.world_keys, action_size=1, dt=DT, horizon=HORIZON
        )
        adapter = RslRlVecEnv(env, device="cpu")
    except Exception:
        backend.close()
        raise
    return backend, task, adapter


@contextmanager
def capture_backend_failure(backend, record, name):
    with backend:
        try:
            yield backend
        except BaseException as error:
            failure = {
                "type": type(error).__name__, "message": str(error),
                "accepted_worlds": getattr(error, "accepted_worlds", None),
                "acceptance_known": getattr(error, "acceptance_known", None),
                "physics_accepted": getattr(error, "physics_accepted", None),
            }
            try:
                failure["actual_public_states"] = backend.snapshot()
            except Exception as snapshot_error:
                failure["snapshot_error"] = repr(snapshot_error)
            record[name] = failure
            raise


def evaluate(policy, adapter, record, label):
    task, keys = adapter.env.task, adapter.env.world_keys
    observations = adapter.reset()
    initial_worlds = adapter.env.batch.snapshot(keys)
    frames = [observations["policy"].cpu().numpy().copy()]
    actions, costs, qv, efforts, terminal, masks, rewards, physics = [], [], [], [], [], [], [], []
    progress = {
        "label": label, "initial_worlds": initial_worlds, "observations": frames,
        "actions": actions, "costs": costs, "accepted_joint_qv": qv,
        "actual_effort_n": efforts, "terminal_observations": terminal,
        "final_masks": masks, "adapter_rewards": rewards, "physics_frames": physics,
    }
    record["evaluation_in_progress"] = progress
    with torch.inference_mode():
        for _ in range(EVAL_STEPS):
            action = policy(observations)
            progress["last_proposed_action"] = action.detach().cpu().numpy().copy()
            observations, reward, _, extras = adapter.step(action)
            transition = task.last_transition
            q, v = task.coordinates(transition["after"], keys)
            actions.append(action.detach().cpu().numpy().copy())
            costs.append(transition["cost"].copy())
            efforts.append(transition["actual_effort_n"].copy())
            qv.append(np.column_stack((q, v)))
            rewards.append(reward.cpu().numpy().copy())
            frames.append(observations["policy"].cpu().numpy().copy())
            terminal.append(extras["final_observation"]["policy"].cpu().numpy().copy())
            masks.append(extras["final_mask"].cpu().numpy().copy())
            physics.append({
                "accepted_worlds": transition["after"],
                "actual_contact": transition["contact"],
                "raw_actions": transition["actions"],
                "actual_effort_n": transition["actual_effort_n"],
                "actual_cost": transition["cost"],
            })
    cost_array = np.array(costs)
    contact = np.array([
        [frame["actual_contact"][key]["robot_wrench"].force_n[2] for key in keys]
        for frame in physics
    ])
    result = {
        "initial_worlds": initial_worlds,
        "mean_cost": float(cost_array.mean()),
        "mean_cost_by_mass": {
            str(mass): float(cost_array[:, task.masses == mass].mean()) for mass in MASSES
        },
        "observations": np.array(frames),
        "actions": np.array(actions),
        "costs": cost_array,
        "adapter_rewards": np.array(rewards),
        "terminal_observations": np.array(terminal),
        "final_masks": np.array(masks),
        "accepted_joint_qv": np.array(qv),
        "actual_effort_n": np.array(efforts),
        "actual_robot_contact_force_z_n": contact,
        "nonzero_contact_by_mass": {
            str(mass): bool(np.any(contact[:, task.masses == mass] > 0)) for mass in MASSES
        },
        "physics_frames": physics,
    }
    record[label] = result
    record.pop("evaluation_in_progress", None)
    return result


def run(output, provenance):
    plants = output / "plants"
    plants.mkdir()
    paths = {}
    for mass in MASSES:
        path = plants / ("sphere-" + str(mass) + "kg.urdf")
        path.write_text(urdf(mass), encoding="utf-8")
        paths[mass] = path
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    record = {
        "schema": "phymeo-contact-ppo-complete-v1",
        "plan": PLAN,
        "sources": provenance,
        "plants": {str(mass): original.fact(path) for mass, path in paths.items()},
        "plant_xml": {str(mass): urdf(mass) for mass in MASSES},
        "status": "starting",
    }
    task = restored_task = None
    try:
        backend, task, adapter = make_environment(paths)
        with capture_backend_failure(backend, record, "backend_failure"):
            runner = original.OnPolicyRunner(
                adapter, copy.deepcopy(original.PPO), log_dir=str(output / "logs"), device="cpu"
            )
            before = evaluate(runner.get_inference_policy(), adapter, record, "before")
            save_gzip(output / "before_evaluation.json.gz", before)
            record.update(before=before, stage="train")
            initial_actor = {
                name: value.detach().clone()
                for name, value in runner.alg.actor.state_dict().items()
            }
            adapter.reset()
            runner.learn(num_learning_iterations=ITERATIONS, init_at_random_ep_len=False)
            after = evaluate(runner.get_inference_policy(), adapter, record, "after")
            save_gzip(output / "after_evaluation.json.gz", after)
            changed = any(
                not torch.equal(value, runner.alg.actor.state_dict()[name])
                for name, value in initial_actor.items()
            )
            checkpoint = output / "trained.pt"
            infos = {
                "seed": SEED, "plan": PLAN, "sources": provenance,
                "environment_resume": "new original worlds; no mid-episode state resume claim",
            }
            runner.save(str(checkpoint), infos=infos)
            record.update(
                after=after, actor_parameters_changed=changed,
                checkpoint=original.fact(checkpoint), stage="reload",
            )
        restored_backend, restored_task, restored_adapter = make_environment(paths)
        with capture_backend_failure(restored_backend, record, "reload_backend_failure"):
            restored_runner = original.OnPolicyRunner(
                restored_adapter, copy.deepcopy(original.PPO), log_dir=None, device="cpu"
            )
            loaded_infos = restored_runner.load(str(checkpoint), map_location="cpu")
            if loaded_infos != infos:
                raise AssertionError("native checkpoint provenance did not round-trip")
            restored = evaluate(
                restored_runner.get_inference_policy(), restored_adapter, record, "restored"
            )
            save_gzip(output / "restored_evaluation.json.gz", restored)
            record["restored"] = restored
            for name in (
                "observations", "costs", "accepted_joint_qv", "actual_effort_n",
                "actual_robot_contact_force_z_n", "terminal_observations",
            ):
                np.testing.assert_allclose(restored[name], after[name], rtol=0, atol=1e-10)
            np.testing.assert_allclose(restored["actions"], after["actions"], rtol=0, atol=1e-7)
            np.testing.assert_array_equal(restored["final_masks"], after["final_masks"])
            if original.serial(restored["initial_worlds"]) != original.serial(
                after["initial_worlds"]
            ):
                raise AssertionError("reload did not construct the same fresh physical worlds")
            for first, second in zip(
                after["physics_frames"], restored["physics_frames"], strict=True
            ):
                for key in range(WORLDS):
                    if original.serial(first["actual_contact"][key]) != original.serial(
                        second["actual_contact"][key]
                    ):
                        raise AssertionError("restored actual contact/event trace differs")
        if before["mean_cost"] <= 1e-12 or any(
            value <= 1e-12 for value in before["mean_cost_by_mass"].values()
        ):
            raise AssertionError("initial cost is not a meaningful learning comparison")
        overall = 1 - after["mean_cost"] / before["mean_cost"]
        groups = {
            str(mass): 1 - after["mean_cost_by_mass"][str(mass)]
            / before["mean_cost_by_mass"][str(mass)]
            for mass in MASSES
        }
        contact_ok = all(
            all(evaluation["nonzero_contact_by_mass"].values())
            for evaluation in (before, after, restored)
        )
        passed = changed and overall >= IMPROVEMENT and all(
            value >= IMPROVEMENT for value in groups.values()
        ) and contact_ok
        record.update(
            status="passed" if passed else "learning_target_missed", stage="finished",
            relative_cost_reduction=overall, relative_cost_reduction_by_mass=groups,
            actual_public_contact_observed=contact_ok, checkpoint_evaluation_reproduced=True,
        )
        return {
            "status": record["status"],
            "before_cost": before["mean_cost"], "after_cost": after["mean_cost"],
            "before_cost_by_mass": before["mean_cost_by_mass"],
            "after_cost_by_mass": after["mean_cost_by_mass"],
            "relative_cost_reduction": overall, "relative_cost_reduction_by_mass": groups,
            "actor_parameters_changed": changed, "actual_public_contact_observed": contact_ok,
            "checkpoint": record["checkpoint"], "checkpoint_evaluation_reproduced": True,
        }
    except BaseException as error:
        record.update(
            status="failed", error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc(),
        )
        raise
    finally:
        if task is not None:
            record.update(last_request=task.last_request, last_transition=task.last_transition)
        if restored_task is not None:
            record.update(
                reload_last_request=restored_task.last_request,
                reload_last_transition=restored_task.last_transition,
            )
        save_gzip(output / "complete_result.json.gz", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "phymeo-rsl-ppo-host-contact-v1", "plan": PLAN, "status": "starting",
        "python": sys.version, "platform": platform.platform(),
        "versions": {
            name: version(name) for name in ("torch", "rsl-rl-lib", "tensordict", "numpy")
        },
        "actual_module_files": {
            "phymeo": ph.__file__, "torch": torch.__file__,
            "training_package": sys.modules["phymeo.training"].__file__,
            "rsl_runner": sys.modules[original.OnPolicyRunner.__module__].__file__,
            "resident_env": resident_env.__file__, "rsl_adapter": rsl_rl.__file__,
            "scene_backend": scene_backend.__file__, "application_helper": original.__file__,
        },
    }
    try:
        if os.environ.get("WORLD_SIZE", "1") != "1":
            raise ValueError("this fixed CPU holdout requires one training process")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        provenance = {
            name: original.fact(path)
            for name, path in {
                "example": __file__, "application_helper": original.__file__,
                "training_package": sys.modules["phymeo.training"].__file__,
                "resident_env": resident_env.__file__, "rsl_adapter": rsl_rl.__file__,
                "scene_backend": scene_backend.__file__,
            }.items()
        }
        report["sources"] = provenance
        original.write_json(
            output / "declared_plan.json",
            {
                "plan": PLAN, "sources": provenance,
                "actual_module_files": report["actual_module_files"],
                "versions": report["versions"],
            },
        )
        report.update(run(output, provenance))
    except BaseException as error:
        report.update(
            status="failed", error_type=type(error).__name__,
            error=str(error), traceback=traceback.format_exc(),
        )
        raise
    finally:
        original.write_json(output / "summary.json", report)
        print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
