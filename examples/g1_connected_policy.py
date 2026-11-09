"""Run the official G1 policy through the public environment/runner connection.

This is a HOST/NumPy adapter, not a zero-copy training or GPU-contact claim.
The original G1 protocol and engineering gates are shared with the standalone
reference harness. Only physical scene steps produce published q/v.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

import phymeo as ph
import g1_phymeo_support as original
from phymeo.builders.floating_mjcf import load_floating_mjcf
from phymeo.control.policy_runner import PolicyRunner
from phymeo.experiments.humanoid_support import read_foot_support
from phymeo.training import ResidentVectorEnv
from phymeo.training.scene_backend import SceneBatchBackend, SceneControl


class G1Task:
    """Asset-specific observation/torque mapping over original scene readings.

    Pending actions are inputs, not accepted policy history. An action becomes
    the previous action only when an observation has a greater accepted step
    number. Rejected attempts cannot advance that history. A rebuilt scene's
    step zero clears it. No q/v is ever written by this adapter.
    """

    def __init__(self, assets, dt, journal):
        self.assets, self.dt, self.journal = assets, dt, journal
        self.asset = load_floating_mjcf(assets / "g1_primitive_actuated.xml")
        config = yaml.safe_load((assets / "deploy.yaml").read_text(encoding="utf-8"))
        (self.names, self.p_from_o, self.o_from_p, self.default,
         self.kp, self.kd, self.scale, self.offset) = original.protocol(
             config, self.asset, assets / "g1_primitive_actuated.xml")
        with np.load(assets / "dance1_subject2.npz", allow_pickle=False) as data:
            self.motion = {name: data[name].copy() for name in data.files}
        self.decimation = round(float(config["step_dt"]) / dt)
        if self.decimation < 1 or not math.isclose(self.decimation * dt, .02, abs_tol=1e-12):
            raise ValueError("physical dt must divide the official .02 second control period")
        self.colliders = original.foot_colliders(self.asset)
        self.initial, self.placement = original.initial_state(
            "policy", self.asset, self.default, self.p_from_o, self.colliders, self.motion)
        self.alignment = original.yaw(original.torso(
            self.initial.q[[6, 3, 4, 5]], self.initial.q[7:][self.o_from_p])) @ original.yaw(
                original.rotation(self.motion["body_quat_w"][0, 0])).T
        self.cap = np.asarray(self.asset.effort_limit, dtype=float)
        self.context, self.states, self.seen, self.pending, self.previous = {}, {}, {}, {}, {}
        self.last_transition = {}
        self.weight = original.GRAVITY * (
            1 / self.asset.model.root.inverse_mass[0] + self.asset.model.children.mass.sum())

    def factory(self, world):
        builder = ph.ContactSceneBuilder(gravity=(0, 0, -original.GRAVITY))
        builder.add_floating_urdf(
            original.ROBOT, self.assets / "g1_primitive_actuated.xml", loader=load_floating_mjcf,
            robot_state=self.initial, kp=np.zeros(29), kd=np.zeros(29), effort_limit=self.cap)
        builder.add_box(original.FLOOR, (10, 10, .1), mass=None, position=(0, 0, -.1))
        builder.pair(original.ROBOT, original.FLOOR,
                     ph.RobotContactMaterial(stiffness=50000, damping=100, friction=1))
        scene = builder.build(backend="host")
        self.context[world] = dict(
            feet={name: tuple(scene.shape_key(original.ROBOT, row.link_name, row.name)
                              for row in rows) for name, rows in self.colliders.items()},
            ground=scene.body_keys[0, original.FLOOR],
            floor_row=scene.rigid_rows[0, original.FLOOR])
        return scene

    def observe(self, states, keys):
        observations = []
        for key in keys:
            reading = states[key]
            tick = reading.step_number
            if tick == 0:
                self.previous[key] = np.zeros(29)
            elif tick > self.seen.get(key, -1):
                self.previous[key] = self.pending[key].copy()
            self.seen[key], self.states[key] = tick, reading
            row = reading.robots[0, original.ROBOT]
            frame = min(tick // self.decimation, len(self.motion["joint_pos"]) - 1)
            relative = original.torso(row.q[[6, 3, 4, 5]], row.q[7:][self.o_from_p]).T @ (
                self.alignment @ original.torso(self.motion["body_quat_w"][frame, 0],
                                               self.motion["joint_pos"][frame]))
            observations.append(np.r_[
                self.motion["joint_pos"][frame], self.motion["joint_vel"][frame],
                relative[:, :2].reshape(-1),
                original.rotation(row.q[[6, 3, 4, 5]]).T @ row.v[3:6],
                row.q[7:][self.o_from_p] - self.default, row.v[6:][self.o_from_p],
                self.previous[key]])
        return np.asarray(observations, dtype=np.float32)

    def controls(self, actions, keys):
        result = {}
        for key, action in zip(keys, actions, strict=True):
            self.pending[key] = np.array(action, copy=True)
            row = self.states[key].robots[0, original.ROBOT]
            target = self.offset + self.scale * action
            torque = (self.kp * (target - row.q[7:][self.o_from_p])
                      - self.kd * row.v[6:][self.o_from_p])[self.p_from_o]
            packet = ph.FloatingRobotCommand(ph.RobotCommand(
                np.clip(target[self.p_from_o], self.asset.lower, self.asset.upper),
                feedforward=np.clip(torque, -self.cap, self.cap)))
            result[key] = SceneControl(actions={original.ROBOT: packet})
        self.journal.append("control_request", entry_steps={key: self.seen[key] for key in keys},
                            raw_actions=actions, controls={key: {
                                "actions": packet.actions, "external": packet.external,
                            } for key, packet in result.items()})
        return result

    def evaluate(self, before, after, actions, keys):
        rewards, terminated = [], []
        for key in keys:
            reading = after[key]
            row = reading.robots[0, original.ROBOT]
            # Match the standalone journal's held control-frame reference for
            # this completed substep, without floating clock rounding.
            frame = min((reading.step_number - 1) // self.decimation,
                        len(self.motion["joint_pos"]) - 1)
            error = float(np.sqrt(np.mean((row.q[7:][self.o_from_p]
                                          - self.motion["joint_pos"][frame]) ** 2)))
            up = original.rotation(row.q[[6, 3, 4, 5]])[:, 2]
            tilt = float(np.arctan2(np.linalg.norm(up[:2]), up[2]))
            self.last_transition[key] = dict(reading=reading, error=error, tilt=tilt, frame=frame)
            rewards.append(-error * error)
            terminated.append(row.q[2] < original.CRITERIA["minimum_root_height_m"]
                              or tilt > original.CRITERIA["maximum_fall_tilt_rad"])
        return np.asarray(rewards), np.asarray(terminated, dtype=bool)


def statistics():
    return dict(accepted_steps=0, accepted_duration_s=0., minimum_root_height_m=float("inf"),
                maximum_root_tilt_rad=0., maximum_root_xy_drift_m=0.,
                maximum_foot_center_xy_drift_m=0., maximum_reference_xy_error_m=0.,
                maximum_joint_rmse_rad=0., maximum_force_bodyweight_ratio=0.,
                maximum_nonfoot_ground_force_n=0., settled_samples=0,
                settled_vertical_load_sum=0., both_loaded_samples=0,
                effort_caps_observed=True, reaction_balance_observed=True)


def run(args, journal):
    import onnxruntime as ort

    task = G1Task(args.assets.resolve(), args.dt, journal)
    steps = round(args.duration / args.dt)
    if steps < 1 or not math.isclose(steps * args.dt, args.duration, abs_tol=1e-12):
        raise ValueError("duration must contain whole physical steps")
    if math.ceil(args.duration / .02) > len(task.motion["joint_pos"]):
        raise ValueError("requested duration exceeds the unchanged motion")
    session = ort.InferenceSession(str(task.assets / "policy.onnx"), providers=["CPUExecutionProvider"])
    keys = tuple(range(args.worlds))
    stats = {key: statistics() for key in keys}
    initial_feet = original.foot_geometry(task.asset, task.initial.q, task.colliders)
    journal.context["statistics_by_world"] = stats
    sources = {
        name: original.source_record(task.assets / name) for name in
        ("g1_primitive_actuated.xml", "deploy.yaml", "dance1_subject2.npz", "policy.onnx")}
    if (task.assets / "policy.onnx.data").exists():
        sources["policy.onnx.data"] = original.source_record(task.assets / "policy.onnx.data")
    journal.append("initialization", schema="g1-connected-policy-v1", sources=sources,
        initial_q=task.initial.q, initial_v=task.initial.v, dt_s=args.dt,
        physical_joint_names=task.asset.joint_names,
        control_decimation=task.decimation, bodyweight_n=task.weight,
        criteria=original.CRITERIA, backend="HOST_scene_NumPy_CPU_ONNX",
        connection="SceneBatchBackend -> ResidentVectorEnv -> PolicyRunner")

    def policy(observations, due):
        journal.append("policy_request", observation=observations, due_mask=due)
        actions = np.zeros((len(keys), 29))
        for index in np.flatnonzero(due):
            raw = session.run(["actions"], {
                "obs": observations[index:index + 1].astype(np.float32)})[0][0]
            actions[index] = original.vector(raw, 29, "actual ONNX action")
        journal.append("policy_response", due_mask=due, actions=actions)
        return actions

    with SceneBatchBackend(keys, task.factory) as backend:
        env = ResidentVectorEnv(backend, task, keys, action_size=29, dt=args.dt,
                                horizon=steps + 1)
        runner = PolicyRunner(env, policy, control_decimation=task.decimation,
                              initial_observation=task.observe(backend.snapshot(), keys))
        for tick in range(steps):
            journal.append("attempt", tick=tick,
                           states={key: original.state_record(task.states[key].robots[0, original.ROBOT])
                                   for key in keys})
            try:
                result = runner.step()
            except Exception:
                # evaluate() may retain a real endpoint before an automatic reset
                # or reset observer fails. Preserve it before reading the current
                # backend, which may already expose a rebuilt step-zero scene.
                if task.last_transition:
                    endpoints = {key: {
                        "step_number": item["reading"].step_number,
                        "time_s": item["reading"].time_s,
                        "state": original.state_record(item["reading"].robots[0, original.ROBOT]),
                    } for key, item in task.last_transition.items()}
                    journal.context["last_accepted"] = endpoints
                    journal.append("physical_publication", publications=endpoints,
                                   environment_transition=False)
                try:
                    publications = {key: {
                        "step_number": reading.step_number, "time_s": reading.time_s,
                        "robots": {str(address): original.state_record(row)
                                   for address, row in reading.robots.items()},
                    } for key, reading in backend.snapshot().items()}
                    journal.context["failure_publications"] = publications
                    journal.append("failure_publication_snapshot", publications=publications)
                except Exception as snapshot_error:
                    journal.context["failure_snapshot_error"] = str(snapshot_error)
                raise
            journal.context["last_accepted"] = {key: {
                "step_number": item["reading"].step_number,
                "time_s": item["reading"].time_s,
                "state": original.state_record(item["reading"].robots[0, original.ROBOT]),
            } for key, item in task.last_transition.items()}
            for key in keys:
                item, context = task.last_transition[key], task.context[key]
                reading = item["reading"]
                row = reading.robots[0, original.ROBOT]
                support = read_foot_support(
                    reading, context["feet"], context["ground"],
                    floor_rigid_rows=(context["floor_row"],), robot_address=(0, original.ROBOT),
                    reference_root_position_m=task.initial.q[:3], positive_load_threshold_n=1.)
                geometry = original.foot_geometry(task.asset, row.q, task.colliders)
                root_reference = task.initial.q[:3] + task.alignment @ (
                    task.motion["body_pos_w"][item["frame"], 0]
                    - task.motion["body_pos_w"][0, 0])
                force = np.asarray(support.total_force_n)
                reaction = np.asarray(support.selected_ground_force_n)
                stat = stats[key]
                stat["accepted_steps"] += 1
                stat["accepted_duration_s"] = reading.time_s
                stat["minimum_root_height_m"] = min(stat["minimum_root_height_m"], float(row.q[2]))
                for name, value in (
                    ("maximum_root_tilt_rad", item["tilt"]),
                    ("maximum_root_xy_drift_m", np.linalg.norm(row.q[:2] - task.initial.q[:2])),
                    ("maximum_foot_center_xy_drift_m", max(np.linalg.norm(
                        geometry[name]["capsule_endpoint_center_m"][:2]
                        - initial_feet[name]["capsule_endpoint_center_m"][:2]) for name in geometry)),
                    ("maximum_reference_xy_error_m", np.linalg.norm(row.q[:2] - root_reference[:2])),
                    ("maximum_joint_rmse_rad", item["error"]),
                    ("maximum_force_bodyweight_ratio", np.linalg.norm(force) / task.weight),
                    ("maximum_nonfoot_ground_force_n", np.linalg.norm(
                        np.asarray(support.published_floor_force_n) - reaction)),
                ):
                    stat[name] = max(stat[name], float(value))
                stat["effort_caps_observed"] &= bool(np.all(np.abs(row.drive_effort) <= task.cap + 1e-9))
                stat["reaction_balance_observed"] &= bool(np.linalg.norm(force + reaction) <= 1e-8 * task.weight)
                if support.sample_time_s >= .25:
                    stat["settled_samples"] += 1
                    stat["settled_vertical_load_sum"] += float(force[2] / task.weight)
                    stat["both_loaded_samples"] += int(len(support.loaded_feet) == 2)
                journal.append("accepted", tick=tick, world=key, time_s=reading.time_s,
                               state=original.state_record(row), support=support,
                               foot_contacts=original.foot_contacts(reading, context["feet"], context["ground"]),
                               joint_rmse_rad=item["error"], policy_result=result)
                if not stat["effort_caps_observed"] or not stat["reaction_balance_observed"]:
                    raise RuntimeError("accepted state failed original effort or reaction gate")
            if any(row["tilt"] > original.CRITERIA["maximum_fall_tilt_rad"]
                   or row["reading"].robots[0, original.ROBOT].q[2] < .35
                   for row in task.last_transition.values()):
                raise RuntimeError("actual connected-policy trajectory fell; no continued reset episode")
    assessments = {key: original.evaluate("policy", value, True, None) for key, value in stats.items()}
    success = all(item["success"] for item in assessments.values())
    failure = None if success else dict(
        type="EngineeringAssessment", message="original engineering gates failed",
        failed_criteria={key: [name for name, passed in item["gates"].items() if not passed]
                         for key, item in assessments.items()})
    if failure is not None:
        journal.append("first_failure", **failure)
    return dict(schema="g1-connected-policy-summary-v1", duration_s=args.duration,
                statistics_by_world=stats, assessments=assessments,
                success=success, first_failure=failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--dt", type=float, default=.0005)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.worlds <= 32 or not np.isfinite([args.dt, args.duration]).all() or min(args.dt, args.duration) <= 0:
        parser.error("positive finite dt/duration and 1..32 HOST logical worlds required")
    journal = original.Journal(args.output)
    try:
        journal.append("request", arguments=vars(args), status="UNVALIDATED_NEW_CONNECTION")
        try:
            report = run(args, journal)
        except Exception as error:
            import traceback
            report = dict(success=False, error=type(error).__name__, message=str(error),
                          traceback=traceback.format_exc(), context=journal.context,
                          accepted_worlds=getattr(error, "accepted_worlds", None),
                          acceptance_known=getattr(error, "acceptance_known", None),
                          physics_accepted=getattr(error, "physics_accepted", None))
            journal.append("first_failure", **report)
        journal.finish(report)
    finally:
        journal.close()
    print(original.json_value(report))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
