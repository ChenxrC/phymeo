"""Continuous PushT goal feedback; proposals and native paths never advance physics."""

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

from phymeo.control.manipulation_diagnostic import NativeJointPD
from phymeo.control.manipulation_push_objective import propose_planar_pushes
from phymeo.control.manipulation_push_planner import NativePushRoutePlanner
from phymeo.control.manipulation_push_progress import BoundedPushProgress
from phymeo.control.manipulation_wrench import measure_object_contact_wrench
from phymeo.training.manipulation_tasks import H

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_manipulation_control import plain, reference, run  # noqa: E402
from run_manipulation_push import SOURCE_SHA, model_digest, restore_query  # noqa: E402


def hold_plan(scene, model):
    names = scene.robot_joint_names
    qa = [int(model.jnt_qposadr[model.joint(n).id]) for n in names]
    kp = [250.0 if k == "P" else 25.0 for k in scene.robot_kinds]
    kd = [12.0 if k == "P" else 3.0 for k in scene.robot_kinds]
    q0 = model.qpos0[qa].copy()
    return (
        NativeJointPD(model, names, kp, kd),
        ((0.0, "settle", q0), (0.2, "settle", q0)),
        dict(
            valid=True,
            qualification="initial reference hold only; future routes individually sampled",
            kp=kp,
            kd=kd,
            goal=scene.goals[scene.default_goal],
        ),
    )


class PushFeedbackDriver:
    """One sequential owner, one Env episode, bounded native HOST proposals."""

    def __init__(self, scene, model, output, saved):
        if model_digest(model) != saved["summary"]["model_sha256"]:
            raise ValueError("feedback native model differs from candidate input")
        self.scene, self.model, self.output = scene, model, output
        output.mkdir()
        self.query = restore_query(saved["query"])
        if not np.array_equal(self.query.q, model.qpos0):
            raise ValueError("feedback reference query is not original q0")
        self.object_body = model.body("object").id
        self.pusher = model.geom("pusher").id
        self.qa = [int(model.jnt_qposadr[model.joint(n).id]) for n in scene.robot_joint_names]
        self.goal = scene.goals[scene.default_goal]
        self.target = next(t for t in scene.targets if t.key == self.goal.target_key)
        self.ref_position, self.ref_rotation, _ = self.pose(model.qpos0)
        self.planner = NativePushRoutePlanner(
            model,
            scene.robot_joint_names,
            object_geom_names=scene.object_geom_names,
            pusher_geom_name="pusher",
        )
        self.monitor = BoundedPushProgress()
        self.initial_cost = self.last_cost = self.cost(model.qpos0)
        self.active = None
        self.previous_direction = None
        self.segment_impulse = 0.0
        self.total_impulse = 0.0
        self.tick = 0
        self.last_q = model.qpos0.copy()
        self.stopped = False
        self.reason = "continue"
        self.routes = []
        self.completed = 0
        self.hold = model.qpos0[self.qa].copy()

    def pose(self, q):
        scratch = mj.MjData(self.model)
        scratch.qpos[:] = q
        mj.mj_kinematics(self.model, scratch)
        return (
            scratch.xpos[self.object_body].copy(),
            scratch.xmat[self.object_body].reshape(3, 3).copy(),
            scratch.geom_xpos[self.pusher].copy(),
        )

    def cost(self, q):
        pos, rot, _ = self.pose(q)
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        error = (self.target.yaw - yaw + math.pi) % (2 * math.pi) - math.pi
        return float(
            np.sum(
                ((np.asarray(self.target.position[:2]) - pos[:2]) / self.goal.position_tolerance_m)
                ** 2
            )
            + (error / self.goal.yaw_tolerance_rad) ** 2
        )

    def reference(self, tick, q, v):
        if not self.monitor.started:
            if tick != 0:
                raise ValueError("feedback episode must start at tick zero")
            self.last_q = q.copy()
            self.initial_cost = self.last_cost = self.cost(q)
            self.monitor.start(0, self.initial_cost)
        if tick != self.tick or not np.array_equal(q, self.last_q):
            raise ValueError("feedback reference is not its accepted endpoint")
        if tick < 200:
            return self.hold.copy(), np.zeros_like(self.hold), "settle", dict(segment=0)
        if self.active is None and not self.stopped:
            pos, rot, pusher = self.pose(q)
            tilt = math.acos(float(np.clip(rot[2, 2], -1, 1)))
            if tilt > self.goal.tilt_tolerance_rad:
                self.stopped, self.reason = True, "unsupported_tilt"
            else:
                proposals = propose_planar_pushes(
                    self.query,
                    reference_position=self.ref_position,
                    reference_rotation=self.ref_rotation,
                    current_position=pos,
                    current_rotation=rot,
                    target_xy=self.target.position[:2],
                    target_yaw=self.target.yaw,
                    pusher_height_m=float(pusher[2]),
                    position_tolerance_m=self.goal.position_tolerance_m,
                    yaw_tolerance_rad=self.goal.yaw_tolerance_rad,
                )
                route = self.planner.plan(q, proposals, previous_direction=self.previous_direction)
                path = self.output / f"route-{len(self.routes) + 1:02}.json.gz"
                with gzip.open(path, "xt", encoding="utf-8") as stream:
                    json.dump(
                        plain(
                            dict(
                                tick=tick,
                                q=q,
                                v=v,
                                cost=self.cost(q),
                                proposals=proposals,
                                route=route,
                            )
                        ),
                        stream,
                        allow_nan=False,
                    )
                self.routes.append(
                    dict(
                        file=path.name,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        tick=tick,
                        status=route.status,
                        ik_calls=route.ik_calls,
                        path_calls=route.path_calls,
                    )
                )
                if route.status != "sampled_reference_ready":
                    self.stopped, self.reason = True, route.status
                else:
                    self.active = (tick, route)
        if self.stopped:
            return q[self.qa].copy(), np.zeros(len(self.qa)), "stopped", dict(reason=self.reason)
        start, route = self.active
        desired, velocity, stage = reference(route.nodes, (tick - start) * H)
        return (
            desired,
            velocity,
            stage,
            dict(
                segment=self.completed + 1,
                route=len(self.routes),
                route_start_tick=start,
                route_elapsed_s=(tick - start) * H,
                selected_index=route.proposal.index,
                selected_distance_m=route.proposal.distance_m,
                predicted_improvement=route.proposal.predicted_improvement,
            ),
        )

    def accepted(self, tick, q, arrays, result):
        if tick != self.tick + 1:
            raise ValueError("feedback accepted ticks are not adjacent")
        entry_q = arrays["entry_q"][0, : self.model.nq]
        if not np.array_equal(entry_q, self.last_q):
            raise ValueError("feedback force entry q differs from preceding endpoint")
        pos, rot, _ = self.pose(entry_q)
        com = pos + rot @ self.model.body_ipos[self.object_body]
        measurement = measure_object_contact_wrench(
            arrays,
            0,
            object_body=self.object_body,
            object_com_world=com,
            group_geom_ids=(self.pusher,),
        )
        return self.accepted_measurement(tick, q, measurement, result, input_q=entry_q)

    def accepted_measurement(self, tick, q, measurement, result, *, input_q):
        """Advance progress once using a wrench on the object from this accepted entry."""
        if tick != self.tick + 1 or not np.array_equal(input_q, self.last_q):
            raise ValueError("feedback measurement differs from preceding accepted endpoint")
        if measurement is None or (measurement.input_tick, measurement.output_tick) != (
            tick - 1,
            tick,
        ):
            raise ValueError("feedback lacks preceding accepted force entry")
        force = np.asarray(measurement.group_force_world, dtype=np.float64)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError("feedback requires a finite object force in world coordinates")
        impulse = float(np.linalg.norm(force)) * H
        if not math.isfinite(impulse):
            raise ValueError("nonfinite feedback impulse")
        endpoint = np.asarray(q)
        if endpoint.shape != (self.model.nq,) or not np.isfinite(endpoint).all():
            raise ValueError("feedback requires a finite accepted endpoint")
        cost = self.cost(endpoint)
        if not math.isfinite(cost):
            raise ValueError("nonfinite feedback endpoint cost")
        self.segment_impulse += impulse
        self.total_impulse += impulse
        self.tick, self.last_q = tick, endpoint.copy()
        self.last_cost = cost
        receipt = None
        complete = (
            self.active is not None
            and (tick - self.active[0]) * H >= self.active[1].nodes[-1][0] - 1e-12
        )
        if complete or result.success or tick >= self.monitor.policy.max_steps:
            receipt = self.monitor.observe(
                tick,
                self.last_cost,
                contact_impulse_ns=self.segment_impulse,
                goal_success=bool(result.success),
            )
            if complete:
                self.previous_direction = self.active[1].proposal.direction_world
            if complete:
                self.completed += 1
            if complete:
                self.segment_impulse = 0.0
                self.active = None
            if self.monitor.terminal:
                self.stopped, self.reason = True, self.monitor.status
        return dict(
            measurement=measurement,
            progress=receipt,
            completed_route=complete,
            current_cost=self.last_cost,
            segment_impulse_ns=self.segment_impulse,
            total_impulse_ns=self.total_impulse,
            status=self.reason,
        )

    def summary(self):
        return dict(
            status=self.reason,
            tick=self.tick,
            initial_cost=self.initial_cost,
            final_cost=self.last_cost,
            completed_routes=self.completed,
            routes=self.routes,
            progress=self.monitor.history,
            policy=self.monitor.policy,
            partial_route=None
            if self.active is None
            else dict(
                start_tick=self.active[0],
                duration_s=self.active[1].nodes[-1][0],
                elapsed_s=(self.tick - self.active[0]) * H,
                contact_impulse_ns=self.segment_impulse,
            ),
            total_contact_impulse_ns=self.total_impulse,
            ik_calls=self.planner.ik_calls,
            path_calls=self.planner.path_calls,
            qualification="HOST goal-feedback references; original CUDA physics and Goal; no learned policy",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", type=int, choices=(5, 6, 7), required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("candidate dataset binding differs")
    saved = json.loads(args.source.read_text(encoding="utf-8"))
    if saved["schema"] != "original-push-boundary-endpoints-v1":
        raise ValueError("candidate schema differs")
    rows = {row["summary"]["world"]: row for row in saved["rows"]}

    def factory(scene, model, output):
        return PushFeedbackDriver(scene, model, output, rows[args.world])

    run(
        args.output,
        12000,
        (args.world,),
        plan_builder=hold_plan,
        record_entry=True,
        reference_driver_factory=factory,
    )


if __name__ == "__main__":
    main()
