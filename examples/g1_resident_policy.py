"""Official G1 policy on resident TYX floating dynamics and finite floor contact.

Initial holdout::

    python examples/g1_resident_policy.py --assets PATH --duration 2 \
        --tyx-mode cuda --cache-root PATH --output resident-g1.jsonl

The unchanged G1Task supplies the original 154-value ONNX protocol and engineering
criteria. Its scene factory is never called. Public resident checkpoints supply
all moving q/v; each .0005 s step recomputes explicit capped PD and compliant
contact, while PolicyRunner holds the original raw action for 40 physical steps.
The original 29-joint/30-body MJCF, 33 physical sphere/capsule colliders, root
origin/COM velocity conversion, masses, passive actuators and limits are kept.
The finite floor and explicit robot-internal exclusions match the HOST example.

This is a mixed pipeline: TYX dynamics/FK/bounds, HOST finite narrow phase and
contact law, HOST NumPy observation copies, CPU ONNX. It is not device-only
collision, GPU policy inference, self-contact, CCD or a validated long dance.
Full requests, accepted contact receipts and the first failed publication mask
are journaled. No state is fitted, repaired or injected between physical steps.
AUTHORING_ONLY checkpoint: this new backend connection is UNEXECUTED.
"""

import argparse
import math
import traceback
from dataclasses import replace
from itertools import combinations, product
from pathlib import Path
from types import SimpleNamespace

import g1_phymeo_support as original
import numpy as np
from g1_connected_policy import G1Task, statistics

import phymeo as ph
from phymeo.control.policy_runner import PolicyRunner
from phymeo.training import ResidentVectorEnv
from phymeo.training.resident_stepper import ResidentContactStepper

WORLD, H = 0, 0.0005


def policy_reading(checkpoint):
    """Read-only protocol view of an actual published device checkpoint."""
    row = SimpleNamespace(
        q=np.asarray(checkpoint.position, dtype=float),
        v=np.asarray(checkpoint.velocity, dtype=float),
    )
    return SimpleNamespace(
        step_number=checkpoint.tick,
        time_s=checkpoint.tick * H,
        robots={(0, original.ROBOT): row},
    )


class ResidentG1Task(G1Task):
    """Reuse the official policy protocol; translate only the backend boundary."""

    def __init__(self, assets, journal):
        super().__init__(assets, H, journal)
        self.tree = ph.FloatingRobotTree(
            self.asset.model,
            self.initial,
            kp=np.zeros(29),
            kd=np.zeros(29),
            effort_limit=self.cap,
        )
        self.last_controls = {}

    def observe(self, states, keys):
        return super().observe({key: policy_reading(states[key]) for key in keys}, keys)

    def evaluate(self, before, after, actions, keys):
        return super().evaluate(
            {key: policy_reading(before[key]) for key in keys},
            {key: policy_reading(after[key]) for key in keys},
            actions,
            keys,
        )

    def controls(self, actions, keys):
        controls, requests = {}, {}
        for key, action in zip(keys, actions, strict=True):
            self.pending[key] = np.array(action, copy=True)
            row = self.states[key].robots[0, original.ROBOT]
            target = self.offset + self.scale * action
            torque = (
                self.kp * (target - row.q[7:][self.o_from_p]) - self.kd * row.v[6:][self.o_from_p]
            )[self.p_from_o]
            drive = np.column_stack(
                (
                    np.clip(target[self.p_from_o], self.asset.lower, self.asset.upper),
                    np.zeros(29),
                    np.clip(torque, -self.cap, self.cap),
                )
            )
            controls[key] = self.tree.inputs(drive)
            requests[key] = dict(
                entry_q=row.q,
                entry_v=row.v,
                raw_action=action,
                raw_target_policy_order=target,
                raw_torque_physical_order_nm=torque,
                actual_device_input=controls[key],
            )
        self.last_controls = controls
        self.journal.context["last_control_request"] = requests
        self.journal.append("control_request", entry_steps=self.seen, requests=requests)
        return controls


def contact_registration(task):
    """Original material vertices, principal frames and model-token ownership."""
    asset = task.asset
    if asset.model.body_count != 30 or len(asset.joint_names) != 29:
        raise ValueError("the original 29-joint G1 physical tree is required")
    if len(asset.collisions) != 33 or any(
        row.shape.kind not in ("sphere", "capsule") for row in asset.collisions
    ):
        raise ValueError("the unchanged 33 G1 sphere/capsule colliders are required")
    shapes, bindings, by_name = [], [], {}
    for ordinal, collider in enumerate(asset.collisions, 1):
        key = ph.CollisionKey(WORLD, original.ROBOT, ordinal)
        shape = replace(collider.shape, key=key)
        shapes.append(shape)
        bindings.append(
            ph.ResidentFloatingTreeBinding.from_asset(
                asset, collider.shape.key, key=key, world=WORLD
            )
        )
        by_name[(collider.link_name, collider.name)] = key
    floor = ph.CollisionKey(WORLD, original.FLOOR, 1)
    floor_shape = ph.CollisionShape(floor, "hull", tuple(range(10001, 10009)))
    floor_points = np.array(list(product((-10.0, 10.0), (-10.0, 10.0), (-0.2, 0.0))))
    bindings.append(ph.ResidentStaticBinding(floor, ph.CollisionMotion(floor_points)))
    robot_keys = tuple(shape.key for shape in shapes)
    model = ph.CollisionModel(tuple(shapes) + (floor_shape,), tuple(combinations(robot_keys, 2)))
    law = ph.RobotContactMaterial(stiffness=50000, damping=100, friction=1)
    materials = {(key, floor): law for key in robot_keys}
    feet = {
        side: tuple(by_name[(row.link_name, row.name)] for row in rows)
        for side, rows in task.colliders.items()
    }
    return model, tuple(bindings), materials, feet, floor


def foot_geometry(frames, feet):
    """Use accepted device FK material points, never a separate HOST trajectory."""
    result = {}
    for side, keys in feet.items():
        result[side] = dict(
            capsule_endpoint_center_m=np.vstack([frames[key].world_points_m for key in keys]).mean(
                axis=0
            ),
            sole_height_m=min(
                float(np.min(frames[key].world_points_m[:, 2]) - frames[key].shape.radius)
                for key in keys
            ),
        )
    return result


def foot_support(receipt, feet, floor):
    by_foot, reaction = {side: np.zeros(3) for side in feet}, np.zeros(3)
    for site in receipt.sites:
        first, second = site.pair
        for side, keys in feet.items():
            if first in keys and second == floor:
                by_foot[side] += site.first_force_n
                reaction += site.second_force_n
            elif second in keys and first == floor:
                by_foot[side] += site.second_force_n
                reaction += site.first_force_n
    published = np.zeros(3)
    if floor in receipt.static_reactions:
        published = np.asarray(receipt.static_reactions[floor].force_n)
    return dict(
        sample_time_s=receipt.entry_tick * H,
        commit_time_s=receipt.exit_tick * H,
        foot_forces_n=by_foot,
        total_force_n=sum(by_foot.values(), start=np.zeros(3)),
        selected_ground_force_n=reaction,
        published_floor_force_n=published,
        loaded_feet=tuple(
            side
            for side, force in by_foot.items()
            if force[2] > original.CRITERIA["positive_foot_load_threshold_n"]
        ),
    )


def assess_step(task, stat, receipt, feet, floor, initial_feet):
    endpoint = task.last_transition[WORLD]
    row = endpoint["reading"].robots[0, original.ROBOT]
    support = foot_support(receipt, feet, floor)
    geometry = foot_geometry(receipt.diagnostics["accepted_frames"], feet)
    reference = task.initial.q[:3] + task.alignment @ (
        task.motion["body_pos_w"][endpoint["frame"], 0] - task.motion["body_pos_w"][0, 0]
    )
    force = support["total_force_n"]
    reaction = support["selected_ground_force_n"]
    stat["accepted_steps"] += 1
    stat["accepted_duration_s"] = receipt.exit_tick * H
    stat["minimum_root_height_m"] = min(stat["minimum_root_height_m"], float(row.q[2]))
    for name, value in (
        ("maximum_root_tilt_rad", endpoint["tilt"]),
        ("maximum_root_xy_drift_m", np.linalg.norm(row.q[:2] - task.initial.q[:2])),
        (
            "maximum_foot_center_xy_drift_m",
            max(
                np.linalg.norm(
                    geometry[side]["capsule_endpoint_center_m"][:2]
                    - initial_feet[side]["capsule_endpoint_center_m"][:2]
                )
                for side in feet
            ),
        ),
        ("maximum_reference_xy_error_m", np.linalg.norm(row.q[:2] - reference[:2])),
        ("maximum_joint_rmse_rad", endpoint["error"]),
        ("maximum_force_bodyweight_ratio", np.linalg.norm(force) / task.weight),
        (
            "maximum_nonfoot_ground_force_n",
            np.linalg.norm(support["published_floor_force_n"] - reaction),
        ),
    ):
        stat[name] = max(stat[name], float(value))
    stat["effort_caps_observed"] &= bool(
        np.all(
            np.abs(receipt.actual_drive)
            <= task.cap + original.CRITERIA["effort_cap_absolute_tolerance_nm"]
        )
    )
    stat["reaction_balance_observed"] &= bool(
        np.linalg.norm(force + reaction)
        <= original.CRITERIA["reaction_balance_relative_tolerance"] * task.weight
    )
    if support["sample_time_s"] >= original.CRITERIA["settling_time_s"]:
        stat["settled_samples"] += 1
        stat["settled_vertical_load_sum"] += float(force[2] / task.weight)
        stat["both_loaded_samples"] += int(len(support["loaded_feet"]) == 2)
    return support, geometry


def retain_failure(journal, batch, plan, runner, task, error, transition_emitted):
    publication = dict(
        accepted_worlds=getattr(
            error, "accepted_worlds", {WORLD: True} if transition_emitted else {}
        ),
        acceptance_known=getattr(error, "acceptance_known", transition_emitted),
        physics_accepted=getattr(error, "physics_accepted", True if transition_emitted else None),
        environment_transition_emitted=transition_emitted,
        plan_status=plan.last_status,
        last_accepted_contact_receipts=plan.last_accepted,
        task_endpoints_before_possible_reset={
            key: dict(
                tick=item["reading"].step_number,
                q=item["reading"].robots[0, original.ROBOT].q,
                v=item["reading"].robots[0, original.ROBOT].v,
                frame=item["frame"],
                tilt=item["tilt"],
                error=item["error"],
            )
            for key, item in task.last_transition.items()
        },
    )
    for name, read in (
        ("states", batch.snapshot),
        ("geometry", batch.tree_kinematics),
        ("runner", runner.snapshot),
    ):
        try:
            publication[name] = read()
        except Exception as read_error:
            publication[name + "_read_error"] = dict(
                type=type(read_error).__name__, message=str(read_error)
            )
    journal.context["failure_publication"] = publication
    journal.append("failure_publication_snapshot", **publication)


def run(args, journal):
    import onnxruntime as ort

    task = ResidentG1Task(args.assets.resolve(), journal)
    steps = round(args.duration / H)
    if steps < 1 or not math.isclose(steps * H, args.duration, rel_tol=0, abs_tol=1e-12):
        raise ValueError("duration must contain whole unchanged .0005 s physical steps")
    if (steps + task.decimation - 1) // task.decimation > len(task.motion["joint_pos"]):
        raise ValueError("requested duration exceeds the unchanged motion")
    session = ort.InferenceSession(
        str(task.assets / "policy.onnx"), providers=["CPUExecutionProvider"]
    )
    if session.get_inputs()[0].name != "obs" or session.get_outputs()[0].name != "actions":
        raise ValueError("official ONNX obs/actions names have changed")
    collision, bindings, materials, feet, floor = contact_registration(task)
    initial_feet = original.foot_geometry(task.asset, task.initial.q, task.colliders)
    stat = statistics()
    journal.context["statistics"] = stat
    sources = {
        name: original.source_record(task.assets / name)
        for name in (
            "g1_primitive_actuated.xml",
            "deploy.yaml",
            "dance1_subject2.npz",
            "policy.onnx",
        )
    }
    if (task.assets / "policy.onnx.data").exists():
        sources["policy.onnx.data"] = original.source_record(task.assets / "policy.onnx.data")
    journal.append(
        "initialization",
        schema="g1-resident-policy-v1",
        sources=sources,
        source_adapters={
            "original": original.source_record(Path(original.__file__)),
            "policy_protocol": original.source_record(
                Path(__file__).with_name("g1_connected_policy.py")
            ),
            "resident_example": original.source_record(Path(__file__)),
        },
        initial_q=task.initial.q,
        initial_v=task.initial.v,
        placement=task.placement,
        physical_model=task.asset.model,
        physical_joint_names=task.asset.joint_names,
        policy_joint_names=task.names,
        physical_from_policy=task.p_from_o,
        policy_from_physical=task.o_from_p,
        default_joint_pos_policy_order=task.default,
        kp_policy_order=task.kp,
        kd_policy_order=task.kd,
        scale_policy_order=task.scale,
        offset_policy_order=task.offset,
        effort_caps_physical_order_nm=task.cap,
        policy_alignment=task.alignment,
        dt_s=H,
        control_decimation=task.decimation,
        bodyweight_n=task.weight,
        criteria=original.CRITERIA,
        collision_model=collision,
        bindings=bindings,
        materials=materials,
        feet=feet,
        floor=floor,
        geometry_capacity=dict(shapes=256, candidates=4096, sites=128),
        backend=f"TYX_{args.tyx_mode}_dynamics_HOST_narrow_phase_NumPy_CPU_ONNX",
        connection="TyxWorldBatch -> ResidentContactStepper -> ResidentVectorEnv -> PolicyRunner",
        self_contact="explicitly_excluded_same_as_original_HOST_example",
        requested_physical_steps=steps,
    )

    def policy(observations, due):
        if observations.shape != (1, 154):
            raise ValueError("the unchanged one-world 154-value policy observation is required")
        journal.append("policy_request", observation=observations, due_mask=due)
        actions = np.zeros((1, 29))
        if due[0]:
            raw = session.run(["actions"], {"obs": observations.astype(np.float32)})[0][0]
            actions[0] = original.vector(raw, 29, "actual ONNX action")
        journal.append("policy_response", due_mask=due, actions=actions)
        return actions

    with ph.TyxWorldBatch([(WORLD, task.tree)], args.tyx_mode, cache_root=args.cache_root) as batch:
        with batch.prepare_contacts(
            collision, bindings, materials, candidate_capacity=4096, site_capacity=128
        ) as plan:
            env = ResidentVectorEnv(
                batch,
                task,
                (WORLD,),
                action_size=29,
                dt=H,
                horizon=steps + 1,
                stepper=ResidentContactStepper(batch, plan),
            )
            runner = PolicyRunner(
                env,
                policy,
                control_decimation=task.decimation,
                initial_observation=task.observe(batch.snapshot(), (WORLD,)),
            )
            journal.append(
                "resident_registration",
                initial_published_state=batch.snapshot(),
                initial_device_geometry=batch.tree_kinematics(),
                initial_energy=batch.summary(),
                plan_stats=plan.stats(),
                batch_stats=batch.stats(),
            )
            for tick in range(steps):
                entry = batch.snapshot()[WORLD]
                journal.context["last_entry"] = entry
                journal.append("attempt", tick=tick, dt_s=H, entry=entry, runner=runner.snapshot())
                transition_emitted = False
                try:
                    result = runner.step()
                    transition_emitted = True
                    receipt = plan.last_accepted[WORLD]
                    if receipt.entry_tick != tick or receipt.exit_tick != tick + 1:
                        raise RuntimeError(
                            "accepted contact receipt does not bind this physical step"
                        )
                    journal.context["last_accepted"] = receipt
                    support, geometry = assess_step(task, stat, receipt, feet, floor, initial_feet)
                    journal.append(
                        "accepted",
                        tick=tick,
                        time_s=receipt.exit_tick * H,
                        contact_receipt=receipt,
                        accepted_state=receipt.accepted_state,
                        support=support,
                        accepted_foot_geometry=geometry,
                        policy_result=result,
                        plan_stats=plan.stats(),
                        batch_stats=batch.stats(),
                    )
                    if not stat["effort_caps_observed"] or not stat["reaction_balance_observed"]:
                        raise RuntimeError(
                            "accepted state failed the original effort or reaction gate"
                        )
                    if bool(result.transition.terminated[0]):
                        raise RuntimeError(
                            "actual resident G1 trajectory fell; no reset continuation"
                        )
                except Exception as error:
                    retain_failure(journal, batch, plan, runner, task, error, transition_emitted)
                    raise
    assessment = original.evaluate("policy", stat, True, None)
    failure = None
    if not assessment["success"]:
        failure = dict(
            type="EngineeringAssessment",
            message="original engineering gates failed",
            failed_criteria=[name for name, passed in assessment["gates"].items() if not passed],
        )
        journal.append("first_failure", **failure)
    return dict(
        schema="g1-resident-policy-summary-v1",
        duration_s=args.duration,
        backend=args.tyx_mode,
        statistics=stat,
        assessment=assessment,
        success=assessment["success"],
        first_failure=failure,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--tyx-mode", choices=("interpret", "cuda"), default="cuda")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be positive and finite")
    journal = original.Journal(args.output)
    try:
        journal.append("request", arguments=vars(args), status="UNVALIDATED_NEW_CONNECTION")
        try:
            report = run(args, journal)
        except Exception as error:
            failure = dict(
                type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
                accepted_worlds=getattr(error, "accepted_worlds", None),
                acceptance_known=getattr(error, "acceptance_known", None),
                physics_accepted=getattr(error, "physics_accepted", None),
            )
            stat = journal.context.get("statistics")
            report = dict(
                schema="g1-resident-policy-summary-v1",
                success=False,
                first_failure=failure,
                context=journal.context,
                assessment=(original.evaluate("policy", stat, False, failure) if stat else None),
            )
            journal.append("first_failure", **failure)
        journal.finish(report)
    finally:
        journal.close()
    print(original.json_value(report))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
