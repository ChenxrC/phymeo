"""Registered contact-pose routes over the unchanged public CUDA diagnostic.

This is an explicit robot controller experiment, not a learned policy. Pose and
joint-path queries use CPU kinematics/collision; only the existing Env owns
physical stepping. No object state is authored or attached to a robot.
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

from phymeo.control.manipulation_diagnostic import NativeJointPD
from phymeo.control.manipulation_path import audit_native_joint_path
from phymeo.control.manipulation_pose_ik import solve_native_pose_ik

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_manipulation_control import run, write  # noqa: E402


class PreflightRejected(ValueError):
    pass


def pose(model, q):
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_kinematics(model, data)
    body = model.body("tool").id
    rotation = data.xmat[body].reshape(3, 3).copy()
    return data.xpos[body] + rotation @ np.array((0.0, 0.0, -0.14)), rotation


def builder(world, qualification_path):
    def build(scene, model):
        names = scene.robot_joint_names
        arm = names[: len(scene.robot_kinds)]
        addresses = [int(model.jnt_qposadr[model.joint(n).id]) for n in names]
        q0 = model.qpos0.copy()
        contact_q, raised_q = q0.copy(), q0.copy()
        if world == 1:
            # Previous registered rear candidate, now with its complete orientation.
            contact_q[addresses[1:3]] = (0.13196221564344993, 0.02025608165829705)
        elif world == 3:
            # Declared inward-and-up reference; only robot target coordinates.
            raised_q[addresses[1:3]] = (-0.8, 0.8)
        else:
            raise ValueError("this protocol only registers worlds 1 and 3")
        contact_target = pose(model, contact_q)
        lift_target = pose(model, raised_q)
        approach = solve_native_pose_ik(model, *contact_target, q0, arm)
        lift = solve_native_pose_ik(model, *lift_target, approach.configuration, arm)
        data = dict(
            world=world,
            schema="registered-contact-route-v1",
            contact_target=contact_target,
            lift_target=lift_target,
            approach=approach,
            lift=lift,
            qualification="sampled references only; not CCD",
        )
        if world == 3:
            negatives = []
            for pair in ((0.0, 0.0), (0.35, -0.35), (-0.35, 0.35)):
                seed = q0.copy()
                seed[addresses[3:5]] = pair
                negatives.append(
                    solve_native_pose_ik(model, (0.4, 0.0, 0.052), np.eye(3), seed, arm)
                )
            data["fixed_low_orientation_queries"] = negatives
        times = (0.0, 0.2, 0.7, 1.1, 1.7, 2.4)
        labels = ("initial", "settle", "approach", "close", "lift", "hold")
        knots = [
            q0.copy(),
            q0.copy(),
            approach.configuration.copy(),
            approach.configuration.copy(),
            lift.configuration.copy(),
            lift.configuration.copy(),
        ]
        for q in knots[3:]:
            q[addresses[-2:]] = 0.044
        upper = [q.copy() for q in knots]
        for q in upper:
            q[addresses[-2:]] = np.minimum(0.06, q[addresses[-2:]] + 0.016)
        allowance = 0.026 if world == 1 else 0.023
        allowed = tuple(
            (pad, obj, allowance)
            for pad in scene.finger_geom_names
            for obj in scene.object_geom_names
        )
        normal_path = audit_native_joint_path(model, times, knots, names, allowed_contacts=allowed)
        upper_path = audit_native_joint_path(model, times, upper, names, allowed_contacts=allowed)
        data.update(
            nominal_path=normal_path,
            upper_reference_path=upper_path,
            allowed_contacts=allowed,
            upper_qualification="fixed +16mm reference at every knot; two paths do not certify all intervening offsets",
            physical_object_qualification="FREE object remains at original initial pose in queries, not a predicted attached object",
        )
        valid = (
            approach.converged
            and lift.converged
            and normal_path.sampled_clear
            and upper_path.sampled_clear
        )
        data["valid"] = valid
        write(qualification_path, data)
        print(
            json.dumps(
                dict(
                    world=world,
                    preflight=valid,
                    approach_error_m=approach.position_error_m,
                    approach_error_rad=approach.orientation_error_rad,
                    lift_error_m=lift.position_error_m,
                    lift_error_rad=lift.orientation_error_rad,
                    nominal_clear=normal_path.sampled_clear,
                    upper_clear=upper_path.sampled_clear,
                    nominal_samples=normal_path.sample_count,
                    upper_samples=upper_path.sample_count,
                )
            ),
            flush=True,
        )
        if not valid:
            raise PreflightRejected("registered pose or reference path did not qualify")
        kp = [250.0 if kind == "P" else 25.0 for kind in scene.robot_kinds] + [150.0, 150.0]
        kd = [12.0 if kind == "P" else 3.0 for kind in scene.robot_kinds] + [2.0, 2.0]
        nodes = [
            (t, label, q[addresses].copy())
            for t, label, q in zip(times, labels, knots, strict=True)
        ]
        audit = dict(
            valid=True,
            goal=scene.goals[scene.default_goal],
            kp=kp,
            kd=kd,
            local_point=(0.0, 0.0, -0.14),
            waypoints=[approach, lift],
            robot_joint_names=names,
            q_addresses=addresses,
            reference_knots=nodes,
            route_qualification_file=str(qualification_path),
            nominal_path_binding=normal_path.input_binding,
            upper_reference_path_binding=upper_path.input_binding,
            qualification="original task oracle; finite reference queries, no grasp or place promise",
        )
        return NativeJointPD(model, names, kp, kd), nodes, audit

    return build


def main(output):
    if not output.is_absolute() or output.exists():
        raise ValueError("fresh absolute output root required")
    output.mkdir(parents=True)
    attempts = []
    for world in (1, 3):
        try:
            run(
                output / f"world-{world}",
                2400,
                (world,),
                slip_guard=True,
                plan_builder=builder(world, output / f"world-{world}-qualification.json"),
            )
        except PreflightRejected as error:
            attempts.append(
                dict(world=world, status="PREFLIGHT_REJECTED", physical_steps=0, reason=str(error))
            )
        except BaseException as error:
            attempts.append(dict(world=world, status="EXECUTION_ERROR", reason=str(error)))
            write(output / "attempts.json", attempts)
            raise
        else:
            manifest = json.loads(
                (output / f"world-{world}" / "manifest.json").read_text(encoding="utf8")
            )
            attempts.append(
                dict(
                    world=world,
                    status=manifest["status"],
                    physical_steps=manifest["steps"],
                    task_success=manifest["worlds"][0]["task_success"],
                    stopped_reason=manifest["stopped_reason"],
                )
            )
        write(output / "attempts.json", attempts)
    print(json.dumps(attempts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
