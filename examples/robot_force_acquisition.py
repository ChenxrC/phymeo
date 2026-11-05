"""Acquire through actual bilateral feedback, then test lift retention and release.

The controller changes only RobotCommand. The application starts aligned over a
finite table; it is not an approach planner. Every accepted physical frame and
any failed predeclared expectation is saved. Existing A1 inputs are untouched.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np

from phymeo.builders.mimic_urdf import load_mimic_urdf
from phymeo.builders.panda import PANDA_URDF, load_panda
from phymeo.constraints.robot_contact import JointLimits, RobotContactMaterial
from phymeo.control.gripper_acquisition import GripperAcquisition
from phymeo.control.site_ik import SiteIKTarget, solve_multi_site_ik
from phymeo.core.capabilities import SolverFailure
from phymeo.core.robot import RobotState
from phymeo.dynamics.rotation import matrix_quaternion
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_payload_scene import PadFace

CASES = ("panda-sphere", "mimic-box", "mimic-zero-friction", "mimic-overload")
LIFT_M = 0.020
LIFT_DURATION_S = 0.600
HOLD_DURATION_S = 0.100
RELEASE_TIMEOUT_S = 1.200


def make_case(case, urdf=None):
    if case not in CASES:
        raise ValueError("unknown frozen acquisition case")
    b = ContactSceneBuilder()
    if case == "panda-sphere":
        asset = load_panda()
        q = asset.configuration(np.pi * np.array([0, -0.25, 0, -0.75, 0, 0.5, 0.25]), 0.060)
        center = np.mean(
            [asset.authored_link_pose("panda_" + s + "pad", q)[0] for s in ("left", "right")],
            axis=0,
        )
        names = [("panda_" + s + "finger", "panda_" + s + "finger_pad") for s in ("left", "right")]
        pads = [
            PadFace.named(*name, 1, sign, tuple(product((-0.005, 0.005), repeat=2)))
            for name, sign in zip(names, (-1, 1), strict=True)
        ]
        cap = list(asset.effort_limit)
        cap[-2:] = [4, 4]
        b.add_urdf(
            10,
            PANDA_URDF,
            loader=load_panda,
            robot_state=RobotState(q, np.zeros(9)),
            kp=[500] * 7 + [4500] * 2,
            kd=[30] * 7 + [18] * 2,
            effort_limit=cap,
            pads=pads,
        )
        b.add_sphere(20, 0.020, mass=0.04096, position=center)
        b.add_box(30, (0.015, 0.012, 0.014), mass=0.06, position=center + [0.10, 0, -0.006])
        b.add_box(40, (0.14, 0.08, 0.02), mass=None, position=center - [0, 0, 0.04])
        ids, opened, closed, speed = (7, 8), (0.03, 0.03), (0.01, 0.01), (0.025, 0.025)
        dt, masses, top, lift = (
            1 / 1600,
            (0.04096, 0.06),
            center[2] - 0.02,
            ("site", "panda_hand_tcp"),
        )
        payload = dict(kind="sphere", radius_m=0.02, mass_kg=0.04096)
    else:
        source = (
            Path(urdf)
            if urdf is not None
            else (Path(__file__).parents[1] / "tests/reference_newton/robot/mimic_jaws.urdf")
        )
        asset = load_mimic_urdf(source)
        q, center = np.zeros(2), np.array([0.25, -0.125, 0.25])
        names = [(s, s + "_rubber") for s in ("left", "right")]
        pads = [
            PadFace.named(*name, 0, sign, tuple(product((-0.005, 0.005), repeat=2)))
            for name, sign in zip(names, (1, -1), strict=True)
        ]
        cap = [40, 8]
        b.add_urdf(
            10,
            source,
            loader=load_mimic_urdf,
            robot_state=RobotState(q, np.zeros(2)),
            kp=[2700, 6000],
            kd=[95, 35],
            effort_limit=cap,
            pads=pads,
        )
        mass = 1.0 if case == "mimic-overload" else 0.1
        b.add_box(20, (0.02, 0.02, 0.02), mass=mass, position=center)
        b.add_sphere(30, 0.015, mass=0.03, position=(0.33, -0.125, 0.245))
        b.add_box(40, (0.12, 0.12, 0.01), mass=None, position=(0.25, -0.125, 0.22))
        ids, opened, closed, speed = (1,), (0,), (0.018,), (0.02,)
        dt, masses, top, lift = 0.001, (mass, 0.03), 0.23, ("coordinate", 0)
        payload = dict(kind="box", full_size_m=[0.04] * 3, mass_kg=mass)
    mu = 0.0 if case == "mimic-zero-friction" else 0.65
    for a, c in combinations((10, 20, 30, 40), 2):
        law = (
            RobotContactMaterial(1200, 1.2, mu, 0.04)
            if (a, c) == (10, 20)
            else (
                RobotContactMaterial(2200, 3, 0.45, 0.04)
                if c == 40 and a != 10
                else RobotContactMaterial(6000, 16, 0.35, 0.04)
            )
        )
        b.pair(a, c, law)
    scene = b.build()
    keys = tuple(scene.shape_key(10, *name) for name in names)
    controller = GripperAcquisition(
        scene.assets[0, 10],
        robot=(0, 10),
        payload_key=scene.body_keys[0, 20][0],
        pad_groups=tuple((key,) for key in keys),
        joint_indices=ids,
        open_q=opened,
        closed_q=closed,
        speed_limit=speed,
        effort_limit=cap,
        force_target_N=2,
        force_band_N=0.25,
        contact_min_N=1,
        dwell_s=0.020,
        loss_dwell_s=0.010,
        stop_speed=(0.002,) * len(ids),
        timeout_s=1.5,
    )
    return (
        scene,
        controller,
        dict(
            case=case,
            dt=dt,
            q=q,
            center=center,
            masses=masses,
            top=top,
            lift=lift,
            pad_keys=keys,
            primary_key=scene.body_keys[0, 20][0],
            secondary_key=scene.body_keys[0, 30][0],
            table_key=scene.body_keys[0, 40][0],
            rows=[scene.rigid_rows[0, key] for key in (20, 30)],
            cap=cap,
            payload_geometry=payload,
            expected_holding=case in {"panda-sphere", "mimic-box"},
            friction=mu,
            controller_inputs=dict(
                joint_indices=ids,
                open_q=opened,
                closed_q=closed,
                speed_limit=speed,
                effort_limit=cap,
                force_target_N=2,
                force_band_N=0.25,
                contact_min_N=1,
                dwell_s=0.020,
                loss_dwell_s=0.010,
                stop_speed=(0.002,) * len(ids),
                timeout_s=1.5,
            ),
        ),
    )


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _frame_data(scene, label):
    reading, geometry = scene.observe_scene()
    return dict(
        label=label,
        reading=_plain(reading),
        geometry=dict(
            time_s=geometry.time_s,
            units="m",
            shapes=[
                dict(
                    key=_plain(shape.key),
                    kind=shape.kind,
                    radius_m=shape.radius,
                    vertex_ids=shape.vertex_ids,
                    triangles=shape.triangles,
                    name=geometry.names[shape.key],
                    world_points_m=geometry.poses[shape.key].start.tolist(),
                )
                for shape in geometry.model.shapes
            ],
        ),
    )


def _relative_and_clearance(reading, geometry, info):
    first, second = [geometry.poses[key].start for key in info["pad_keys"]]
    origin = (first.mean(axis=0) + second.mean(axis=0)) / 2
    axes = first[[4, 2, 1]] - first[0]
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    if not np.allclose(axes @ axes.T, np.eye(3), atol=1e-10, rtol=0):
        raise ValueError("relative gripper readout requires an ordinary cuboid reference pad")
    relative = axes @ (reading.rigid.position[info["rows"][0]] - origin)
    primary = next(s for s in geometry.model.shapes if s.key == info["primary_key"])
    vertices = geometry.poses[primary.key].start
    low = float(vertices[:, 2].min() - (primary.radius if primary.kind == "sphere" else 0))
    return relative, low - info["top"], origin


def _table_force(reading, body_key, table_key):
    force = np.zeros(3)
    for row in reading.contacts:
        if {row["first"], row["second"]} == {body_key, table_key}:
            force += (
                row["force_on_first_N"] if row["first"] == body_key else row["force_on_second_N"]
            )
    return force


def _lift_goal(scene, reading, info):
    q = np.array(reading.robots[0, 10].q, copy=True)
    mode, target = info["lift"]
    if mode == "coordinate":
        q[target] += LIFT_M
        return q, dict(status="analytic_prismatic_target", shift_m=LIFT_M)
    asset = scene.assets[0, 10]
    position, rotation = asset.authored_link_pose(target, q)
    result = solve_multi_site_ik(
        asset.model,
        q,
        [
            SiteIKTarget(
                asset.sites[target],
                position_world_m=position + [0, 0, LIFT_M],
                orientation_world_xyzw=matrix_quaternion(rotation),
            )
        ],
        active_joints=tuple(range(7)),
        limits=JointLimits(asset.lower, asset.upper, 1),
    )
    return result.configuration if result.converged else None, _plain(result)


def run_case(scene, controller, info):
    dt = info["dt"]
    stage = "acquiring"
    trace, events = [], [_frame_data(scene, "initial-open-on-table")]
    first_failure = execution_failure = ik = None
    contact_ready = held = released = False
    support_after_release = second_support = release_recorded = False
    acquired_relative = hold_relative = lift_start_q = lift_goal = None
    lift_start = hold_start = release_start = None
    hold_good_time = 0.0
    posture = np.array(info["q"], copy=True)
    max_impulse = max_cap_excess = 0.0
    primary_weight = info["masses"][0] * 9.81
    initial, initial_geometry = scene.observe_scene()
    initial_relative, initial_clearance, _ = _relative_and_clearance(
        initial, initial_geometry, info
    )
    budget = (
        int(np.ceil((1.5 + LIFT_DURATION_S + HOLD_DURATION_S + RELEASE_TIMEOUT_S + 0.1) / dt)) + 3
    )
    for _ in range(budget):
        before, geometry = scene.observe_scene()
        t = before.time_s
        relative, clearance, _ = _relative_and_clearance(before, geometry, info)
        table_force = _table_force(before, info["primary_key"], info["table_key"])
        if stage == "lifting":
            u = np.clip((t - lift_start) / LIFT_DURATION_S, 0, 1)
            blend = u**3 * (10 + u * (-15 + 6 * u))
            posture = lift_start_q + blend * (lift_goal - lift_start_q)
        intent = "release" if stage == "releasing" else "acquire"
        try:
            proposal = controller.propose(before, dt_s=dt, posture_target=posture, intent=intent)
            if stage == "acquiring" and proposal.contact_ready:
                contact_ready = True
                acquired_relative = relative.copy()
                lift_start_q = np.array(before.robots[0, 10].q, copy=True)
                lift_goal, ik = _lift_goal(scene, before, info)
                events.append(_frame_data(scene, "contact-ready"))
                if lift_goal is None:
                    first_failure = dict(reason="lift_ik_failed", time_s=t, result=ik)
                    stage, release_start = "releasing", t
                else:
                    stage, lift_start = "lifting", t
                    posture = lift_start_q.copy()
            if stage in {"acquiring", "lifting", "holding"} and proposal.phase in {
                "lost",
                "empty",
                "timeout",
            }:
                if first_failure is None:
                    first_failure = dict(reason=proposal.reason, time_s=t, phase=proposal.phase)
                stage, release_start = "releasing", t
            slip = (
                0.0
                if acquired_relative is None
                else float(np.linalg.norm(relative - acquired_relative))
            )
            if stage in {"lifting", "holding"} and slip > 0.003:
                if first_failure is None:
                    first_failure = dict(
                        reason="relative_slip",
                        time_s=t,
                        slip_m=slip,
                        clearance_m=clearance,
                        table_force_N=table_force.tolist(),
                    )
                stage, release_start = "releasing", t
            if stage == "lifting" and t - lift_start >= LIFT_DURATION_S - 1e-12:
                stage, hold_start, hold_relative = "holding", t, relative.copy()
                posture = np.array(lift_goal, copy=True)
                events.append(_frame_data(scene, "lift-window-end"))
            if stage == "holding":
                hold_slip = float(np.linalg.norm(relative - hold_relative))
                good = (
                    clearance >= 0.010
                    and max(0.0, table_force[2]) < 0.1 * primary_weight
                    and (min(proposal.bilateral_N) >= 1 and hold_slip <= 0.002)
                )
                # Force is sampled at load_time_s, pose at time_s. Credit only
                # accepted intervals wholly inside the declared holding window.
                measured_h = (
                    0.0
                    if before.load_time_s is None or before.load_time_s < hold_start
                    else (t - before.load_time_s)
                )
                hold_good_time = hold_good_time + measured_h if good else 0.0
                if t - hold_start >= HOLD_DURATION_S - 1e-12:
                    held = hold_good_time >= HOLD_DURATION_S - 1e-12
                    if not held and first_failure is None:
                        first_failure = dict(
                            reason="physical_holding_gate",
                            time_s=t,
                            clearance_m=clearance,
                            table_force_N=table_force.tolist(),
                            hold_slip_m=hold_slip,
                            good_time_s=hold_good_time,
                        )
                    events.append(_frame_data(scene, "holding-window-end"))
                    stage, release_start = "releasing", t
            if stage == "releasing":
                proposal = controller.propose(
                    before, dt_s=dt, posture_target=posture, intent="release"
                )
                if not release_recorded:
                    events.append(_frame_data(scene, "release-request"))
                    release_recorded = True
                if before.load_time_s is not None and before.load_time_s >= release_start:
                    support_after_release |= table_force[2] > 0
                if proposal.phase == "released":
                    released = True
                    if support_after_release and second_support:
                        break
                if t - release_start >= RELEASE_TIMEOUT_S:
                    if first_failure is None:
                        first_failure = dict(reason="release_timeout", time_s=t)
                    break
            after = scene.step(dt, actions={10: proposal.command})
        except (ValueError, RuntimeError, SolverFailure) as exc:
            execution_failure = dict(
                reason=str(exc),
                type=type(exc).__name__,
                phase=stage,
                accepted_step=before.step_number,
                accepted_time_s=t,
            )
            break
        after_reading, after_geometry = scene.observe_scene()
        rel_after, gap_after, center_after = _relative_and_clearance(
            after_reading, after_geometry, info
        )
        for row, mass in zip(info["rows"], info["masses"], strict=True):
            delta = mass * (after.rigid.velocity[row] - before.rigid.velocity[row])
            applied = after.loads.inputs.rigid.force[row] + mass * np.array([0, 0, -9.81])
            max_impulse = max(max_impulse, float(np.max(abs(delta - dt * applied))))
        max_cap_excess = max(
            max_cap_excess,
            float(np.max(abs(after.robots[0, 10].drive_effort) - np.asarray(info["cap"]))),
        )
        second_force = _table_force(after, info["secondary_key"], info["table_key"])
        second_support |= second_force[2] > 0
        trace.append(
            dict(
                stage=stage,
                proposal=_plain(proposal),
                reading=_plain(after),
                relative_position_m=rel_after.tolist(),
                gripper_center_m=center_after.tolist(),
                clearance_m=gap_after,
                table_force_N=_table_force(after, info["primary_key"], info["table_key"]).tolist(),
                slip_since_contact_ready_m=None
                if acquired_relative is None
                else float(np.linalg.norm(rel_after - acquired_relative)),
            )
        )
    else:
        execution_failure = dict(
            reason="whole_protocol_step_budget", accepted_time_s=scene.observe().time_s
        )
    events.append(_frame_data(scene, "last-accepted"))
    last = scene.observe()
    checks = dict(
        actual_bilateral_acquisition=contact_ready,
        expected_holding_outcome=bool(held) == info["expected_holding"],
        actual_release=released,
        primary_supported_after_release=bool(support_after_release),
        second_body_has_real_support=bool(second_support),
        actuator_caps=max_cap_excess <= 1e-9,
        free_body_impulses=max_impulse < 3e-12,
        no_execution_failure=execution_failure is None,
    )
    return dict(
        schema="gripper-acquisition-a2-v1",
        case=info["case"],
        units="SI",
        protocol=dict(
            controller=_plain(info["controller_inputs"]),
            dt_s=dt,
            lift_m=LIFT_M,
            lift_duration_s=LIFT_DURATION_S,
            hold_duration_s=HOLD_DURATION_S,
            release_timeout_s=RELEASE_TIMEOUT_S,
            initial_q=info["q"].tolist(),
            initial_center_m=info["center"].tolist(),
            initial_relative_m=initial_relative.tolist(),
            initial_clearance_m=initial_clearance,
            payload_geometry=info["payload_geometry"],
            finger_friction=info["friction"],
        ),
        checks=checks,
        expected_physical_holding=info["expected_holding"],
        physical_holding_achieved=held,
        accepted_steps=last.step_number,
        time_s=last.time_s,
        contact_ready_observed=contact_ready,
        actual_release_observed=released,
        first_physical_failure=first_failure,
        execution_failure=execution_failure,
        max_free_body_impulse_error_Ns=max_impulse,
        max_motor_cap_excess=max_cap_excess,
        lift_target_result=ik,
        final_position_m=last.rigid.position[info["rows"]].tolist(),
        final_velocity_m_s=last.rigid.velocity[info["rows"]].tolist(),
        trace=trace,
        events=events,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("output exists; preserve the original frozen observation")
    scene, controller, info = make_case(args.case, args.urdf)
    with scene:
        result = run_case(scene, controller, info)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"trace", "events"}}, indent=2))
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
