"""Bounded diagnostic control through the public heterogeneous CUDA Env.

CPU MuJoCo is used for model compilation, analytic IK and inverse-dynamics bias
only. Every physical state is advanced by TYX CUDA. This is not policy training.
"""

import argparse
import gzip
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import mujoco as mj
import numpy as np
import torch

from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.control.manipulation_contact import measure_finger_slip
from phymeo.control.manipulation_diagnostic import NativeJointPD, solve_native_point_ik
from phymeo.control.manipulation_grip import BoundedSlipClosure
from phymeo.training.manipulation_env import HeterogeneousManipulationEnv
from phymeo.training.manipulation_tasks import H, manipulation_scenes
from phymeo.training.mujoco_tyx_env import MuJoCoTyxActionSpec


def plain(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if is_dataclass(x):
        return {field.name: plain(getattr(x, field.name)) for field in fields(x)}
    if isinstance(x, Mapping):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    return x


def write(path, data):
    path.write_text(
        json.dumps(plain(data), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )


def emit(stream, data):
    stream.write(
        json.dumps(plain(data), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    )


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point(model, q, local):
    data = mj.MjData(model)
    data.qpos[:] = q
    mj.mj_kinematics(model, data)
    body = model.body("tool").id
    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ local


def build_plan(scene, model):
    """Fixed author schedule; IK failures hold the original robot, never teleport."""
    names = scene.robot_joint_names
    arm = names[: len(scene.robot_kinds)]
    addresses = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    q0 = model.qpos0.copy()
    goal = scene.goals[scene.default_goal]
    pushing = goal.kind == "pusht"
    local = np.array([0.0, 0.0, -0.16 if pushing else -0.14])
    start = point(model, q0, local)
    obj = np.array(scene.initial_object_position)
    obj[2] -= 0.001  # Authored support height, not a physical state write.
    target = next(t for t in scene.targets if t.key == goal.target_key)
    targets = []
    if pushing:
        direction = np.array(target.position[:2]) - obj[:2]
        direction /= np.linalg.norm(direction)
        # A conservative boundary approach; full object pose success remains the oracle.
        extent = max(
            abs(np.dot(np.array(s.position[:2]), direction))
            + np.dot(np.array(s.size[:2]), np.abs(direction))
            for s in scene.object_shapes
        )
        near = start.copy()
        near[:2] = obj[:2] - direction * (extent + 0.018 + 0.01)
        end = near.copy()
        end[:2] += np.array(target.position[:2]) - obj[:2] + direction * 0.01
        targets = [
            (0.2, "settle", start, 0.0),
            (0.8, "approach", near, 0.0),
            (1.8, "push", end, 0.0),
            (2.2, "hold", end, 0.0),
        ]
    else:
        grasp_point = obj.copy()
        grasp_point[2] = max(grasp_point[2], 0.045)  # 40 mm pad half-height + 5 mm floor margin.
        lifted = grasp_point + np.array([0.0, 0.0, 0.07])
        targets = [
            (0.2, "settle", start, 0.0),
            (0.7, "descend", grasp_point, 0.0),
            (1.1, "close", grasp_point, 0.044),
            (1.7, "lift", lifted, 0.044),
        ]
        if goal.kind == "place":
            moved = lifted.copy()
            moved[:2] = target.position[:2]
            lower = moved.copy()
            lower[2] = grasp_point[2] + 0.003
            targets += [
                (2.5, "transport", moved, 0.044),
                (3.1, "lower", lower, 0.044),
                (3.5, "release", lower, 0.0),
                (4.0, "settle-place", lower, 0.0),
            ]
        else:
            targets += [(2.2, "hold", lifted, 0.044)]
    nodes = [(0.0, "initial", q0[addresses].copy())]
    audit = []
    seed = q0.copy()
    valid = True
    for seconds, label, desired, closing in targets:
        result = solve_native_point_ik(model, desired, seed, arm, local_point=local)
        audit.append(dict(seconds=seconds, stage=label, target=desired, ik=result))
        if not result.converged:
            valid = False
        seed = result.configuration.copy()
        configuration = seed[addresses].copy()
        if not pushing:
            configuration[-2:] = closing
        nodes.append((seconds, label, configuration))
    if not valid:
        nodes = [
            (0.0, "ik-incomplete-hold", q0[addresses].copy()),
            (4.0, "ik-incomplete-hold", q0[addresses].copy()),
        ]
    kp = [250.0 if k == "P" else 25.0 for k in scene.robot_kinds] + (
        [150.0, 150.0] if not pushing else []
    )
    kd = [12.0 if k == "P" else 3.0 for k in scene.robot_kinds] + (
        [2.0, 2.0] if not pushing else []
    )
    return (
        NativeJointPD(model, names, kp, kd),
        nodes,
        dict(
            valid=valid,
            goal=goal,
            local_point=local,
            kp=kp,
            kd=kd,
            waypoints=audit,
            qualification="kinematic IK only, not collision-free path or task success",
        ),
    )


def reference(nodes, seconds):
    for i in range(1, len(nodes)):
        end, stage, q1 = nodes[i]
        begin, _, q0 = nodes[i - 1]
        if seconds <= end:
            u = np.clip((seconds - begin) / (end - begin), 0.0, 1.0)
            smooth = u * u * (3.0 - 2.0 * u)
            rate = 6.0 * u * (1.0 - u) / (end - begin)
            return q0 + smooth * (q1 - q0), rate * (q1 - q0), stage
    return nodes[-1][2], np.zeros_like(nodes[-1][2]), nodes[-1][1]


def run(output, steps, indices, preflight_only=False, slip_guard=False, plan_builder=build_plan):
    if output.exists() or not output.is_absolute():
        raise ValueError("fresh absolute output required")
    if (
        not 1 <= steps <= 4000
        or len(set(indices)) != len(indices)
        or any(i not in range(8) for i in indices)
    ):
        raise ValueError("bounded steps and unique authored scene indices required")
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    scenes = {i: manipulation_scenes()[i] for i in indices}
    compilations = {i: compile_mujoco_xml(s.mjcf, require_asset=False) for i, s in scenes.items()}
    native = {i: c.fresh_native_model() for i, c in compilations.items()}
    controls, plans, audits, worlds = {}, {}, {}, []
    for i, s in scenes.items():
        controls[i], plans[i], audits[i] = plan_builder(s, native[i])
        source = output / "sources" / (s.key + ".xml")
        source.write_text(s.mjcf, encoding="utf-8")
        goal = s.goals[s.default_goal]
        worlds.append(
            dict(
                world=i,
                key=s.key,
                robot_kind=s.robot_kinds,
                goal_kind=goal.kind,
                instruction_en=goal.instructions_en[0],
                source_file=source.relative_to(output).as_posix(),
                source_sha256=sha(source),
                nq=native[i].nq,
                nv=native[i].nv,
                na=native[i].na,
                task_success=False,
            )
        )
    grip_controls = {}
    if slip_guard:
        for i, scene in scenes.items():
            if len(scene.finger_geom_names) == 2:
                bounds = np.array(
                    [
                        native[i].jnt_range[native[i].joint(n).id]
                        for n in scene.robot_joint_names[-2:]
                    ]
                )
                grip_controls[i] = BoundedSlipClosure(bounds[:, 0], bounds[:, 1])
                audits[i]["slip_feedback"] = dict(
                    addresses=dict(
                        object_body=native[i].body("object").id,
                        pads=[native[i].geom(n).id for n in scene.finger_geom_names],
                    ),
                    max_correction_m=0.016,
                    max_rate_m_s=0.02,
                    slip_deadband_m_s=0.015,
                    gain=2.0,
                    min_normal_n=0.05,
                    max_dt_s=0.01,
                    measurement="previous accepted input frame, one step causal delay",
                    actuation="finger position reference bias; original reference velocity retained",
                    continuation="controller state is traced; final Env checkpoint alone is not a controller resume",
                )
    write(output / "control-plan.json", audits)
    if preflight_only:
        write(
            output / "preflight.json",
            dict(
                status="KINEMATIC_ONLY",
                worlds=worlds,
                converged={i: a["valid"] for i, a in audits.items()},
            ),
        )
        return
    manifest = dict(
        schema="phymeo-manipulation-control-v1",
        mode="cuda",
        status="FAILED_PREFIX",
        controller="diagnostic-native-pd",
        slip_feedback=bool(slip_guard),
        h=H,
        steps=0,
        requested_steps=steps,
        state_file="states.jsonl.gz",
        worlds=worlds,
        stopped_reason="not-started",
    )
    success = dict.fromkeys(indices, False)
    stages = dict.fromkeys(indices, "initial")
    stats = {
        i: dict(
            peak_finger_n=[0.0, 0.0],
            peak_clearance_m=0.0,
            min_clearance_m=0.0,
            peak_support_n=0.0,
            max_speed_m_s=0.0,
            lifted=False,
            transported=False,
        )
        for i in indices
    }
    wall = time.perf_counter()
    last_sample = -1
    ordinal = 0
    accepted_states = None

    def sample(batch, state_stream, step):
        nonlocal ordinal, last_sample
        states = accepted_states
        emit(
            state_stream,
            dict(
                ordinal=ordinal,
                step=step,
                worlds=[
                    dict(
                        world=i,
                        q=states[i].q,
                        v=states[i].v,
                        act=states[i].act,
                        time=states[i].time,
                        tick=states[i].tick,
                        stage=stages[i],
                        success=success[i],
                    )
                    for i in indices
                ],
            ),
        )
        ordinal += 1
        last_sample = step
        state_stream.flush()

    def forbid(*args, **kwargs):
        raise AssertionError("CPU physics stepping is forbidden")

    try:
        specs = tuple(MuJoCoTyxModelSpec(s.key, compilations[i]) for i, s in scenes.items())
        rows = tuple(MuJoCoTyxWorldSpec(i, s.key) for i, s in scenes.items())
        with (
            patch.object(mj, "mj_step", forbid),
            patch.object(mj, "mj_step1", forbid),
            patch.object(mj, "mj_step2", forbid),
        ):
            with MuJoCoTyxHeterogeneousBatch(specs, rows, mode="cuda") as batch:
                action_specs = {
                    i: MuJoCoTyxActionSpec(
                        tuple(range(native[i].nu)), scale=np.full(native[i].nu, 20.0, np.float32)
                    )
                    for i in indices
                }
                with HeterogeneousManipulationEnv(
                    batch,
                    scenes,
                    goals={i: s.goals[s.default_goal] for i, s in scenes.items()},
                    action_specs=action_specs,
                    horizons=dict.fromkeys(indices, steps + 1),
                    auto_reset=False,
                ) as env:
                    with (
                        gzip.open(
                            output / manifest["state_file"], "xt", encoding="utf-8"
                        ) as state_stream,
                        gzip.open(output / "trace.jsonl.gz", "xt", encoding="utf-8") as trace,
                    ):
                        accepted_states = batch.integration_snapshot()
                        sample(batch, state_stream, 0)
                        try:
                            for tick in range(steps):
                                obs = env.observation
                                q, v = obs["q"].cpu().numpy(), obs["v"].cpu().numpy()
                                raw = np.zeros((len(indices), env.num_actions), np.float32)
                                diagnostics, feedback = {}, {}
                                entry_arrays = (
                                    {
                                        name: obs[name].detach().cpu().numpy()
                                        for name in (
                                            "entry_meta",
                                            "entry_clock",
                                            "entry_contact_ids",
                                            "entry_contact_data",
                                            "entry_body_mask",
                                            "entry_body_position",
                                            "entry_body_linear_velocity",
                                            "entry_body_angular_velocity",
                                        )
                                    }
                                    if grip_controls
                                    else {}
                                )
                                for row, i in enumerate(indices):
                                    desired, velocity, stages[i] = reference(plans[i], tick * H)
                                    if i in grip_controls:
                                        measurement = measure_finger_slip(
                                            entry_arrays,
                                            row,
                                            object_body=native[i].body("object").id,
                                            pad_geom_ids=tuple(
                                                native[i].geom(n).id
                                                for n in scenes[i].finger_geom_names
                                            ),
                                        )
                                        if measurement is None:
                                            if tick != 0:
                                                raise ValueError(
                                                    "slip feedback lost accepted entry"
                                                )
                                            feedback[i] = dict(measurement=None, closure=None)
                                        else:
                                            if (
                                                measurement.input_tick,
                                                measurement.output_tick,
                                            ) != (tick - 1, tick):
                                                raise ValueError(
                                                    "slip feedback is not the preceding accepted input"
                                                )
                                            closure = grip_controls[i].update(
                                                measurement.time_s,
                                                desired[-2:],
                                                measurement.normal_loads,
                                                measurement.slip_speeds,
                                                stages[i]
                                                in ("close", "lift", "hold", "transport", "lower"),
                                            )
                                            desired = desired.copy()
                                            desired[-2:] = closure.targets
                                            feedback[i] = dict(
                                                measurement=measurement, closure=closure
                                            )
                                    ctrl, diagnostics[i] = controls[i].command(
                                        q[row, : native[i].nq],
                                        v[row, : native[i].nv],
                                        desired,
                                        velocity,
                                    )
                                    raw[row, : native[i].nu] = np.arctanh(
                                        np.asarray(ctrl, np.float32) / np.float32(20.0)
                                    )
                                transition = env.step(
                                    torch.tensor(raw, device=env.device),
                                    world_keys=env.world_keys,
                                    expected_frame=env.current_frame,
                                )
                                manifest["steps"] = tick + 1
                                accepted_states = batch.integration_snapshot()
                                for i, result in transition.results.items():
                                    success[i] = result.success
                                    metrics = dict(result.metrics)
                                    frame = transition.success_frames[i]
                                    stat = stats[i]
                                    stat["peak_finger_n"] = np.maximum(
                                        stat["peak_finger_n"], frame.finger_normal_forces_n
                                    ).tolist()
                                    stat["peak_clearance_m"] = max(
                                        stat["peak_clearance_m"], metrics["clearance_m"]
                                    )
                                    stat["min_clearance_m"] = min(
                                        stat["min_clearance_m"], metrics["clearance_m"]
                                    )
                                    stat["peak_support_n"] = max(
                                        stat["peak_support_n"], frame.support_up_force_n
                                    )
                                    stat["max_speed_m_s"] = max(
                                        stat["max_speed_m_s"], metrics["speed_m_s"]
                                    )
                                    stat["lifted"], stat["transported"] = (
                                        result.history.lifted,
                                        result.history.transported,
                                    )
                                emit(
                                    trace,
                                    dict(
                                        step=tick + 1,
                                        raw_action=transition.action,
                                        native_channels=transition.native_channels,
                                        action_receipt=transition.action_receipt,
                                        accepted=transition.accepted_worlds,
                                        success_frames=transition.success_frames,
                                        results=transition.results,
                                        controller=diagnostics,
                                        slip_feedback=feedback,
                                        feedback_source=entry_arrays,
                                        contact_ids=transition.observation["entry_contact_ids"],
                                        contact_data=transition.observation["entry_contact_data"],
                                        contact_world_wrench=transition.observation[
                                            "entry_contact_wrench_world"
                                        ],
                                        contact_body_wrench=transition.observation[
                                            "entry_contact_body_wrench"
                                        ],
                                        q=transition.observation["q"],
                                        v=transition.observation["v"],
                                        clock=transition.observation["clock"],
                                    ),
                                )
                                if (tick + 1) % 10 == 0:
                                    sample(batch, state_stream, tick + 1)
                                if (tick + 1) % 100 == 0:
                                    print(
                                        json.dumps(
                                            dict(
                                                step=tick + 1,
                                                seconds=(tick + 1) * H,
                                                success=success,
                                                stats=stats,
                                            )
                                        ),
                                        flush=True,
                                    )
                                if transition.final_mask.any().item():
                                    manifest["stopped_reason"] = "first_task_terminated"
                                    break
                            else:
                                manifest["stopped_reason"] = "registered_window_complete"
                            manifest["status"] = "COMPLETE"
                        finally:
                            if manifest["steps"] != last_sample:
                                sample(batch, state_stream, manifest["steps"])
                        (output / "final.checkpoint").write_bytes(env.checkpoint().to_bytes())
    except BaseException as error:
        manifest["failure"] = dict(
            type=type(error).__name__,
            message=str(error),
            phase=getattr(error, "phase", None),
            action_accepted=getattr(error, "action_accepted", None),
            state_accepted=getattr(error, "state_accepted", None),
        )
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - wall
        manifest["saved_frames"] = ordinal
        for row in worlds:
            row["task_success"] = success[row["world"]]
        for key, name in (("state_sha256", "states.jsonl.gz"), ("trace_sha256", "trace.jsonl.gz")):
            if (output / name).exists():
                manifest[key] = sha(output / name)
        write(output / "manifest.json", manifest)
        write(
            output / "summary.json",
            dict(
                success=success,
                metrics=stats,
                kinematic_valid={i: a["valid"] for i, a in audits.items()},
                qualification="diagnostic control only; no learned network or optimizer",
                **{
                    k: manifest[k]
                    for k in (
                        "status",
                        "steps",
                        "requested_steps",
                        "stopped_reason",
                        "wall_seconds",
                    )
                },
            ),
        )
    print(json.dumps(plain(manifest)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2200)
    parser.add_argument("--worlds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--slip-guard", action="store_true")
    args = parser.parse_args()
    run(args.output, args.steps, tuple(args.worlds), args.preflight_only, args.slip_guard)
