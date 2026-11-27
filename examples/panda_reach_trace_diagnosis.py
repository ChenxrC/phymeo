"""Diagnose a saved panda_state_ppo reach paired-physics NPZ with native MuJoCo FK."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

from _panda_ppo_reach import (
    BOARD_DISPLACEMENT_M, BOARD_ROTATION_RAD, DISTANCE_M, DWELL_S,
    ORIENTATION_RAD, PART_DRIFT_M, SITE_NAMES,
)


LIMITS = dict(distance_m=DISTANCE_M, orientation_rad=ORIENTATION_RAD,
              board_displacement_m=BOARD_DISPLACEMENT_M, board_tilt_rad=BOARD_ROTATION_RAD,
              part_drift_m=PART_DRIFT_M)
GROUPS = {"left_arm": slice(0, 7), "right_arm": slice(7, 14), "fingers": slice(14, 16)}
NEAR_SATURATION = .95


def norm(a):
    return np.linalg.norm(a, axis=-1)


def angle(relative):
    anti = relative - relative.swapaxes(-1, -2)
    sine = np.linalg.norm(anti, axis=(-2, -1)) / math.sqrt(8)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    return np.arctan2(sine, cosine)


def kinematics(model, positions):
    data = mj.MjData(model)
    sites = [model.site(name).id for name in SITE_NAMES]
    xyz = np.empty((len(positions), 3, 3))
    rotation = np.empty((len(positions), 3, 3, 3))
    for step, q in enumerate(positions):
        data.qpos[:] = q
        mj.mj_kinematics(model, data)
        xyz[step] = data.site_xpos[sites]
        rotation[step] = data.site_xmat[sites].reshape(3, 3, 3)
    board, hand, pin = xyz.transpose(1, 0, 2)
    board_r, hand_r, pin_r = rotation.transpose(1, 0, 2, 3)
    error = pin - hand
    error_in_hand = np.einsum("tji,tj->ti", hand_r, error)
    metrics = dict(distance_m=norm(error_in_hand), orientation_rad=angle(hand_r.transpose(0, 2, 1) @ pin_r),
                   board_displacement_m=norm(board - board[0]), board_tilt_rad=angle(board_r[0].T @ board_r),
                   part_drift_m=norm(pin - pin[0]))
    return hand, error, error_in_hand, metrics


def gate_counts(mask, dt):
    # A run of N qualified endpoint samples includes exactly N-1 complete intervals.
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    longest = int(np.argmax(ends - starts)) if len(starts) else None
    bounds = None if longest is None else [int(starts[longest]), int(ends[longest] - 1)]
    count = 0 if bounds is None else bounds[1] - bounds[0]
    return dict(passing_endpoints=int(mask.sum()), reset_pass=bool(mask[0]), final_pass=bool(mask[-1]),
                first_pass_step=int(starts[0]) if len(starts) else None, longest_run_endpoints=bounds,
                longest_intervals=count, longest_dwell_s=count * dt)


def motor_usage(action, ctrl_ratio, nearest, repeat):
    # Map the held policy request to each interval where it was applied. A policy
    # tick intersecting the closest sample then contributes to both adjacent phases.
    held = np.repeat(action, repeat, axis=0)
    results = {}
    for phase, interval in (("full", slice(None)), ("to_closest", slice(0, nearest)),
                            ("after_closest", slice(nearest, None))):
        usage = {}
        for group, indices in GROUPS.items():
            requested = np.abs(held[interval, indices])
            applied = ctrl_ratio[interval, indices]
            if requested.size:
                usage[group] = dict(
                    action_abs_mean=float(requested.mean()), action_abs_max=float(requested.max()),
                    action_near_fraction=float((requested >= NEAR_SATURATION).mean()),
                    effort_fraction_mean=float(applied.mean()), effort_fraction_max=float(applied.max()),
                    effort_near_fraction=float((applied >= NEAR_SATURATION).mean()),
                )
        results[phase] = usage
    return results


def world_report(model, q, action, ratios, repeat, saved_mask, final_values, dwell_steps, dwell_s):
    dt = float(model.opt.timestep)
    hand, error, local_error, metrics = kinematics(model, q)
    dist = metrics["distance_m"]
    best = int(dist.argmin())
    masks = {key: metrics[key] <= limit for key, limit in LIMITS.items()}
    masks["all"] = np.logical_and.reduce(list(masks.values()))
    np.testing.assert_array_equal(masks["all"][1:], saved_mask, err_msg="native vs recorded substep mask")
    for key in LIMITS:
        np.testing.assert_allclose(metrics[key][-1], final_values[key], rtol=2e-4, atol=2e-5, err_msg=key)
    gates = {key: gate_counts(mask, dt) for key, mask in masks.items()}
    np.testing.assert_equal(gates["all"]["longest_intervals"], dwell_steps, err_msg="recorded max dwell steps")
    np.testing.assert_allclose(gates["all"]["longest_dwell_s"], dwell_s, rtol=2e-6, atol=1e-9)
    snapshots = {}
    for label, step in (("reset", 0), ("closest", best), ("final", len(dist) - 1)):
        snapshots[label] = dict(
            physics_step=step, time_s=step * dt, right_xyz_world_m=hand[step].tolist(),
            right_delta_from_reset_world_m=(hand[step] - hand[0]).tolist(),
            pin_minus_right_world_m=error[step].tolist(), pin_minus_right_hand_frame_m=local_error[step].tolist(),
            **{key: float(values[step]) for key, values in metrics.items()},
        )
    changes = np.diff(dist)
    motion = dict(reset_to_closest_gain_m=float(dist[0] - dist[best]),
                  closest_to_final_loss_m=float(dist[-1] - dist[best]),
                  reset_to_final_gain_m=float(dist[0] - dist[-1]),
                  sum_decreasing_distance_m=float(np.maximum(-changes, 0).sum()),
                  sum_increasing_distance_m=float(np.maximum(changes, 0).sum()),
                  right_path_length_m=float(norm(np.diff(hand, axis=0)).sum()),
                  right_max_distance_from_reset_m=float(norm(hand - hand[0]).max()),
                  left_max_joint_excursion_rad=float(np.abs(q[:, :7] - q[0, :7]).max()),
                  right_max_joint_excursion_rad=float(np.abs(q[:, 7:14] - q[0, 7:14]).max()),
                  board_max_displacement_m=float(metrics["board_displacement_m"].max()),
                  board_max_tilt_rad=float(metrics["board_tilt_rad"].max()),
                  pin_max_drift_m=float(metrics["part_drift_m"].max()))
    return dict(snapshots=snapshots, motion=motion, gates=gates,
                success=gates["all"]["longest_intervals"] >= math.ceil(DWELL_S / dt),
                motor_usage=motor_usage(action, ratios, best, repeat))


def aggregate(rows):
    def mean(records):
        return {k: np.mean([r[k] for r in records], axis=0).tolist() for k in records[0]}

    snapshots = {key: mean([r["snapshots"][key] for r in rows]) for key in rows[0]["snapshots"]}
    gates = {}
    for key in rows[0]["gates"]:
        counts = [r["gates"][key] for r in rows]
        gates[key] = dict(worlds_ever_pass=sum(g["passing_endpoints"] > 0 for g in counts),
                          worlds_pass_final=sum(g["final_pass"] for g in counts),
                          passing_world_endpoints=sum(g["passing_endpoints"] for g in counts),
                          longest_dwell_s=max(g["longest_dwell_s"] for g in counts))
    return dict(success_worlds=sum(r["success"] for r in rows), gates=gates, mean_snapshots=snapshots,
                min_closest_distance_m=min(r["snapshots"]["closest"]["distance_m"] for r in rows),
                mean_motion=mean([r["motion"] for r in rows]),
                mean_full_motor_usage={key: mean([r["motor_usage"]["full"][key] for r in rows]) for key in GROUPS})


def diagnose(trace_path, scene_path):
    model = mj.MjModel.from_xml_path(str(scene_path))
    if (model.nq, model.nv, model.nu) != (23, 22, 16):
        raise ValueError("expected the original 23/22/16 dual-Panda free-pin scene")
    dt = float(model.opt.timestep)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid scene timestep")
    bounds = np.array(model.actuator_ctrlrange)
    bounds[14:, 0] = np.maximum(bounds[14:, 0], -20.)
    bounds[14:, 1] = np.minimum(bounds[14:, 1], 20.)
    if not np.all((bounds[:, 0] < 0) & (bounds[:, 1] > 0)):
        raise ValueError("expected two-sided physical motor control ranges")
    branches = {}
    trace_hash = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    scene_hash = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    report = dict(schema="panda-reach-trace-diagnosis-v1", trace=trace_path.name, trace_sha256=trace_hash,
                  scene_sha256=scene_hash, site_names=list(SITE_NAMES), dt_s=dt, thresholds=LIMITS,
                  dwell_s=DWELL_S, required_intervals=math.ceil(DWELL_S / dt), branches=branches,
                  counting="Each world includes reset and every accepted physical substep endpoint. "
                           "Closest is the earliest minimum distance. A dwell interval requires both endpoints.",
                  motor_usage_definition="Fractions count joint-interval samples at or above 95% of "
                                         "unit policy action or same-sign physical motor control limit. "
                                         "The finger control bounds include the runner's 20 N cap.")
    with np.load(trace_path, allow_pickle=False) as data:
        for name in ("before", "after", "motors_off"):
            def read(field, shape=None):
                value = data[f"{name}_{field}"]
                if shape is not None and value.shape != shape:
                    raise ValueError(f"{name}_{field}: expected {shape}, got {value.shape}")
                if not np.isfinite(value).all():
                    raise ValueError(f"{name}_{field}: nonfinite observations")
                return value

            sub = read("substep_qpos")
            if sub.ndim != 4 or sub.shape[-1] != 23 or min(sub.shape[:3]) < 1:
                raise ValueError(f"{name}: expected [policy step, substep, world, 23] accepted qpos")
            steps, repeat, worlds = sub.shape[:3]
            q = read("qpos", (steps + 1, worlds, 23))
            v = read("qvel", (steps + 1, worlds, 22))
            if name == "before":
                shape, reset_q, reset_v = sub.shape, q[0].copy(), v[0].copy()
                report.update(world_count=worlds, policy_steps=steps, action_repeat=repeat,
                              physical_intervals=steps * repeat, duration_s=steps * repeat * dt)
            elif sub.shape != shape or not np.array_equal(q[0], reset_q) or not np.array_equal(v[0], reset_v):
                raise ValueError(f"{name}: paired trajectory dimensions or reset q/v differ")
            np.testing.assert_array_equal(sub[:, -1], q[1:], err_msg=f"{name} policy/substep qpos")
            action = read("action", (steps, worlds, 16))
            controls = read("controls", (steps, repeat, worlds, 16)).reshape(-1, worlds, 16)
            ratios = np.where(controls >= 0, controls / bounds[:, 1], controls / bounds[:, 0])
            if np.abs(action).max() > 1.000001 or ratios.max() > 1.000001:
                raise ValueError(f"{name}: recorded actions or physical controls exceed limits")
            if name == "motors_off" and np.any(controls):
                raise ValueError("motors_off contains nonzero delivered controls")
            mask = read("substep_qualified", (steps, repeat, worlds))
            final_mask = read("final_reach_qualified", (worlds,))
            if mask.dtype != bool or final_mask.dtype != bool or not np.array_equal(final_mask, mask[-1, -1]):
                raise ValueError(f"{name}: expected matching boolean qualification masks")
            dwell_steps = read("reach_max_qualified_dwell_steps", (worlds,))
            if not np.issubdtype(dwell_steps.dtype, np.integer):
                raise ValueError(f"{name}: expected integer dwell steps")
            dwell_s = read("reach_max_qualified_dwell_s", (worlds,))
            final_values = {key: read("final_reach_" + key, (worlds,)) for key in LIMITS}
            positions = np.concatenate([q[:1], sub.reshape(-1, worlds, 23)])
            if not np.allclose(norm(positions[..., 19:23]), 1., rtol=0., atol=1e-4):
                raise ValueError(f"{name}: invalid free-pin quaternion")
            rows = []
            for world in range(worlds):
                row = world_report(model, positions[:, world], action[:, world], ratios[:, world], repeat,
                                   mask[:, :, world].reshape(-1), {k: v[world] for k, v in final_values.items()},
                                   dwell_steps[world], dwell_s[world])
                rows.append(dict(world=world, **row))
            branches[name] = dict(aggregate=aggregate(rows), worlds=rows)

    # Older traces can be copied on their own. Attribute the baseline only when
    # the paired evaluation receipt identifies this exact evidence and scene.
    receipt_path = trace_path.with_name("summary.json")
    if receipt_path.exists():
        summary = json.loads(receipt_path.read_text(encoding="utf-8"))
        evaluation = summary.get("evaluation", {})
        if evaluation.get("file") == trace_path.name:
            if evaluation["sha256"] != trace_hash or summary["contract"]["model_sha256"] != scene_hash:
                raise ValueError("paired evaluation receipt digest differs from supplied trace or scene")
            report["evaluation_receipt"] = {key: evaluation[key] for key in (
                "policy_roles", "training_reset_sha256", "evaluation_reset_sha256", "baseline_source"
            ) if key in evaluation}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.trace, args.scene)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, branch in report["branches"].items():
        agg = branch["aggregate"]
        closest = agg["mean_snapshots"]["closest"]
        print(f"{name}: {agg['success_worlds']}/{report['world_count']} worlds met dwell, "
              f"mean closest distance {closest['distance_m']:.4f} m at {closest['time_s']:.3f} s")


if __name__ == "__main__":
    main()
