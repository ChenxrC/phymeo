"""Source-bound A/B readouts for the original and half-load-compensated PushT route.

python -m experiments.area_pressure.pusht_load_comparison --baseline OLD.json
    --candidate NEW.json --baseline-audit OLD_AUDIT.json --candidate-audit NEW_AUDIT.json
    --output AB.json [--plot AB.png]

Independent saved-data audits are required. This tool binds their exact inputs;
CPU queries below evaluate only named geometry/pose, never dynamics or controls.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .pusht_active_comparison import H, _constant, _digest, _pose_errors, _read


def _require(ok, reason):
    if not ok:
        raise ValueError("PushT load comparison: " + reason)


def bind_comparison(baseline, candidate, baseline_audit, candidate_audit):
    """Compare only the intersection of two independently verified windows."""
    for summary, audit in ((baseline, baseline_audit), (candidate, candidate_audit)):
        _require(audit["status"] == "verified-saved-prefix", "independent audit status")
        _require(audit["verified_steps"] == summary["recorded_steps"], "audit prefix count")
        _require(
            0
            <= summary["recorded_steps"]
            <= summary["accepted_steps"]
            <= summary["requested_steps"],
            "accepted/recorded/requested count",
        )
    _require(
        baseline_audit["source_checkpoint_sha256"] == candidate_audit["source_checkpoint_sha256"],
        "different starting checkpoints",
    )
    _require(
        baseline["initial_state"] == candidate["initial_state"]
        and _same_f32(baseline["initial_state"]["q"], candidate["initial_state"]["q"])
        and _same_f32(baseline["initial_state"]["v"], candidate["initial_state"]["v"]),
        "different initial state/clock",
    )
    for name in ("start_tick", "world", "local_tick_origin", "requested_steps", "goal"):
        _require(baseline[name] == candidate[name], "different " + name)
    for name in (
        "start_tick",
        "end_tick",
        "duration_s",
        "elapsed_at_entry_s",
        "remaining_steps",
        "nodes",
        "sha256",
        "route_index",
    ):
        _require(
            baseline["active_route"][name] == candidate["active_route"][name],
            "different route " + name,
        )
    for name in ("source_sha256", "scene_asset_sha256"):
        _require(
            baseline["provenance"][name] == candidate["provenance"][name],
            "different physical " + name,
        )
    _require(
        baseline.get("control_variant", "position") == "position"
        and candidate.get("control_variant") == "load-compensation-half",
        "expected control variants",
    )
    _require(
        baseline_audit["control"].get("variant", "position") == "position"
        and candidate_audit["control"].get("variant") == "load-compensation-half",
        "audit control variants",
    )
    common = min(baseline["recorded_steps"], candidate["recorded_steps"])
    return dict(
        common_steps=common,
        full_recorded_window=common == baseline["requested_steps"],
        excluded_baseline_steps=baseline["recorded_steps"] - common,
        excluded_candidate_steps=candidate["recorded_steps"] - common,
        baseline_unrecorded_accepted_steps=baseline["accepted_steps"] - baseline["recorded_steps"],
        candidate_unrecorded_accepted_steps=candidate["accepted_steps"]
        - candidate["recorded_steps"],
        source_checkpoint_sha256=baseline_audit["source_checkpoint_sha256"],
        route_sha256=baseline["active_route"]["sha256"],
    )


def _same_f32(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def causal_prefix(baseline, candidate):
    """A changed input state must follow, not precede, a changed submitted command."""
    _require(len(baseline) == len(candidate), "common causal window length")
    first_comp = first_ctrl = first_state = None
    identical = 0
    for a, b in zip(baseline, candidate, strict=True):
        tick = a["tick"]
        _require(b["tick"] == tick, "unaligned causal ticks")
        _require(not np.any(np.asarray(a["correction_nm"])), "baseline contains compensation")
        if first_comp is None and np.any(np.asarray(b["correction_nm"])):
            first_comp = tick
        same_ctrl = _same_f32(a["ctrl"], b["ctrl"])
        same_state = _same_f32(a["q"], b["q"]) and _same_f32(a["v"], b["v"])
        if not same_ctrl and first_ctrl is None:
            _require(
                first_comp is not None and first_comp <= tick,
                "command changed before nonzero compensation",
            )
            _require(same_state, "first command difference already uses a different input state")
            first_ctrl = tick
        if not same_state and first_state is None:
            _require(
                first_ctrl is not None and first_ctrl < tick,
                "state changed before a causal command",
            )
            first_state = tick
        if first_ctrl is None:
            _require(same_ctrl and same_state, "unexplained pre-command divergence")
            identical += 1
    return dict(
        first_nonzero_compensation_tick=first_comp,
        first_command_difference_tick=first_ctrl,
        first_input_state_difference_tick=first_state,
        identical_entry_and_command_steps=identical,
        predivergence_fp32_bytes_verified=True,
        interpretation="input-state changes can first follow a different submitted command; no future state is used",
    )


def stage_motion(rows, stage):
    selected = [row for row in rows if row["stage"] == stage]
    if not selected:
        return dict(
            samples=0,
            input_tick_start=None,
            input_tick_end=None,
            pusher_advance_m=None,
            reference_advance_m=None,
            object_advance_m=None,
        )
    first, last = selected[0], selected[-1]
    return dict(
        samples=len(selected),
        input_tick_start=first["tick"],
        input_tick_end=last["tick"],
        pusher_advance_m=last["pusher_along_m"] - first["pusher_along_m"],
        reference_advance_m=last["reference_along_m"] - first["reference_along_m"],
        object_advance_m=last["object_along_m"] - first["object_along_m"],
        phase="first to last input configuration in this same stage",
    )


def _load_run(summary_path, audit_path):
    summary_path, audit_path = Path(summary_path).resolve(), Path(audit_path).resolve()
    summary, audit = _read(summary_path), _read(audit_path)
    _require(
        summary["schema"] == "phymeo-pusht-active-route-v1"
        and audit["schema"] == "phymeo-pusht-active-comparison-v1",
        "saved schemas",
    )
    _require(
        audit["source"]["summary_sha256"] == _digest(summary_path)
        and audit["source"]["journal_sha256"] == summary["journal"]["sha256"]
        and audit["source_endpoint_bound"] is True,
        "audit/source binding",
    )
    journal_path = (summary_path.parent / summary["journal"]["file"]).resolve()
    _require(_digest(journal_path) == summary["journal"]["sha256"], "journal changed after audit")
    _require(
        audit["verified_steps"] == summary["recorded_steps"] == summary["journal"]["records"],
        "independent recorded count",
    )
    if summary.get("control_variant") == "load-compensation-half":
        _require(
            audit["control"]["load_compensation_verified_steps"] == audit["verified_steps"],
            "compensation not independently audited",
        )
    rows = []
    with journal_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line, parse_constant=_constant)
            rows.append(
                {
                    key: row[key]
                    for key in (
                        "original_input_tick",
                        "original_output_tick",
                        "input_state",
                        "output_state",
                        "submitted_control",
                        "controller",
                        "area",
                        "native",
                        "geometry",
                        "goal_result",
                        "feedback_accepted",
                    )
                }
            )
    _require(len(rows) == audit["verified_steps"], "journal row count")
    endpoint = rows[-1]["output_state"] if rows else summary["initial_state"]
    _require(endpoint == audit["verified_final_state"], "audit verified endpoint")
    return dict(
        summary=summary,
        audit=audit,
        rows=rows,
        source=dict(
            summary_file=str(summary_path),
            summary_sha256=_digest(summary_path),
            audit_file=str(audit_path),
            audit_sha256=_digest(audit_path),
            journal_sha256=summary["journal"]["sha256"],
        ),
    )


class _Geometry:
    """Named original-model FK and primitive gap queries, with no integration."""

    def __init__(self, summary):
        import mujoco as mj

        from phymeo.training.manipulation_tasks import manipulation_scenes

        self.mj = mj
        self.scene = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
        _require(
            self.scene.source_sha256 == summary["provenance"]["source_sha256"],
            "original model source",
        )
        self.model = mj.MjModel.from_xml_string(self.scene.mjcf)
        self.data = mj.MjData(self.model)
        self.robot_qa = [
            int(self.model.jnt_qposadr[self.model.joint(name).id])
            for name in self.scene.robot_joint_names
        ]
        self.pusher = int(self.model.geom("pusher").id)
        self.object_geoms = tuple(
            int(self.model.geom(name).id) for name in self.scene.object_geom_names
        )
        self.target = next(
            t
            for t in self.scene.targets
            if list(t.position[:2]) == summary["goal"]["target_xy"]
            and t.yaw == summary["goal"]["target_yaw"]
        )

    def pose(self, state, target_q=None):
        self.data.qpos[:] = state["q"]
        self.mj.mj_kinematics(self.model, self.data)
        actual = self.data.geom_xpos[self.pusher].copy()
        distances = [
            float(
                self.mj.mj_geomDistance(self.model, self.data, self.pusher, gid, 1.0, np.zeros(6))
            )
            for gid in self.object_geoms
        ]
        depth = max(0.0, -min(distances))
        if target_q is None:
            desired = None
        else:
            self.data.qpos[self.robot_qa] = target_q
            self.mj.mj_kinematics(self.model, self.data)
            desired = self.data.geom_xpos[self.pusher].copy()
        return actual, desired, depth

    def errors(self, state, goal):
        from phymeo.training.manipulation_tasks import footprint_iou

        result = _pose_errors(state, goal)
        result["footprint_iou"] = float(
            footprint_iou(
                self.scene.object_shapes,
                state["object_position_world_m"],
                state["object_quaternion_wxyz"],
                self.target,
            )
        )
        return result


def _curve(row, direction, geometry, goal):
    state, controller, area = row["input_state"], row["controller"], row["area"]
    actual, target, depth = geometry.pose(state, controller["target_q"])
    comp = controller.get("load_compensation", {})
    correction = np.asarray(comp.get("correction_nm", [0.0, 0.0]), dtype=float)
    force_a = np.asarray(area["total_force_on_pusher_world_n"], dtype=float)
    force_b = np.asarray(row["feedback_accepted"]["measurement"]["group_force_world"], dtype=float)
    _require(np.array_equal(force_a, -force_b), "saved bilateral force ownership")
    support = np.zeros(3)
    for contact in row["native"]["supports"]:
        if set(contact["geometry_names"]) & set(geometry.scene.object_geom_names):
            support += np.asarray(contact["force_on_supported_world_n"])
    metrics = dict(row["goal_result"]["metrics"])
    pd = controller["pd"]
    unclipped = np.asarray(comp.get("combined_unclipped_nm", pd["unclipped_effort"]), dtype=float)
    effort = np.asarray(comp.get("combined_effort_nm", pd["effort"]), dtype=float)
    command = np.asarray(row["submitted_control"]["ctrl"], dtype=float)
    return dict(
        tick=row["original_input_tick"],
        time_s=state["time"],
        stage=controller["stage"],
        object_xy_m=state["object_position_world_m"][:2],
        object_yaw_rad=_pose_errors(state, goal)["yaw_rad"],
        position_error_m=metrics["position_error_m"],
        yaw_error_rad=metrics["yaw_error_rad"],
        footprint_iou=metrics["footprint_iou"],
        normal_load_n=area["normal_load_n"],
        area_m2=row["geometry"]["clipped_area_m2"],
        pusher_world_m=actual.tolist(),
        reference_world_m=target.tolist(),
        pusher_along_m=float(actual @ direction),
        reference_along_m=float(target @ direction),
        object_along_m=float(np.asarray(state["object_position_world_m"]) @ direction),
        tool_force_along_n=float(force_a @ direction),
        object_force_along_n=float(force_b @ direction),
        support_force_along_n=float(support @ direction),
        ctrl_nm=command.tolist(),
        correction_nm=correction.tolist(),
        compensation_enabled=bool(comp.get("enabled", False)),
        saturated=bool(np.any(unclipped != effort)),
        robot_position_error_rad=float(np.max(abs(np.asarray(pd["target"]) - pd["position"]))),
        robot_velocity_error_rad_s=float(
            np.max(abs(np.asarray(pd["target_velocity"]) - pd["velocity"]))
        ),
        pusher_tracking_error_m=float(np.linalg.norm(target - actual)),
        source_capsule_box_penetration_m=depth,
    )


def _lane_metrics(run, rows, curves, direction, geometry):
    summary = run["summary"]
    initial = summary["initial_state"]
    endpoint = rows[-1]["output_state"] if rows else initial
    p0, _, _ = geometry.pose(initial)
    p1, _, _ = geometry.pose(endpoint)
    displacement = (
        np.asarray(endpoint["object_position_world_m"]) - initial["object_position_world_m"]
    )
    first, last = (
        geometry.errors(initial, summary["goal"]),
        geometry.errors(endpoint, summary["goal"]),
    )
    loaded = [c for c in curves if c["normal_load_n"] > 0]
    peak = max(curves, key=lambda c: c["normal_load_n"]) if curves else None
    push = stage_motion(curves, "push")
    push_rows = [row for row in curves if row["stage"] == "push"]
    push.update(
        normal_impulse_ns=float(H * sum(c["normal_load_n"] for c in push_rows)),
        tool_impulse_along_ns=float(H * sum(c["tool_force_along_n"] for c in push_rows)),
        object_impulse_along_ns=float(H * sum(c["object_force_along_n"] for c in push_rows)),
        table_on_object_impulse_along_ns=float(
            H * sum(c["support_force_along_n"] for c in push_rows)
        ),
    )
    return dict(
        compared_steps=len(rows),
        initial_state=initial,
        compared_endpoint_state=endpoint,
        initial=first,
        endpoint=last,
        endpoint_phase="output of the last common recorded step; independent pose/IoU, not a new Goal evaluation",
        object_displacement_world_m=displacement.tolist(),
        object_advance_m=float(displacement @ direction),
        pusher_advance_m=float((p1 - p0) @ direction),
        position_error_change_m=last["position_error_m"] - first["position_error_m"],
        yaw_error_change_rad=last["yaw_error_rad"] - first["yaw_error_rad"],
        footprint_iou_change=last["footprint_iou"] - first["footprint_iou"],
        push=push,
        normal_impulse_ns=float(H * sum(c["normal_load_n"] for c in curves)),
        peak_normal_load_n=peak["normal_load_n"] if peak else None,
        peak_normal_load_input_tick=peak["tick"] if peak else None,
        first_positive_load_input_tick=loaded[0]["tick"] if loaded else None,
        tool_impulse_along_ns=float(H * sum(c["tool_force_along_n"] for c in curves)),
        object_impulse_along_ns=float(H * sum(c["object_force_along_n"] for c in curves)),
        table_on_object_impulse_along_ns=float(H * sum(c["support_force_along_n"] for c in curves)),
        saturated_steps=sum(c["saturated"] for c in curves),
        peak_absolute_motor_torque_nm=np.max(
            np.abs([c["ctrl_nm"] for c in curves]), axis=0
        ).tolist()
        if curves
        else None,
        max_source_capsule_box_penetration_m=max(
            (c["source_capsule_box_penetration_m"] for c in curves), default=None
        ),
        max_clipped_area_m2=max((c["area_m2"] for c in curves), default=None),
        max_pusher_tracking_error_m=max(
            (c["pusher_tracking_error_m"] for c in curves), default=None
        ),
        max_robot_position_error_rad=max(
            (c["robot_position_error_rad"] for c in curves), default=None
        ),
        max_robot_velocity_error_rad_s=max(
            (c["robot_velocity_error_rad_s"] for c in curves), default=None
        ),
        goal_success_on_common_inputs=any(row["goal_result"]["success"] for row in rows),
        termination=run["audit"]["termination"],
        source_accepted_steps=summary["accepted_steps"],
        source_recorded_steps=summary["recorded_steps"],
        source_final_state_reported=summary["final_state"],
        errors=summary.get("errors"),
        failed_original_input_tick=summary.get("failed_original_input_tick"),
        unrecorded_accepted_step=run["audit"].get("unrecorded_accepted_step"),
    )


def compare(baseline_path, candidate_path, baseline_audit_path, candidate_audit_path):
    base = _load_run(baseline_path, baseline_audit_path)
    candidate = _load_run(candidate_path, candidate_audit_path)
    binding = bind_comparison(
        base["summary"], candidate["summary"], base["audit"], candidate["audit"]
    )
    count = binding["common_steps"]
    pairs = [base["rows"][:count], candidate["rows"][:count]]
    normalized = []
    for rows in pairs:
        normalized.append(
            [
                dict(
                    tick=row["original_input_tick"],
                    q=row["input_state"]["q"],
                    v=row["input_state"]["v"],
                    ctrl=row["submitted_control"]["ctrl"],
                    correction_nm=row["controller"]
                    .get("load_compensation", {})
                    .get("correction_nm", [0.0, 0.0]),
                )
                for row in rows
            ]
        )
    causality = causal_prefix(*normalized)
    output_difference = next(
        (
            a["original_output_tick"]
            for a, b in zip(*pairs, strict=True)
            if not _same_f32(a["output_state"]["q"], b["output_state"]["q"])
            or not _same_f32(a["output_state"]["v"], b["output_state"]["v"])
        ),
        None,
    )
    _require(
        output_difference is None
        or (
            causality["first_command_difference_tick"] is not None
            and output_difference > causality["first_command_difference_tick"]
        ),
        "output state changed before a causal command",
    )
    causality["first_output_state_difference_tick"] = output_difference
    for left, right in zip(*pairs, strict=True):
        a, b = left["controller"], right["controller"]
        _require(
            a["stage"] == b["stage"]
            and a["target_q"] == b["target_q"]
            and a["target_v"] == b["target_v"]
            and a["route"]["route_elapsed_s"] == b["route"]["route_elapsed_s"],
            "different paired reference phase/target",
        )
    route_path = Path(base["summary"]["active_route"]["file"])
    if not route_path.is_absolute():
        route_path = Path(baseline_path).resolve().parent / route_path
    _require(_digest(route_path) == binding["route_sha256"], "saved common route bytes")
    route = _read(route_path)
    direction = np.asarray(route["route"]["proposal"]["direction_world"], dtype=float)
    _require(
        direction.shape == (3,)
        and np.isfinite(direction).all()
        and np.linalg.norm(direction[:2]) > 0,
        "saved planar push direction",
    )
    direction = np.r_[direction[:2] / np.linalg.norm(direction[:2]), 0.0]
    geometry = _Geometry(base["summary"])
    curves = [
        [_curve(row, direction, geometry, base["summary"]["goal"]) for row in rows]
        for rows in pairs
    ]
    metrics = {
        name: _lane_metrics(run, rows, values, direction, geometry)
        for name, run, rows, values in zip(
            ("position", "load-compensation-half"), (base, candidate), pairs, curves, strict=True
        )
    }
    report = dict(
        schema="phymeo-pusht-load-comparison-v1",
        status="compared-audited-common-prefix",
        sources=dict(position=base["source"], load_compensation_half=candidate["source"]),
        binding=binding,
        causal_prefix=causality,
        push_direction_world=direction.tolist(),
        common_input_start_tick=base["summary"]["start_tick"],
        common_input_end_tick=base["summary"]["start_tick"] + count - 1 if count else None,
        lanes=metrics,
        limits=[
            "All A/B metrics use only the common independently audited recorded prefix.",
            "Reference pusher travel is not actual pusher or object displacement.",
            "Push-stage travel uses its first and last input configurations, separate from the window output endpoint.",
            "Native capsule/box signed overlap is an original-shape query, not PL-material penetration or CCD.",
            "Load impulses use saved actual bilateral forces at accepted input geometry and fixed 1 ms steps.",
            "Pose and IoU at the final output do not reevaluate Goal history or establish task success.",
        ],
    )
    report["candidate_minus_baseline"] = {
        key: None
        if metrics["position"][key] is None or metrics["load-compensation-half"][key] is None
        else metrics["load-compensation-half"][key] - metrics["position"][key]
        for key in (
            "object_advance_m",
            "pusher_advance_m",
            "position_error_change_m",
            "yaw_error_change_rad",
            "footprint_iou_change",
            "normal_impulse_ns",
            "peak_normal_load_n",
            "object_impulse_along_ns",
            "saturated_steps",
            "max_source_capsule_box_penetration_m",
            "max_pusher_tracking_error_m",
        )
    }
    curve_data = dict(
        schema="phymeo-pusht-load-curves-v1",
        binding=binding,
        goal=base["summary"]["goal"],
        push_direction_world=direction.tolist(),
        sample_phase="accepted input configuration and same-step actual load",
        lanes=dict(zip(("position", "load-compensation-half"), curves, strict=True)),
    )
    return report, curve_data


def plot_comparison(report, curves, path):
    """Plot already saved comparison/curves; no model, CUDA or audit rerun needed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk = next(
        (
            family
            for family in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei")
            if any(font.name == family for font in font_manager.fontManager.ttflist)
        ),
        None,
    )

    def label(zh, en):
        return zh if cjk else en

    if cjk:
        plt.rcParams["font.family"] = cjk
        plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"position": "#31688e", "load-compensation-half": "#d35e19"}
    styles = {"position": "-", "load-compensation-half": "--"}
    for name, values in curves["lanes"].items():
        if not values:
            continue
        color, style = colors[name], styles[name]
        t = np.asarray([row["time_s"] for row in values])
        display = (
            label("位置控制", "Position control")
            if name == "position"
            else label("位置＋半载荷补偿", "Position + half load")
        )
        settings = dict(color=color, linestyle=style, linewidth=1.8, label=display)
        push = [row for row in values if row["stage"] == "push"]
        push_times = [row["time_s"] for row in push]
        for axis, field, scale in (
            (axes[0, 0], "pusher_along_m", 1000),
            (axes[0, 1], "object_along_m", 1000),
        ):
            if push:
                axis.plot(
                    push_times,
                    scale * (np.asarray([row[field] for row in push]) - push[0][field]),
                    **settings,
                )
        axes[1, 0].plot(t, [row["normal_load_n"] for row in values], **settings)
        axes[1, 1].plot(
            t, 1000 * np.asarray([row["position_error_m"] for row in values]), **settings
        )
        axes[2, 0].plot(t, np.rad2deg([row["yaw_error_rad"] for row in values]), **settings)
        axes[2, 1].plot(
            t, 1000 * np.asarray([row["pusher_tracking_error_m"] for row in values]), **settings
        )
        if name == "position" and push:
            reference = np.asarray([row["reference_along_m"] for row in push])
            axes[0, 0].plot(
                push_times,
                1000 * (reference - reference[0]),
                color="#555555",
                linestyle=":",
                linewidth=1.7,
                label=label("共同参考", "Common reference"),
            )
    for axis, title, ylabel in (
        (
            axes[0, 0],
            label("推杆在推动阶段走了多远？", "Pusher travel during push"),
            label("沿推动方向（mm）", "Along push (mm)"),
        ),
        (
            axes[0, 1],
            label("T 块在推动阶段走了多远？", "Object travel during push"),
            label("沿推动方向（mm）", "Along push (mm)"),
        ),
        (axes[1, 0], label("推杆接触载荷", "Area normal load"), label("法向力（N）", "Force (N)")),
        (
            axes[1, 1],
            label("T 块到目标的位置误差", "Distance to goal"),
            label("距离（mm）", "Distance (mm)"),
        ),
        (
            axes[2, 0],
            label("T 块朝向误差", "Object orientation error"),
            label("偏航误差（度）", "Yaw error (degrees)"),
        ),
        (
            axes[2, 1],
            label("推杆实际中心与参考中心的距离", "Tool tracking error"),
            label("距离（mm）", "Distance (mm)"),
        ),
    ):
        axis.set(xlabel=label("仿真时间（s）", "Simulation time (s)"), ylabel=ylabel, title=title)
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    extent = (
        "full recorded route window"
        if report["binding"]["full_recorded_window"]
        else "COMMON PREFIX ONLY"
    )
    fig.suptitle(
        label(
            "同一路线的两次推动"
            if report["binding"]["full_recorded_window"]
            else "两次推动的共同已验证区间",
            "PushT position vs half-load compensation — " + extent,
        )
    )
    path = Path(path)
    with path.open("xb") as stream:
        fig.savefig(stream, format=path.suffix.lstrip(".") or "png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-audit", type=Path)
    parser.add_argument("--candidate-audit", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--plot-only", type=Path, help="existing comparison JSON; requires --plot")
    args = parser.parse_args()
    if args.plot_only is not None:
        _require(args.plot is not None, "--plot-only requires --plot")
        report = _read(args.plot_only)
        curves_path = args.plot_only.resolve().parent / report["curves"]["file"]
        _require(_digest(curves_path) == report["curves"]["sha256"], "saved curve binding")
        plot_comparison(report, _read(curves_path), args.plot)
        return
    _require(
        all(
            path is not None
            for path in (
                args.baseline,
                args.candidate,
                args.baseline_audit,
                args.candidate_audit,
                args.output,
            )
        ),
        "comparison requires both summaries, both audits, and --output",
    )
    report, curves = compare(
        args.baseline, args.candidate, args.baseline_audit, args.candidate_audit
    )
    curves_path = args.output.with_suffix(".curves.json")
    with curves_path.open("x", encoding="utf-8") as stream:
        json.dump(curves, stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
    report["curves"] = dict(file=curves_path.name, sha256=_digest(curves_path))
    if args.plot is not None:
        plot_comparison(report, curves, args.plot)
        report["plot"] = dict(file=str(args.plot), sha256=_digest(args.plot))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=report["status"], **report["binding"]), allow_nan=False))


if __name__ == "__main__":
    main()
