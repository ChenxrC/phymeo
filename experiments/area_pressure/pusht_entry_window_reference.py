"""Compare a completed GPU window with a separately advanced FP64 trajectory.

Run with ``python -m experiments.area_pressure.pusht_entry_window_reference
--trajectory ORIGINAL_DIRECTORY --window full-window.json --output NEW.json``.
This consumes existing GPU results; it does not instantiate NativeRowDevice or
launch CUDA. The reference shares the discrete geometry/constitutive model and
the existing FP64 root, not the GPU trajectory or its force/velocity seeds.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from phymeo.training.manipulation_tasks import manipulation_scenes

from .pusht_entry_area import build_area
from .pusht_entry_rows import (
    DEFAULT_INPUTS,
    DEFAULT_POSES,
    _array,
    _digest,
    _read_json,
    _reject_constant,
    assemble_entry,
    iter_controls,
    load_entries,
)
from .pusht_entry_working_set import _reference

CHANNELS = ("ctrl", "qfrc_applied", "xfrc_applied")


def _require(condition, message):
    if not condition:
        raise ValueError("PushT independent window reference: " + message)


def _number(value, name):
    _require(type(value) in (int, float) and np.isfinite(value), name + " must be finite")
    return float(value)


def _controls_match(step, command, model):
    """Missing legacy values are recorded as missing, never inferred as matched."""
    shapes = ((model.nu,), (model.nv,), (model.nbody, 6))
    packets = []
    if any(key in step for key in CHANNELS):
        packets.append(("top_level", step))
    for name in ("entry_control", "controls"):
        if name in step:
            _require(isinstance(step[name], dict), name + " must be a control object")
            packets.append((name, step[name]))
    for name, packet in packets:
        _require(all(key in packet for key in CHANNELS), name + " is missing control channels")
        for key, shape in zip(CHANNELS, shapes, strict=True):
            _require(
                np.array_equal(_array(packet[key], shape, name + "." + key), command[key]),
                f"tick {command['tick']}: recorded {name}.{key} differs from original control",
            )
    for key in (
        "trace_line_sha256",
        "saved_trace_sha256",
        "archive_sha256",
        "model_sha256",
        "source_sha256",
        "world",
        "control_source_revision",
        "control_revision",
    ):
        if key in step:
            _require(step[key] == command[key], f"tick {command['tick']}: {key} binding differs")
    return dict(
        values="matched_recorded_values" if packets else "legacy_values_not_recorded",
        recorded_value_locations=[name for name, _ in packets],
        original_line_hash_recorded="trace_line_sha256" in step,
    )


def _load_window(trajectory, window, model, source, entries):
    initial = entries["first_load"]
    summary = _read_json(window)
    count, first = summary["accepted"], initial["tick"]
    _require(
        summary["status"] == "complete"
        and type(count) is int
        and count > 0
        and type(summary["requested"]) is int
        and count == summary["requested"],
        "window must be complete with every requested step accepted",
    )
    _require(
        summary["start_tick"] == first and summary["scene_sha256"] == source,
        "GPU window initial tick/model source differs",
    )
    _require(
        summary["mode"] == "cuda-native-rows-host-area-open-loop"
        and summary["gradient_pa_m"] == 1e6,
        "unexpected GPU window law/mode",
    )
    _require(
        count <= entries["withdrawn"]["tick"] - first + 1,
        "GPU window extends beyond the recorded runner window",
    )
    if "source_trace_sha256" in summary:
        _require(
            summary["source_trace_sha256"] == initial["saved_trace_sha256"],
            "GPU summary original trace binding differs",
        )
    commands = list(iter_controls(trajectory, first, first + count))
    _require(len(commands) == count, "original control count differs")
    _require(
        commands[0]["trace_line_sha256"] == initial["trace_line_sha256"],
        "first_load does not bind to this original archive entry",
    )
    for command in commands:
        _require(
            command["source_sha256"] == source
            and command["saved_trace_sha256"] == initial["saved_trace_sha256"],
            "original control model/archive binding differs",
        )

    trace_path = window.with_suffix(".jsonl")
    _require(trace_path != window, "summary and JSONL trace must differ")
    steps, checks = [], []
    q, v = np.array(initial["q"], copy=True), np.array(initial["v"], copy=True)
    time, h = initial["time"], float(model.opt.timestep)
    with trace_path.open("rb") as stream:
        for index, line in enumerate(stream):
            _require(index < count, "GPU trace has more entries than its summary")
            step = json.loads(line, parse_constant=_reject_constant)
            command = commands[index]
            _require(
                type(step["tick"]) is int and step["tick"] == command["tick"],
                "GPU entry ticks are not contiguous original input ticks",
            )
            entry_q = _array(step["entry_q"], (model.nq,), "GPU entry_q")
            entry_v = _array(step["entry_v"], (model.nv,), "GPU entry_v")
            _require(
                np.array_equal(q, entry_q) and np.array_equal(v, entry_v),
                f"tick {step['tick']}: GPU entry does not continue the preceding output",
            )
            next_q = _array(step["next_q"], (model.nq,), "GPU next_q")
            velocity = _array(step["velocity"], (model.nv,), "GPU velocity")
            integrated = entry_q.copy()
            mj.mj_integratePos(model, integrated, velocity, h)
            _require(
                np.array_equal(integrated, next_q),
                f"tick {step['tick']}: GPU q does not match its recorded velocity integration",
            )
            _number(step["area_normal_n"], "GPU area_normal_n")
            _number(step["quadrature"]["clearance_m"], "GPU entry clearance")
            _require(
                type(step["quadrature"]["bound_target_met"]) is bool,
                "GPU quadrature observation must be explicit",
            )
            checks.append(_controls_match(step, command, model))
            step = dict(step, window_line_sha256=hashlib.sha256(line).hexdigest())
            steps.append(step)
            q, v = next_q, velocity
            time += h
    _require(len(steps) == count, "GPU trace is shorter than its completed summary")
    final = summary["final_state"]
    _require(
        type(final["tick"]) is int
        and final["tick"] == first + count
        and final["time"] == time
        and np.array_equal(_array(final["q"], (model.nq,), "summary q"), q)
        and np.array_equal(_array(final["v"], (model.nv,), "summary v"), v),
        "GPU summary final_state differs from its contiguous trace",
    )
    return summary, steps, commands, checks


def _layout(model):
    free = np.flatnonzero(model.jnt_type == int(mj.mjtJoint.mjJNT_FREE))
    hinges = np.flatnonzero(model.jnt_type == int(mj.mjtJoint.mjJNT_HINGE))
    _require(
        len(free) == 1 and len(hinges) == 2 and model.nq == 9 and model.nv == 8,
        "expected the original RR plus free-object layout",
    )
    return (
        int(model.jnt_qposadr[free[0]]),
        int(model.jnt_dofadr[free[0]]),
        [
            dict(
                name=model.joint(int(j)).name,
                q=int(model.jnt_qposadr[j]),
                v=int(model.jnt_dofadr[j]),
            )
            for j in hinges
        ],
    )


def _errors(q, v, gpu_q, gpu_v, layout):
    qa, va, hinges = layout
    a, b = q[qa + 3 : qa + 7].copy(), gpu_q[qa + 3 : qa + 7].copy()
    _require(np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0, "nonzero free-joint quaternion")
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    if a @ b < 0:
        b = -b
    # Sign-invariant geodesic angle; avoids acos cancellation near agreement.
    angle = 4 * np.arctan2(np.linalg.norm(a - b), np.linalg.norm(a + b))
    dv = v - gpu_v
    result = dict(
        free_translation_norm_m=float(np.linalg.norm(q[qa : qa + 3] - gpu_q[qa : qa + 3])),
        quaternion_angle_rad=float(angle),
        hinge_angle_max_abs_rad=max(float(abs(q[j["q"]] - gpu_q[j["q"]])) for j in hinges),
        generalized_velocity_max_abs=float(np.max(abs(dv))),
        generalized_velocity_l2=float(np.linalg.norm(dv)),
        free_linear_velocity_norm_m_s=float(np.linalg.norm(dv[va : va + 3])),
        free_angular_velocity_norm_rad_s=float(np.linalg.norm(dv[va + 3 : va + 6])),
        hinge_velocity_max_abs_rad_s=max(float(abs(dv[j["v"]])) for j in hinges),
    )
    for joint in hinges:
        result["hinge_angle_abs_rad:" + joint["name"]] = float(
            abs(q[joint["q"]] - gpu_q[joint["q"]])
        )
    return result


def _peak(peaks, name, value, tick):
    if name not in peaks or value > peaks[name]["value"]:
        peaks[name] = dict(value=float(value), input_tick=tick, output_tick=tick + 1)


def _separation(steps):
    had_overlap, separated = False, False
    events = []
    for step in steps:
        clearance = step["quadrature"]["clearance_m"]
        if clearance < 0:
            had_overlap, separated = True, False
        elif clearance > 0 and had_overlap and not separated:
            events.append(step["tick"])
            separated = True
    return dict(
        first_input_tick=events[0] if events else None,
        transition_input_ticks=events,
        overlap_observed=had_overlap,
        separation_observed=bool(events),
        last_observed_input_tick=steps[-1]["tick"] if steps else None,
    )


def compare_window(trajectory, window):
    """Return a JSON-ready archive, preserving a failed reference's completed prefix."""
    trajectory, window = Path(trajectory).resolve(), Path(window).resolve()
    entries = load_entries()
    initial = entries["first_load"]
    scene = next(s for s in manipulation_scenes() if s.key == "pusht-planar")
    model = mj.MjModel.from_xml_string(scene.mjcf)
    summary, gpu_steps, commands, control_checks = _load_window(
        trajectory, window, model, scene.source_sha256, entries
    )
    h, layout = float(model.opt.timestep), _layout(model)
    source_files = (
        Path(__file__),
        Path(_reference.__code__.co_filename),
        Path(assemble_entry.__code__.co_filename),
        Path(build_area.__code__.co_filename),
        Path(__file__).with_name("capsule_box_reference.py"),
        Path(__file__).with_name("pusht_saved_contact.py"),
    )
    report = dict(
        schema="phymeo-pusht-independent-window-reference-v1",
        status="running",
        comparison="independently advanced FP64 versus recorded GPU; same discrete contact model",
        requested=len(commands),
        reference_steps_completed=0,
        h=h,
        provenance=dict(
            trajectory_directory=str(trajectory),
            window_summary_path=str(window),
            window_summary_sha256=_digest(window),
            window_trace_sha256=_digest(window.with_suffix(".jsonl")),
            original_manifest_sha256=_digest(trajectory / "manifest.json"),
            original_trace_sha256=initial["saved_trace_sha256"],
            source_sha256=scene.source_sha256,
            control_archive_sha256=commands[0]["archive_sha256"],
            model_sha256=commands[0]["model_sha256"],
            first_load_trace_line_sha256=initial["trace_line_sha256"],
            portable_inputs_sha256=_digest(DEFAULT_INPUTS),
            poses_sha256=_digest(DEFAULT_POSES),
            implementation_sha256={path.name: _digest(path) for path in source_files},
            numpy_version=np.__version__,
            mujoco_version=mj.__version__,
        ),
        validation=dict(
            gpu_contiguous_entries=len(gpu_steps),
            gpu_position_integrations_checked=len(gpu_steps),
            gpu_summary_final_state_matches=True,
            original_controls_authenticated_by_iter_controls=True,
            gpu_steps_with_value_comparison=sum(
                c["values"] == "matched_recorded_values" for c in control_checks
            ),
            gpu_steps_without_recorded_control_values=sum(
                c["values"] == "legacy_values_not_recorded" for c in control_checks
            ),
            gpu_steps_with_original_line_hash=sum(
                c["original_line_hash_recorded"] for c in control_checks
            ),
        ),
        interpretation=dict(
            existing_per_step_fp64="GPU max_velocity_error uses the GPU entry state; it is a local solve comparison",
            independent_fp64="only first_load q/v are shared; later q/v and geometry follow the FP64 trajectory",
            control_mode="original held controls, without recomputing feedback from either replay trajectory",
            shared_geometry="same build_area implementation, rebuilt at each trajectory's own entry state",
            accumulated_errors="state difference after each step, not a sum of local solve errors",
            hinge_angles="unwrapped coordinate differences, including joint-limit excursions",
            generalized_velocity_norms="mixed linear/angular coordinate units; component-group norms also reported",
            separation="first strictly positive entry clearance after a negative entry clearance; no load threshold",
            quadrature="bound_target_met is preserved as a static quadrature observation, not a 2% geometry certificate",
            physical_calibration_validated=False,
            geometric_accuracy_validated=False,
            cumulative_error_acceptance_threshold=None,
        ),
        original_archive_summary=_read_json(trajectory / "manifest.json"),
        gpu_window_summary=summary,
        independent_initial_state=dict(
            tick=initial["tick"],
            time=initial["time"],
            q=initial["q"].tolist(),
            v=initial["v"].tolist(),
        ),
        accumulated_error_peaks={},
        load_peaks={},
        trajectory=[],
    )
    for step in gpu_steps:
        _peak(report["load_peaks"], "gpu_area_normal_n", step["area_normal_n"], step["tick"])
    local_errors = [
        (_number(s["max_velocity_error"], "local FP64 velocity error"), s["tick"])
        for s in gpu_steps
    ]
    local_peak = max(local_errors, key=lambda item: item[0])
    report["existing_gpu_entry_fp64_velocity_error_peak"] = dict(
        value=local_peak[0], input_tick=local_peak[1]
    )
    q, v = initial["q"].copy(), initial["v"].copy()
    tick, time = initial["tick"], initial["time"]
    try:
        for command, gpu, check in zip(commands, gpu_steps, control_checks, strict=True):
            _require(command["tick"] == tick, "independent trajectory skipped an original control")
            entry = {key: command[key] for key in CHANNELS}
            entry.update(q=q, v=v, tick=tick, time=time)
            rows = assemble_entry(model, entry)
            area = build_area(model, rows["data"])
            response = rows["mass"] + h * np.diag(model.dof_damping)
            next_v, row_forces, point_forces = _reference(rows, area, v, response, h)
            next_v = _array(next_v, (model.nv,), "independent reference velocity")
            row_forces = _array(row_forces, (len(rows["owners"]),), "independent native forces")
            point_forces = _array(
                point_forces, (len(area["normals"]), 3), "independent point forces"
            )
            next_q = q.copy()
            mj.mj_integratePos(model, next_q, next_v, h)
            next_q = _array(next_q, (model.nq,), "independent reference next_q")
            error = _errors(
                next_q, next_v, np.asarray(gpu["next_q"]), np.asarray(gpu["velocity"]), layout
            )
            normal_n = float(np.sum(point_forces * area["normals"]))
            for name, value in error.items():
                _peak(report["accumulated_error_peaks"], name, value, tick)
            _peak(report["load_peaks"], "independent_area_normal_n", normal_n, tick)
            command_record = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in command.items()
            }
            report["trajectory"].append(
                dict(
                    tick=tick,
                    output_tick=tick + 1,
                    entry_time=time,
                    output_time=time + h,
                    original_control=command_record,
                    gpu_control_verification=check,
                    gpu_window_line_sha256=gpu["window_line_sha256"],
                    entry_q=q.tolist(),
                    entry_v=v.tolist(),
                    next_q=next_q.tolist(),
                    velocity=next_v.tolist(),
                    native_rows=len(row_forces),
                    owners=rows["owners"],
                    native_forces=row_forces.tolist(),
                    area_point_forces=point_forces.tolist(),
                    area_normal_n=normal_n,
                    quadrature=area["quadrature"],
                    gpu=dict(
                        entry_q=gpu["entry_q"],
                        entry_v=gpu["entry_v"],
                        next_q=gpu["next_q"],
                        velocity=gpu["velocity"],
                        area_normal_n=gpu["area_normal_n"],
                        quadrature=gpu["quadrature"],
                        same_entry_fp64_max_velocity_error=gpu["max_velocity_error"],
                    ),
                    accumulated_error=error,
                )
            )
            q, v = next_q, next_v
            tick += 1
            time += h
            report["reference_steps_completed"] += 1
        report["status"] = "complete"
    except Exception as error:
        report.update(
            status="reference_failed",
            failed_input_tick=tick,
            error_type=type(error).__name__,
            error=str(error),
        )
    report["independent_final_state"] = dict(tick=tick, time=time, q=q.tolist(), v=v.tolist())
    completed = report["trajectory"]
    gpu_separation, ref_separation = _separation(gpu_steps), _separation(completed)
    left, right = ref_separation["first_input_tick"], gpu_separation["first_input_tick"]
    report["geometric_separation"] = dict(
        gpu=gpu_separation,
        independent=ref_separation,
        independent_minus_gpu_ticks=left - right
        if left is not None and right is not None
        else None,
        comparison_complete=report["status"] == "complete",
    )
    report["quadrature_observations"] = dict(
        gpu_unmet_static_bound_ticks=[
            s["tick"] for s in gpu_steps if not s["quadrature"]["bound_target_met"]
        ],
        independent_unmet_static_bound_ticks=[
            s["tick"] for s in completed if not s["quadrature"]["bound_target_met"]
        ],
    )
    if completed:
        gpu_peak = report["load_peaks"]["gpu_area_normal_n"]
        ref_peak = report["load_peaks"]["independent_area_normal_n"]
        report["area_normal_peak_difference"] = dict(
            independent_minus_gpu_n=ref_peak["value"] - gpu_peak["value"],
            independent_minus_gpu_input_ticks=ref_peak["input_tick"] - gpu_peak["input_tick"],
            same_complete_window=report["status"] == "complete",
        )
        report["final_accumulated_error"] = completed[-1]["accumulated_error"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--window", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Reserve the new artifact before the potentially long CPU-only reference.
    # Input-validation failures also leave an explicit failure receipt.
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            report = compare_window(args.trajectory, args.window)
        except Exception as error:
            json.dump(
                dict(
                    schema="phymeo-pusht-independent-window-reference-v1",
                    status="validation_failed",
                    error_type=type(error).__name__,
                    error=str(error),
                ),
                stream,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            raise
        json.dump(report, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
