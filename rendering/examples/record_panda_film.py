"""Record a continuous Phymeo grasp and a portable sample of its actual endpoints.

The archive contains every input and output produced by the physical run. The
excerpt uses the existing PandaReplaySource format and can be read without
installing Phymeo, regenerating contacts, or loading the original full trace.
"""
import argparse
import hashlib
import json
from pathlib import Path

from _panda_motion import STATE_PHASE, sample_indices
from phymeo_rendering.adapters.panda_replay import MODEL_SHA256


def encoded(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def record(output, schedule):
    from math import pi
    from phymeo.experiments.panda_cube import run_panda_cube

    output.mkdir(parents=True, exist_ok=False)
    result = run_panda_cube(
        initial_yaw_rad=pi / 15, initial_xy_offset_m=(0.003, -0.001),
        box_size_m=(0.046, 0.058, 0.040), closed_width_m=0.024,
        grip_above_com_m=0.012, mass_kg=0.12, friction=0.65,
        motion_duration_s=10.0 if schedule == "uniform" else 2.25,
        return_finish_s=10.0 if schedule == "return" else None, record=True,
    )
    archive = encoded(result)
    (output / "archive.json").write_bytes(archive)
    archive_sha = digest(archive)
    trace, inputs = result["trace"], result["inputs"]
    dt = inputs["dt_s"]
    if len(trace) != inputs["steps"] or len(trace) != 16000 or trace[-1]["t"] != 10:
        raise ValueError("incomplete ten-second physical window")
    if not all(row["t"] == (i + 1) * dt for i, row in enumerate(trace)):
        raise ValueError("physical step timestamps are discontinuous")
    indices = sample_indices(record_count=len(trace), dt_s=dt, sample_hz=30)
    sample = [{"source_index": i, "record": trace[i]} for i in indices]
    raw = encoded(dict(schema="phymeo-panda-archive-excerpt-v1", inputs=inputs, records=sample))
    (output / "trace.json").write_bytes(raw)
    manifest = dict(schema="phymeo-panda-archive-manifest-v1", archive_sha256=archive_sha,
                    archive_record_count=len(trace), excerpt_sha256=digest(raw),
                    model_sha256=MODEL_SHA256, live_physics_identity=None,
                    base_pose=dict(position_m=[0, 0, 0], orientation_xyzw=[0, 0, 0, 1]),
                    state_phase=STATE_PHASE,
                    records=[dict(source_index=i, raw_record_sha256=digest(encoded(trace[i])))
                             for i in indices])
    (output / "manifest.json").write_bytes(encoded(manifest))
    fields = ("success", "carried_during_transfer", "released_on_target", "first_unresolved_structure",
              "steps", "simulated_seconds", "initial_payload_position_m", "carry_clearance_m",
              "min_carry_bilateral_normal_N", "peak_carry_table_N", "final_target_error_m",
              "peak_bilateral_normal_N", "final_table_force_N", "release_pad_force_N",
              "max_joint_limit_violation", "max_cube_momentum_residual_Ns")
    summary = {k: result[k] for k in fields}
    summary.update(schedule=schedule, archive_sha256=archive_sha, excerpt_sha256=digest(raw),
                   manifest_sha256=digest(encoded(manifest)), physical_dt_s=dt,
                   controller_knot_times_s=inputs["knot_times_s"], sampled_frames=len(sample),
                   first_endpoint_s=trace[indices[0]]["t"], last_endpoint_s=trace[indices[-1]]["t"])
    (output / "recording.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schedule", choices=("return", "uniform"), default="return")
    args = parser.parse_args()
    record(args.output, args.schedule)
