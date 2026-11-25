"""Reconstruct world-frame COM and foot CoP from saved paired G1 standing journals."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import mujoco as mj
import numpy as np

SIDES = ("gpu", "cpu")
MARKS = (0.0, 0.05, 0.1, 0.25, 0.498, 0.5)
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
DISTANCE_M = 0.005
HOLD_S = 0.010
MIN_FZ_N = 1.0


def _array(value, shape, label):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return value


def ground_cop(
    force_world, moment_about_com_world, body_com_world, *, ground_z=0.0, min_fz=MIN_FZ_N
):
    """Move each foot's full saved wrench to (0,0,ground_z), then sum both."""
    if not np.isfinite(ground_z) or not np.isfinite(min_fz) or min_fz < 0:
        raise ValueError("ground_z must be finite and min_fz nonnegative and finite")
    force = _array(force_world, (2, 3), "foot force")
    moment = _array(moment_about_com_world, (2, 3), "foot moment about body COM")
    position = _array(body_com_world, (2, 3), "foot body COM")
    at_ground = moment + np.cross(position - [0.0, 0.0, ground_z], force)
    total_force, total_moment = force.sum(axis=0), at_ground.sum(axis=0)

    def xy(f, m):
        if f[2] <= min_fz:
            return None
        return [float(-m[1] / f[2]), float(m[0] / f[2])]

    return {
        "cop_xy_m": xy(total_force, total_moment),
        "foot_cop_xy_m": [xy(f, m) for f, m in zip(force, at_ground, strict=True)],
        "foot_vertical_n": force[:, 2].tolist(),
        "total_force_world_n": total_force.tolist(),
        "total_moment_about_ground_origin_nm": total_moment.tolist(),
    }


def forward_onset(times_s, x_m, *, displacement_m=DISTANCE_M, hold_s=HOLD_S):
    """First sampled threshold crossing held continuously; missing samples break the hold."""
    if (
        not np.isfinite(displacement_m)
        or not np.isfinite(hold_s)
        or displacement_m <= 0
        or hold_s < 0
    ):
        raise ValueError("invalid forward onset thresholds")
    result = origin = start = previous = None
    for t, x in zip(times_s, x_m, strict=True):
        if not np.isfinite(t) or (previous is not None and t <= previous):
            raise ValueError("sample times must be finite and strictly increasing")
        previous = t
        if x is None:
            start = None
            continue
        if not np.isfinite(x):
            raise ValueError("position must be finite or missing")
        if origin is None:
            origin = x
        if x - origin < displacement_m:
            start = None
        else:
            if start is None:
                start = t
            if result is None and t - start >= hold_s - 1e-12:
                result = float(start)
    return result


class _PoseObserver:
    def __init__(self, model):
        self.model = model
        self.data = mj.MjData(model)
        self.feet = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name) for name in FEET]
        self.pelvis = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
        self.floor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
        if min(self.feet + [self.pelvis, self.floor]) < 0:
            raise ValueError("recording model is missing a foot, pelvis or floor")
        if (
            model.geom_bodyid[self.floor] != 0
            or model.geom_type[self.floor] != mj.mjtGeom.mjGEOM_PLANE
        ):
            raise ValueError("the recorded floor must be a static plane")
        self.mass = np.asarray(model.body_mass, dtype=float)
        self.total_mass = float(self.mass.sum())
        if not np.isfinite(self.mass).all() or np.any(self.mass < 0) or self.total_mass <= 0:
            raise ValueError("invalid model body masses")

    def observe(self, q, load):
        data = self.data
        data.qpos[:] = _array(q, (self.model.nq,), "recorded qpos")
        mj.mj_kinematics(self.model, data)
        if not np.allclose(
            data.geom_xmat[self.floor].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-9, rtol=0
        ):
            raise ValueError("CoP reconstruction requires a horizontal upward floor")
        com = self.mass @ data.xipos / self.total_mass
        result = {"root_xyz_m": data.xpos[self.pelvis].tolist(), "com_xyz_m": com.tolist()}
        if load is None:
            result.update(
                cop_xy_m=None,
                foot_cop_xy_m=None,
                foot_vertical_n=None,
                total_force_world_n=None,
                total_moment_about_ground_origin_nm=None,
                com_minus_cop_xy_m=None,
            )
        else:
            support = ground_cop(
                load["foot_force_world"],
                load["foot_moment_about_body_com_world"],
                data.xipos[self.feet],
                ground_z=float(data.geom_xpos[self.floor, 2]),
            )
            cop = support["cop_xy_m"]
            result.update(
                support, com_minus_cop_xy_m=None if cop is None else (com[:2] - cop).tolist()
            )
        return result


def _difference(a, b):
    result = {}
    for field in (
        "root_xyz_m",
        "com_xyz_m",
        "cop_xy_m",
        "com_minus_cop_xy_m",
        "total_force_world_n",
    ):
        result[field] = (
            None
            if a[field] is None or b[field] is None
            else np.subtract(a[field], b[field]).tolist()
        )
    return result


def analyze_run(model, journal, *, model_sha256):
    journal = Path(journal)
    observer = _PoseObserver(model)
    rows, prior, h, saved_summary = [], None, None, None
    controller = balance_law = initial_state = None
    next_tick = 1

    def append(poses, loads):
        tick = len(rows)
        sample = {
            "pose_tick": tick,
            "time_s": tick * h,
            "load_from_step": None if loads is None else tick + 1,
        }
        for side in SIDES:
            sample[side] = observer.observe(poses[side], None if loads is None else loads[side])
            initial = rows[0][side] if rows else sample[side]
            for name in ("root", "com"):
                field = f"{name}_xyz_m"
                delta = np.subtract(sample[side][field], initial[field])
                sample[side][f"{name}_displacement_xyz_m"] = delta.tolist()
                sample[side][f"{name}_displacement_xy_norm_m"] = float(np.linalg.norm(delta[:2]))
        sample["gpu_minus_cpu"] = _difference(sample["gpu"], sample["cpu"])
        rows.append(sample)

    opener = gzip.open if journal.suffix == ".gz" else open
    with opener(journal, "rt", encoding="utf8") as stream:
        for sequence, line in enumerate(stream):
            record = json.loads(line)
            if record["sequence"] != sequence or saved_summary is not None:
                raise ValueError(f"{journal} has an out-of-order record at line {sequence + 1}")
            event = record["event"]
            if event == "sources":
                if (
                    h is not None
                    or record["inputs"]["g1_primitive_actuated.xml"]["sha256"] != model_sha256
                ):
                    raise ValueError(
                        "duplicate sources or journal XML digest differs from the supplied model"
                    )
                h = float(record["step_s"])
                if not math.isfinite(h) or h <= 0:
                    raise ValueError("invalid time step")
                # Original paired standing journals predate the controller field.
                controller = record.get("controller", "pd")
            elif event == "balance_law":
                if (
                    h is None
                    or prior is not None
                    or balance_law is not None
                    or controller != "balance"
                ):
                    raise ValueError(
                        "expected a single balance_law after balance sources and before initial"
                    )
                balance_law = {k: v for k, v in record.items() if k not in ("sequence", "event")}
            elif event == "initial":
                if h is None or prior is not None:
                    raise ValueError("unexpected initial record")
                if controller == "balance" and balance_law is None:
                    raise ValueError("balance initial state is missing its balance_law")
                initial_state = {
                    "q": _array(record["q"], (model.nq,), "initial qpos").tolist(),
                    "v": _array(record["v"], (model.nv,), "initial qvel").tolist(),
                }
                prior = dict.fromkeys(SIDES, initial_state["q"])
            elif event == "step":
                if prior is None or record["tick"] != next_tick:
                    raise ValueError(f"expected continuous paired step {next_tick}")
                append(prior, {side: record[f"{side}_load"] for side in SIDES})
                prior = {side: record[f"{side}_q"] for side in SIDES}
                next_tick += 1
            elif event == "summary":
                saved_summary = {k: v for k, v in record.items() if k not in ("sequence", "event")}
            else:
                raise ValueError(f"unsupported standing journal event {event!r}")
    if prior is not None:
        # Final endpoint has a pose but no subsequent incoming-contact record.
        append(prior, None)
    index = {round(r["time_s"], 12): r for r in rows}
    times = [r["time_s"] for r in rows]
    onsets = {}
    for side in SIDES:
        onsets[side] = {}
        for field in ("com_xyz_m", "root_xyz_m", "cop_xy_m"):
            x = [None if r[side][field] is None else r[side][field][0] for r in rows]
            first = next(
                (
                    (t, position)
                    for t, position in zip(times, x, strict=True)
                    if position is not None
                ),
                None,
            )
            onsets[side][field] = {
                "reference_time_s": None if first is None else first[0],
                "reference_x_m": None if first is None else first[1],
                "onset_s": forward_onset(times, x),
            }
    return {
        "journal": str(journal),
        "source_status": saved_summary.get("status") if saved_summary else "INCOMPLETE",
        "source_summary": saved_summary,
        "recorded_steps": next_tick - 1,
        "step_s": h,
        "controller": controller,
        "balance_law": balance_law,
        "initial_state": initial_state,
        "model_sha256": model_sha256,
        "model_mass_kg": observer.total_mass,
        "forward_onset": onsets,
        "milestones": {f"{t:.3f}": index.get(t) for t in MARKS},
        "trajectory": rows,
    }


def compare_runs(baseline, candidate):
    if baseline["model_sha256"] != candidate["model_sha256"]:
        raise ValueError("cannot compare journals from different models")
    if baseline["step_s"] is None or baseline["step_s"] != candidate["step_s"]:
        raise ValueError("cannot compare journals with missing or different step sizes")
    if baseline["initial_state"] is None or candidate["initial_state"] is None:
        raise ValueError("cannot compare journals without both initial states")
    for field in ("q", "v"):
        if not np.array_equal(baseline["initial_state"][field], candidate["initial_state"][field]):
            raise ValueError(f"cannot compare journals with different initial {field}")
    first = {round(r["time_s"], 12): r for r in baseline["trajectory"]}
    result = []
    for row in candidate["trajectory"]:
        match = first.get(round(row["time_s"], 12))
        if match is not None:
            result.append(
                {"time_s": row["time_s"], **{s: _difference(row[s], match[s]) for s in SIDES}}
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="exact XML named in each journal's sources"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model_sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
    model = mj.MjModel.from_xml_path(str(args.model.resolve()))
    result = {
        "schema": "g1-standing-com-cop-v1",
        "model_sha256": model_sha,
        "time_basis": "Nominal tick times the recorded timestep. Loads for step k use pose k-1.",
        "foot_order": list(FEET),
        "min_vertical_force_n": MIN_FZ_N,
        "cop_scope": "Both feet's ground wrenches. Non-foot contact excluded; unavailable CoP is null.",
        "forward_definition": {
            "axis": "+world X",
            "displacement_m": DISTANCE_M,
            "hold_s": HOLD_S,
            "reference": "First valid sample per signal; missing samples break the hold.",
        },
        "baseline": analyze_run(model, args.baseline, model_sha256=model_sha),
    }
    if args.candidate:
        result["candidate"] = analyze_run(model, args.candidate, model_sha256=model_sha)
        result["candidate_minus_baseline"] = compare_runs(result["baseline"], result["candidate"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, allow_nan=False, separators=(",", ":"))


if __name__ == "__main__":
    main()
