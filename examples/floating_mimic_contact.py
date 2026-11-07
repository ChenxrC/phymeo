"""Complete HOST floating-mimic contact window through the public scene API.

Frozen passive F03 input: actual entry, elastic loading and possible rebound.
All accepted frames are retained, including full physical/reduced state and
material loads. A failed or unqualified window is saved and exits nonzero.
"""

from argparse import ArgumentParser
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from phymeo.builders.floating_mimic_urdf import load_floating_mimic_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.dynamics.floating_robot import FloatingRobotState, floating_kinematics
from phymeo.runtime._floating_robot_participant import FloatingRobotCommand
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand
from phymeo.solvers.floating_tree import FloatingRobotInputs, floating_dynamics_terms


FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "urdf"
WORLD, ROBOT, BALL = 23, 17, 91
DURATION = 1 / 4
CYCLE = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
MASSES = np.array([2.0, 1.0, 2.0])
INERTIA = np.array([[1 / 5, 3 / 10, 2 / 5], [1 / 16, 5 / 64, 3 / 32], [3 / 32, 7 / 64, 1 / 8]])


def plain(value):
    """Lossless public fields; non-string mapping keys retain their actual type data."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: plain(item) for key, item in value.items()}
        return [{"key": plain(key), "value": plain(item)} for key, item in value.items()]
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [plain(item) for item in sorted(value, key=repr)]
    if isinstance(value, Enum):
        return plain(value.value)
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    raise TypeError(f"unsupported public record type: {type(value).__name__}")


def initial_state(*, transformed=False):
    rotation = CYCLE if transformed else np.eye(3)
    shift = np.array([1 / 2, -1 / 4, 3 / 4]) if transformed else np.zeros(3)
    quaternion = (0.5,) * 4 if transformed else (0, 0, 0, 1)
    return FloatingRobotState(
        np.array([*shift, *quaternion, 0.0]),
        np.r_[rotation @ [0, 1 / 4, 0], np.zeros(4)],
    )


def build_contact_scene(*, transformed=False, reverse=False, fixture_dir=FIXTURES):
    """Original declared layer-outside input, with no target pose writes later."""
    rotation = CYCLE if transformed else np.eye(3)
    shift = np.array([1 / 2, -1 / 4, 3 / 4]) if transformed else np.zeros(3)
    builder = ContactSceneBuilder(gravity=(0, 0, 0))

    def robot():
        builder.add_floating_urdf(
            ROBOT, Path(fixture_dir) / "floating_mimic_jaws.urdf",
            robot_state=initial_state(transformed=transformed), kp=[0], kd=[0],
            effort_limit=[1.5], loader=load_floating_mimic_urdf, world=WORLD,
        )

    def ball():
        builder.add_sphere(
            BALL, 1 / 16, mass=1,
            position=shift + rotation @ np.array([0.254, 1 / 8 + 1 / 1024, 0]),
            world=WORLD,
        )

    for add in ((ball, robot) if reverse else (robot, ball)):
        add()
    builder.pair(ROBOT, BALL, RobotContactMaterial(2048, 0, 0), world=WORLD)
    return builder.build(backend="host")


def zero_drive():
    return FloatingRobotCommand(RobotCommand([0.0], feedforward=[0.0]))


def robot_reading(frame):
    return next(item["value"] for item in frame["reading"]["robots"] if item["key"] == [WORLD, ROBOT])


def sites(frame):
    return [site for pair in frame["reading"]["loads"]["contacts"] for site in pair["sites"]]


def _totals(geometry, rigid, row):
    tensors = (geometry.rotation * INERTIA[:, None, :]) @ geometry.rotation.transpose(0, 2, 1)
    momentum = MASSES[:, None] * geometry.com_velocity
    spin = np.einsum("nij,nj->ni", tensors, geometry.angular_velocity)
    linear = momentum.sum(axis=0) + rigid.velocity[row]
    angular = np.cross(geometry.com_position, momentum).sum(axis=0) + spin.sum(axis=0)
    ball_inertia = (2 / 5) * (1 / 16) ** 2
    angular += np.cross(rigid.position[row], rigid.velocity[row]) + ball_inertia * rigid.angular_velocity[row]
    energy = np.sum(momentum * geometry.com_velocity) / 2 + np.sum(spin * geometry.angular_velocity) / 2
    energy += (rigid.velocity[row] @ rigid.velocity[row] + ball_inertia * (rigid.angular_velocity[row] @ rigid.angular_velocity[row])) / 2
    return linear, angular, float(energy)


def _frame(scene, reading, asset, ball_row, impulse, initial_totals):
    row = reading.robots[WORLD, ROBOT]
    entry = asset.full_state(FloatingRobotState(row.entry_q, row.entry_v))
    end = FloatingRobotState(row.full_q, row.full_v)
    entry_geo = floating_kinematics(asset.tree.model, entry)
    end_geo = floating_kinematics(asset.tree.model, end)
    terms = floating_dynamics_terms(asset.tree.model, entry, FloatingRobotInputs.zeros(2))
    totals = _totals(end_geo, reading.rigid, ball_row)
    residual = None if row.mimic_loads is None else (
        terms.mass_matrix @ row.acceleration_full + terms.bias
        - row.applied_effort_full - row.mimic_loads.constraint_effort_full
    )
    geometry = scene.geometry()
    return {
        "reading": plain(reading), "events": plain(scene.contact_events),
        "entry_full": plain(entry), "entry_geometry": plain(entry_geo),
        "end_geometry": plain(end_geo),
        "geometry": {"time_s": geometry.time_s, "poses": plain(geometry.poses)},
        "entry_mass": terms.mass_matrix.tolist(), "entry_bias": terms.bias.tolist(),
        "full_balance_residual": plain(residual),
        "ball_impulse_ns": impulse.tolist(),
        "linear_momentum_ns": totals[0].tolist(),
        "angular_momentum_nms": totals[1].tolist(),
        "linear_momentum_residual_ns": (totals[0] - initial_totals[0]).tolist(),
        "angular_momentum_residual_nms": (totals[1] - initial_totals[1]).tolist(),
        "kinetic_energy_j": totals[2],
    }


def run_contact_window(*, dt_s=1 / 2048, transformed=False, reverse=False, fixture_dir=FIXTURES):
    """Save every accepted state; failure retains the real prefix and exception."""
    if dt_s not in (1 / 2048, 1 / 4096):
        raise ValueError("frozen dt_s must be 1/2048 or 1/4096")
    fixture = Path(fixture_dir) / "floating_mimic_jaws.urdf"
    protocol = {
        "backend": "host", "dt_s": dt_s, "duration_s": DURATION,
        "world": WORLD, "robot": ROBOT, "ball": BALL,
        "transformed": transformed, "reverse_registration": reverse,
        "gravity": [0, 0, 0], "external": "zero", "drive": "explicit zero",
        "law": {"stiffness_n_per_m": 2048, "damping": 0, "friction": 0},
        "initial_gap_m": 1 / 1024,
        "fixture": str(fixture.resolve()), "fixture_sha256": sha256(fixture.read_bytes()).hexdigest(),
        "driver_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "HOST passive physical encounter; no grasp, state reset, CCD response or static-friction claim",
    }
    frames, failure = [], None
    initial = final = None
    last_returned = None
    try:
        with build_contact_scene(transformed=transformed, reverse=reverse, fixture_dir=fixture_dir) as scene:
            asset, ball_row = scene.assets[WORLD, ROBOT], scene.rigid_rows[WORLD, BALL]
            start = scene.observe()
            start_row = start.robots[WORLD, ROBOT]
            start_geo = floating_kinematics(asset.tree.model, FloatingRobotState(start_row.full_q, start_row.full_v))
            totals = _totals(start_geo, start.rigid, ball_row)
            initial = {"reading": plain(start), "geometry": plain(scene.geometry()), "events": plain(scene.contact_events), "totals": plain(totals)}
            impulse = np.zeros(3)
            try:
                for _ in range(round(DURATION / dt_s)):
                    last_returned = None
                    reading = scene.step(dt_s, actions={ROBOT: zero_drive()})
                    last_returned = reading
                    impulse += dt_s * reading.loads.inputs.rigid.force[ball_row]
                    frames.append(_frame(scene, reading, asset, ball_row, impulse, totals))
            except Exception as exc:
                failure = {"type": type(exc).__name__, "message": str(exc), "attempt": len(frames) + 1, "returned_reading": plain(last_returned)}
            final = {"reading": plain(scene.observe()), "geometry": plain(scene.geometry()), "events": plain(scene.contact_events)}
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "attempt": len(frames) + 1, "returned_reading": plain(last_returned)}
    loaded = [frame for frame in frames if any(site["force_n"] > 0 for site in sites(frame))]
    summary = {
        "status": "complete" if failure is None and len(frames) == round(DURATION / dt_s) else "failed",
        "steps": len(frames), "time_s": None if final is None else final["reading"]["time_s"],
        "first_load_time_s": None if not loaded else loaded[0]["reading"]["load_time_s"],
        "last_load_time_s": None if not loaded else loaded[-1]["reading"]["load_time_s"],
        "peak_force_n": max((sum(site["force_n"] for site in sites(frame)) for frame in frames), default=0.0),
        "has_begin": any(event["kind"] == "begin" for frame in frames for event in frame["events"]["events"]),
        "has_end": any(event["kind"] == "end" for frame in frames for event in frame["events"]["events"]),
        "final_unloaded": bool(frames) and not any(site["force_n"] > 0 for site in sites(frames[-1])),
        "max_linear_residual_ns": max((float(np.linalg.norm(frame["linear_momentum_residual_ns"])) for frame in frames), default=0.0),
        "max_angular_residual_nms": max((float(np.linalg.norm(frame["angular_momentum_residual_nms"])) for frame in frames), default=0.0),
    }
    summary["qualified"] = summary["status"] == "complete" and summary["has_begin"] and summary["has_end"] and summary["final_unloaded"]
    return {"protocol": protocol, "summary": summary, "failure": failure, "initial": initial, "frames": frames, "final": final}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=1 / 2048)
    parser.add_argument("--transformed", action="store_true")
    parser.add_argument("--reverse-registration", action="store_true")
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.is_absolute() or args.out.exists():
        parser.error("--out must be a new absolute .json or .json.gz file")
    result = run_contact_window(dt_s=args.dt, transformed=args.transformed, reverse=args.reverse_registration, fixture_dir=args.fixture_dir)
    payload = (json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        if args.out.suffix == ".gz":
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                compressed.write(payload)
        else:
            stream.write(payload)
    print(json.dumps(result["summary"], allow_nan=False))
    return 0 if result["summary"]["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
