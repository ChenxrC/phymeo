"""Real free-root fixed-tool contact through the public scene API (HOST).

Run from the repository with the installed Phymeo package or PYTHONPATH=src.
The probe URDF is a declared input asset; no target state is written after build.
The probe performs a passive elastic encounter and separates, not a grasp.
"""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from phymeo.builders.floating_urdf import load_floating_urdf
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.dynamics.floating_robot import FloatingRobotState, floating_kinematics
from phymeo import FloatingRobotCommand
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand


FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "urdf"
WORLD, ROBOT, BALL = 7, 101, 205
DURATION = 1 / 4
CYCLE = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])


def load_tools(path, *, gravity):
    return load_floating_urdf(path, gravity=gravity, collapse_fixed_joints=True)


def build_contact_scene(*, transformed=False, reverse=False, fixture_dir=FIXTURES, gap_m=1 / 1024):
    """Return an ordinary public multi-body scene from fixed physical declarations."""
    rotation = CYCLE if transformed else np.eye(3)
    shift = np.array([1 / 2, -1 / 4, 3 / 4]) if transformed else np.zeros(3)
    quaternion = (0.5, 0.5, 0.5, 0.5) if transformed else (0, 0, 0, 1)
    state = FloatingRobotState(
        np.array((*shift, *quaternion, 0, 1 / 8), dtype=float),
        np.r_[rotation @ [0, 1 / 4, 0], np.zeros(5)],
    )
    builder = ContactSceneBuilder(gravity=(0, 0, 0))

    def robot():
        builder.add_floating_urdf(
            ROBOT, Path(fixture_dir) / "floating_fixed_probe.urdf",
            robot_state=state, kp=[0, 0], kd=[0, 0], effort_limit=[16, 32],
            loader=load_tools, world=WORLD,
        )

    def ball():
        builder.add_sphere(
            BALL, 1 / 16, mass=1,
            position=shift + rotation @ np.array([1, 1 / 8 + gap_m, 0]),
            world=WORLD,
        )

    for add in ((ball, robot) if reverse else (robot, ball)):
        add()
    builder.pair(ROBOT, BALL, RobotContactMaterial(2048, 0, 0), world=WORLD)
    return builder.build(backend="host")


def zero_drive():
    # Explicit zeros avoid the public controller's optional bias feedforward.
    return FloatingRobotCommand(RobotCommand([0, 1 / 8], feedforward=[0, 0]))


def _key(value):
    return [value.world, value.body, value.shape]


def _site_rows(reading):
    rows = []
    for pair in reading.loads.contacts:
        for site in pair.sites:
            rows.append({
                "first": _key(pair.first), "second": _key(pair.second),
                "normal": list(pair.normal), "gap_m": pair.gap_m,
                "fraction": site.fraction, "force_n": site.force_n,
                "force_on_second_n": list(site.force_on_second_n),
                "application_first_m": list(site.application_point_first_m),
                "application_second_m": list(site.application_point_second_m),
                "material_first_m": list(site.material.first_point),
                "material_second_m": list(site.material.second_point),
                "first_weights": list(site.material.first_weights),
                "second_weights": list(site.material.second_weights),
                "rate_m_per_s": site.rate_m_per_s, "power_w": site.power_w,
            })
    return rows



def _physical_totals(masses, diagonal, geometry, rigid, row):
    """Measured end-state totals; these are diagnostics, not an integration oracle."""
    tensors = (geometry.rotation * diagonal[:, None, :]) @ geometry.rotation.transpose(0, 2, 1)
    body_p = masses[:, None] * geometry.com_velocity
    spins = np.einsum("nij,nj->ni", tensors, geometry.angular_velocity)
    linear = body_p.sum(axis=0) + rigid.velocity[row]
    angular = np.cross(geometry.com_position, body_p).sum(axis=0) + spins.sum(axis=0)
    ball_inertia = (2 / 5) * (1 / 16) ** 2
    angular += np.cross(rigid.position[row], rigid.velocity[row])
    angular += ball_inertia * rigid.angular_velocity[row]
    kinetic = 0.5 * np.sum(body_p * geometry.com_velocity)
    kinetic += 0.5 * np.sum(spins * geometry.angular_velocity)
    kinetic += 0.5 * np.dot(rigid.velocity[row], rigid.velocity[row])
    kinetic += 0.5 * ball_inertia * np.dot(rigid.angular_velocity[row], rigid.angular_velocity[row])
    return linear, angular, float(kinetic)

def run_contact_window(*, dt_s=1 / 2048, transformed=False, reverse=False, fixture_dir=FIXTURES):
    """Return complete public per-step state/loads for the fixed quarter-second input."""
    if dt_s not in (1 / 2048, 1 / 4096):
        raise ValueError("this frozen example uses dt_s=1/2048 or 1/4096")
    frames = []
    with build_contact_scene(
        transformed=transformed, reverse=reverse, fixture_dir=fixture_dir
    ) as scene:
        asset = scene.assets[WORLD, ROBOT]
        ball_row = scene.rigid_rows[WORLD, BALL]
        initial = scene.observe()
        masses = np.r_[1 / asset.model.root.inverse_mass, asset.model.children.mass]
        initial_robot = initial.robots[WORLD, ROBOT]
        initial_geometry = floating_kinematics(
            asset.model, FloatingRobotState(initial_robot.q, initial_robot.v)
        )
        diagonal = np.vstack(
            (asset.model.root.inertia_diagonal, asset.model.children.inertia_diagonal)
        )
        initial_momentum, initial_angular, initial_energy = _physical_totals(
            masses, diagonal, initial_geometry, initial.rigid, ball_row
        )
        initial_gap = None
        # The geometric initial side gap is an input; no candidate label claims a force.
        accumulated_impulse = np.zeros(3)
        max_linear_residual = 0.0
        for _ in range(round(DURATION / dt_s)):
            reading = scene.step(dt_s, actions={ROBOT: zero_drive()})
            row = reading.robots[WORLD, ROBOT]
            geometry = floating_kinematics(
                asset.model, FloatingRobotState(row.q, row.v)
            )
            entry_geometry = floating_kinematics(
                asset.model, FloatingRobotState(row.entry_q, row.entry_v)
            )
            force = np.asarray(reading.loads.inputs.rigid.force[ball_row])
            torque = np.asarray(reading.loads.inputs.rigid.torque[ball_row])
            accumulated_impulse += dt_s * force
            momentum, angular, kinetic = _physical_totals(
                masses, diagonal, geometry, reading.rigid, ball_row
            )
            residual = momentum - initial_momentum
            angular_residual = angular - initial_angular
            max_linear_residual = max(max_linear_residual, float(np.linalg.norm(residual)))
            sites = _site_rows(reading)
            if sites and initial_gap is None:
                initial_gap = min(site["gap_m"] for site in sites)
            events = scene.contact_events
            materials = []
            for record in events.active:
                for component in record.components:
                    for item in component.material:
                        materials.append({
                            "first_kind": record.first_owner.kind,
                            "second_kind": record.second_owner.kind,
                            "first_owner": [record.first_owner.world, record.first_owner.body, record.first_owner.link_index],
                            "first_local_m": item.first.local_point_m,
                            "second_owner": [record.second_owner.world, record.second_owner.body, record.second_owner.link_index],
                            "second_local_m": item.second.local_point_m,
                        })
            frames.append({
                "step": reading.step_number,
                "time_s": reading.time_s, "load_time_s": reading.load_time_s,
                "q": row.q.tolist(), "v": row.v.tolist(),
                "entry_q": row.entry_q.tolist(), "entry_v": row.entry_v.tolist(),
                "drive_effort": row.drive_effort.tolist(),
                "contact_effort_full": row.contact_effort_full.tolist(),
                "link_wrenches_world": row.link_wrenches_world.tolist(),
                "entry_body_com_m": entry_geometry.com_position.tolist(),
                "body_com_m": geometry.com_position.tolist(),
                "body_com_velocity_m_s": geometry.com_velocity.tolist(),
                "body_rotation_world": geometry.rotation.tolist(),
                "body_omega_rad_s": geometry.angular_velocity.tolist(),
                "entry_ball_position_m": reading.entry_rigid.position[ball_row].tolist(),
                "entry_ball_velocity_m_s": reading.entry_rigid.velocity[ball_row].tolist(),
                "entry_ball_omega_rad_s": reading.entry_rigid.angular_velocity[ball_row].tolist(),
                "ball_position_m": reading.rigid.position[ball_row].tolist(),
                "ball_velocity_m_s": reading.rigid.velocity[ball_row].tolist(),
                "ball_omega_rad_s": reading.rigid.angular_velocity[ball_row].tolist(),
                "ball_force_n": force.tolist(), "ball_torque_nm": torque.tolist(),
                "ball_impulse_ns": accumulated_impulse.tolist(),
                "linear_momentum_residual_ns": residual.tolist(),
                "angular_momentum_residual_nms": angular_residual.tolist(),
                "kinetic_energy_j": kinetic,
                "contact_potential_j": sum(pair.potential_j for pair in reading.loads.contacts),
                "sites": sites, "materials": materials,
                "events": [event.kind for event in events.events],
                "event_sample_time_s": events.sample_time_s,
                "event_commit_time_s": events.commit_time_s,
            })
        final = scene.observe()
        final_geometry = scene.geometry()
        tip_key = scene.shape_key(ROBOT, "tool", "tip_box")
        tip_points = final_geometry.poses[tip_key].start
        # Preserve actual final vertices; no final nearest-feature guess is a force.
        loaded = [frame for frame in frames if any(site["force_n"] > 0 for site in frame["sites"])]
        summary = {
            "steps": len(frames), "time_s": final.time_s,
            "first_positive_load_time_s": None if not loaded else loaded[0]["load_time_s"],
            "last_positive_load_time_s": None if not loaded else loaded[-1]["load_time_s"],
            "peak_force_n": max(
                (sum(site["force_n"] for site in frame["sites"]) for frame in frames), default=0
            ),
            "first_loaded_gap_m": initial_gap,
            "initial_momentum_ns": initial_momentum.tolist(),
            "initial_angular_momentum_nms": initial_angular.tolist(),
            "initial_kinetic_energy_j": initial_energy,
            "final_kinetic_energy_j": frames[-1]["kinetic_energy_j"],
            "final_angular_momentum_residual_nms": frames[-1]["angular_momentum_residual_nms"],
            "max_angular_momentum_residual_nms": max(float(np.linalg.norm(f["angular_momentum_residual_nms"])) for f in frames),
            "ball_impulse_ns": accumulated_impulse.tolist(),
            "max_linear_momentum_residual_ns": max_linear_residual,
            "final_tool_vertices_m": tip_points.tolist(),
            "has_begin": any("begin" in frame["events"] for frame in frames),
            "has_end": any("end" in frame["events"] for frame in frames),
            "final_unloaded": not any(site["force_n"] > 0 for site in frames[-1]["sites"]),
        }
    fixture = Path(fixture_dir) / "floating_fixed_probe.urdf"
    return {
        "protocol": {
            "backend": "host", "dt_s": dt_s, "duration_s": DURATION,
            "transformed": transformed, "reverse_registration": reverse,
            "world": WORLD, "robot": ROBOT, "ball": BALL,
            "gravity": [0, 0, 0], "external": "zero", "drive": "explicit zero",
            "law": {"stiffness_n_per_m": 2048, "damping": 0, "friction": 0},
            "initial_side_gap_m": 1 / 1024,
            "fixture": str(fixture.resolve()), "fixture_sha256": sha256(fixture.read_bytes()).hexdigest(),
            "scope": "passive elastic encounter; no grasp, pin, reset, or floating CCD claim",
        },
        "summary": summary, "frames": frames,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=1 / 2048)
    parser.add_argument("--transformed", action="store_true")
    parser.add_argument("--reverse-registration", action="store_true")
    parser.add_argument("--fixture-dir", type=Path, default=FIXTURES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.is_absolute() or args.out.exists():
        parser.error("--out must be a new absolute file")
    result = run_contact_window(
        dt_s=args.dt, transformed=args.transformed, reverse=args.reverse_registration,
        fixture_dir=args.fixture_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], allow_nan=False))
    summary = result["summary"]
    return 0 if summary["has_begin"] and summary["has_end"] and summary["final_unloaded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
