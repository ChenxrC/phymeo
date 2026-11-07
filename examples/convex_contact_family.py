"""Three convex robot families share one physical contact/owner pipeline.

Each case starts separated with real approach velocities, compresses a compliant
layer, and separates again. Two real URDF sliders retain their masses/inertias.
A public loader supplies explicitly declared convex collision vertices; the
solver does not receive a case name or a prescribed state trajectory.
"""

import argparse
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from phymeo.builders.urdf import load_urdf
from phymeo.constraints.collision_shapes import CollisionModel
from phymeo.constraints.robot_contact import RobotContactMaterial
from phymeo.core.robot import RobotState
from phymeo.runtime.contact_scene import ContactSceneBuilder
from phymeo.runtime.robot_scene import RobotCommand

CASES = (("capsule", "capsule"), ("capsule", "box"), ("hull", "hull"))


def collision_vertices(kind):
    if kind == "capsule":
        return np.array([[-0.5, 0, 0], [0.5, 0, 0]]), 0.05
    if kind == "box":
        return np.array(list(product((-0.6, 0.6), (-0.1, 0.1), (-0.1, 0.1)))), 0.0
    if kind == "hull":
        ring = ((-0.6, 0), (-0.4, -0.1), (0.4, -0.1), (0.6, 0), (0.4, 0.1), (-0.4, 0.1))
        return np.array([(x, y, z) for x, y in ring for z in (-0.1, 0.1)]), 0.0
    raise ValueError("this example declares capsule, box or hexagonal prism")


def make_scene(directory, kinds):
    builder = ContactSceneBuilder(gravity=(0, 0, 0))
    height = 0.02 + sum(0.05 if kind == "capsule" else 0.1 for kind in kinds)
    actions = {}
    for i, kind in enumerate(kinds):
        body, mass, z, speed = (10, 20)[i], (2, 3)[i], (0, height)[i], (0.08, -0.12)[i]
        path = Path(directory) / f"slider-{body}.urdf"
        path.write_text(
            f'''<robot name="convex_slider"><link name="base"/>
<link name="tool"><inertial><mass value="{mass}"/>
<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
<collision name="tool"><geometry><sphere radius=".05"/></geometry></collision></link>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<origin xyz="0 0 {z}"/><axis xyz="0 0 1"/>
<limit lower="-1" upper="1" effort="128" velocity="10"/></joint></robot>''',
            encoding="utf-8",
        )
        asset = load_urdf(path, gravity=(0, 0, 0))
        original = asset.collisions[0]
        vertices, radius = collision_vertices(kind)
        shape = replace(
            original.shape,
            kind="capsule" if kind == "capsule" else "hull",
            vertex_ids=tuple(range(len(vertices))),
            radius=radius,
        )
        collider = replace(original, shape=shape, local_vertices=tuple(map(tuple, vertices)))
        asset = replace(asset, collisions=(collider,), collision_model=CollisionModel((shape,)))

        def loader(path, *, gravity, asset=asset):
            return asset

        builder.add_urdf(
            body,
            path,
            robot_state=RobotState(np.zeros(1), np.array([speed])),
            kp=[0],
            kd=[0],
            loader=loader,
        )
        actions[body] = RobotCommand([0])
    builder.pair(10, 20, RobotContactMaterial(100, 0, 0))
    return builder.build(backend="host"), actions


def run_case(directory, kinds, dt):
    rows = []
    oracle_q, oracle_v = np.zeros(2), np.array([0.08, -0.12])
    mass = np.array([2.0, 3.0])
    maxima = dict(
        oracle_position_m=0.0,
        oracle_velocity_m_s=0.0,
        oracle_force_N=0.0,
        momentum_kg_m_s=0.0,
        power_W=0.0,
    )
    scene, actions = make_scene(directory, kinds)
    with scene:
        for index in range(round(0.5 / dt)):
            gap = 0.02 + oracle_q[1] - oracle_q[0]
            expected_force = 100 * max(0.0, -gap)
            oracle_v += dt * np.array([-expected_force, expected_force]) / mass
            oracle_q += dt * oracle_v
            out = scene.step(dt, actions=actions)
            q = np.array([out.robots[0, body].q[0] for body in (10, 20)])
            v = np.array([out.robots[0, body].v[0] for body in (10, 20)])
            force = sum(pair.force_n for pair in out.loads.contacts)
            generalized_power = sum(
                float(row.contact_effort_full @ row.entry_v) for row in out.robots.values()
            )
            site_power = sum(site.power_w for pair in out.loads.contacts for site in pair.sites)
            maxima["oracle_position_m"] = max(
                maxima["oracle_position_m"], float(abs(q - oracle_q).max())
            )
            maxima["oracle_velocity_m_s"] = max(
                maxima["oracle_velocity_m_s"], float(abs(v - oracle_v).max())
            )
            maxima["oracle_force_N"] = max(maxima["oracle_force_N"], abs(force - expected_force))
            maxima["momentum_kg_m_s"] = max(
                maxima["momentum_kg_m_s"], abs(float(mass @ (v - np.array([0.08, -0.12]))))
            )
            maxima["power_W"] = max(maxima["power_W"], abs(generalized_power - site_power))
            rows.append(
                dict(
                    step=index + 1,
                    time_s=out.time_s,
                    entry_gap_m=gap,
                    force_N=force,
                    q_m=q.tolist(),
                    v_m_s=v.tolist(),
                    events=[event.kind for event in scene.contact_events.events],
                )
            )
    forces = [row["force_N"] for row in rows]
    checks = dict(
        completed=rows[-1]["time_s"] == 0.5,
        initial_separation=forces[0] == 0,
        loaded=max(forces) > 1,
        released=all(f == 0 for f in forces[-4:]),
        birth=any("begin" in row["events"] for row in rows),
        end=any("end" in row["events"] for row in rows),
        mechanics=all(value < 1e-9 for value in maxima.values()),
    )
    return dict(
        kinds=kinds,
        dt_s=dt,
        final_q_m=rows[-1]["q_m"],
        impulse_Ns=dt * sum(forces),
        checks=checks,
        maxima=maxima,
        trace=rows,
    )


def run(directory):
    cases = []
    for index, kinds in enumerate(CASES):
        folder = Path(directory) / str(index)
        folder.mkdir()
        coarse, fine = (run_case(folder, kinds, dt) for dt in (1 / 256, 1 / 512))
        translation = max(
            abs(a - b) for a, b in zip(coarse["final_q_m"], fine["final_q_m"], strict=True)
        )
        impulse_relative = abs(coarse["impulse_Ns"] - fine["impulse_Ns"]) / fine["impulse_Ns"]
        cases.append(
            dict(
                coarse=coarse,
                fine=fine,
                refinement=dict(
                    translation_m=translation,
                    rotation_rad=0.0,
                    impulse_relative=impulse_relative,
                    passed=translation <= 1e-3 and impulse_relative <= 0.05,
                ),
            )
        )
    return dict(
        scope="HOST instantaneous convex owner loads, real 0.5 s windows; not CCD",
        law=dict(stiffness_N_m=100, damping_Ns_m=0, friction=0),
        passed=all(all(c[rate]["checks"].values()) for c in cases for rate in ("coarse", "fine"))
        and all(c["refinement"]["passed"] for c in cases),
        cases=cases,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with TemporaryDirectory(prefix="phymeo-convex-family-") as directory:
        result = run(directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                passed=result["passed"],
                output=str(args.output),
                refinement=[c["refinement"] for c in result["cases"]],
            )
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
