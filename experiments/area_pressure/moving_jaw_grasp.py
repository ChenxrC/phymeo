"""Opposed motorized slides grasp a free box on MuJoCo-TYX CUDA.

Run python -m experiments.area_pressure.moving_jaw_grasp --output grasp.json.
The two worlds differ only in jaw friction. Gravity is disabled so an external
load can be switched on after closure and held through release.
"""

import argparse
import json
from pathlib import Path

from experiments.area_pressure.heterogeneous_support import box_pressure_volume
from phymeo.backends.mujoco_area_friction import AreaFrictionPairLaw
from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.control.body_contact_wrench import BodyContactWrenchReader, BodyContactWrenchRequest
from phymeo.geometry.area_pressure import AreaPressureIdentity
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh

H = 0.004
JAWS = ("left", "right")
MATERIALS = ((41, "grip", 0.6), (7, "smooth", 0.0))
# phase, steps, inward motor force on each jaw (N), world Z force on box (N)
PHASES = (("close", 15, 0.45, 0.0), ("load", 8, 0.45, -0.22), ("open", 12, -0.9, -0.22))


def _declarations():
    jaws = []
    motors = []
    assets = []
    volume = box_pressure_volume(200000.0)
    for name, sign in zip(JAWS, (1, -1), strict=True):
        jaws.append(f'''<body name="{name}" pos="{-sign * 0.02225} 0 0">
            <joint name="{name}-slide" type="slide" axis="{sign} 0 0" damping="8"/>
            <geom type="box" size=".002 .006 .003" mass=".05"/>
        </body>''')
        motors.append(f'<motor name="{name}-close" joint="{name}-slide"/>')
        points = tuple(
            (sign * 0.002, sign * y, z)
            for y, z in ((-0.006, -0.003), (0.006, -0.003), (0.006, 0.003), (-0.006, 0.003))
        )
        face = RigidPressureMesh(
            ("a", "b", "c", "d"), points, ((0, 1, 2), (0, 2, 3)), ("lower", "upper")
        )
        identity = AreaPressureIdentity("object", name, "box-volume", name + "-inner-face")
        assets.append(BodyAreaPressureAsset(identity, volume, face))
    xml = f'''<mujoco model="movable-jaws">
        <compiler angle="radian" fusestatic="false"/>
        <option timestep="{H}" gravity="0 0 0" integrator="Euler" solver="Newton"
                iterations="128" tolerance="1e-10" cone="pyramidal" jacobian="dense"/>
        <default><geom contype="0" conaffinity="0"/></default>
        <worldbody><body name="object"><freejoint/>
            <geom type="box" size=".02 .02 .01" mass=".3"/>
        </body>{"".join(jaws)}</worldbody><actuator>{"".join(motors)}</actuator>
    </mujoco>'''
    compilation = compile_mujoco_xml(xml, require_asset=False)
    initial = MuJoCoInitialState((0, 0, 0, 1, 0, 0, 0, 0, 0), (0,) * 8)
    specs = []
    worlds = []
    for key, model, mu in MATERIALS:
        specs.append(
            MuJoCoTyxModelSpec(
                model,
                compilation,
                contact_capacity=16,
                constraint_capacity=128,
                area_pressure_pairs=tuple(("object", jaw) for jaw in JAWS),
                area_friction_laws=tuple(
                    AreaFrictionPairLaw(a.identity, mu, 0.001) for a in assets
                ),
            )
        )
        worlds.append(MuJoCoTyxWorldSpec(key, model, initial))
    return tuple(specs), tuple(worlds), {s.key: tuple(assets) for s in specs}


def run():
    """Run an uninterrupted clamp/load/open sequence and capture accepted steps."""
    specs, worlds, assets = _declarations()
    records = {model: [] for _, model, _ in MATERIALS}
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
        with batch.prepare_device_workspace(
            include_bodies=True, include_contacts=True, include_area_pressure=True
        ) as ws:
            reader = BodyContactWrenchReader(
                ws, area_identities={k: tuple(a.identity for a in v) for k, v in assets.items()}
            )
            for phase, count, drive, load in PHASES:
                control = MuJoCoControl((drive, drive), qfrc_applied=(0, 0, load, 0, 0, 0, 0, 0))
                batch.set_controls(dict.fromkeys(batch.world_keys, control))
                for _ in range(count):
                    before = batch.integration_snapshot()
                    accepted = scene.step(H, expected_revision=batch.checkpoint_revision)
                    if accepted != dict.fromkeys(batch.world_keys, True):
                        raise RuntimeError(f"CUDA step rejected in {phase}: {accepted}")
                    frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                    requests = tuple(
                        BodyContactWrenchRequest(key, name, before[key].q[:3], group)
                        for key, _, _ in MATERIALS
                        for name, group in (
                            ("object", JAWS),
                            ("left", ("object",)),
                            ("right", ("object",)),
                        )
                    )
                    measurements = iter(reader.measure_many(frame, requests))
                    for key, model, _mu in MATERIALS:
                        start, end = before[key], batch.last_step.states[key]
                        wrenches = {}
                        for name in ("object", *JAWS):
                            reading = next(measurements)
                            wrenches[name] = dict(
                                force=reading.total_force_world,
                                moment=reading.total_moment_world,
                                normal=reading.group_normal_loads,
                                samples=len(reading.contacts),
                            )
                        records[model].append(
                            dict(
                                tick=end.tick,
                                phase=phase,
                                drive_n=drive,
                                load_z_n=load,
                                input_q=start.q.tolist(),
                                input_v=start.v.tolist(),
                                q=end.q.tolist(),
                                v=end.v.tolist(),
                                time_s=end.time,
                                q_contact=end.last_step.qfrc_constraint.tolist(),
                                native_rows=len(end.last_step.constraints),
                                wrenches=wrenches,
                            )
                        )
    return dict(h_s=H, friction={m: mu for _, m, mu in MATERIALS}, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new JSON trace file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = run()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)
    for name, rows in result["records"].items():
        loaded = [r for r in rows if r["phase"] == "load"]
        slip_mm = 1000 * (loaded[0]["input_q"][2] - loaded[-1]["q"][2])
        print(f"{name}: downward displacement during load = {slip_mm:.4f} mm")


if __name__ == "__main__":
    main()
