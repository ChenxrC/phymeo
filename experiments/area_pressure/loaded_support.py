"""Fixed loaded-support experiment through the reusable area-pressure scene API.

CPU MuJoCo compiles the model only. Current FK and dynamics run on TYX; mesh
intersection remains HOST. These one-DOF examples do not model a free balance
problem: the vertical guide carries transverse forces and moments.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime
from phymeo.backends.mujoco_area_pressure_scene import (
    BodyAreaPressureAsset,
    MuJoCoTyxAreaPressureScene,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.geometry.area_pressure import AreaPressureIdentity
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh, TetraPressureMesh


@dataclass(frozen=True)
class SupportPatch:
    name: str
    x_offset: float = 0.0
    x_scale: float = 1.0


SINGLE = (SupportPatch("center"),)
DUAL = (SupportPatch("left", -0.035, 0.5), SupportPatch("right", 0.035, 1.0))
STAGES = ((0.15, 0.0), (0.35, -3.0), (0.55, 9.0))


def build_support(mode, h, patches=SINGLE, *, initial_q=0.0, initial_v=0.0, capacity=64):
    """Build a vertical 0.5 kg body against separate fixed finite support faces."""
    if mode not in ("interpret", "cuda") or h not in (0.001, 0.0005):
        raise ValueError("this experiment uses the two registered modes and timesteps")
    if not patches or len({p.name for p in patches}) != len(patches):
        raise ValueError("support patches must have unique names")
    for patch in patches:
        if not patch.name.isidentifier() or not np.isfinite([patch.x_offset, patch.x_scale]).all():
            raise ValueError("invalid support patch")
        if patch.x_scale <= 0:
            raise ValueError("support area scale must be positive")
    fixed = "".join(
        f'<body name="{patch.name}"><geom type="sphere" size=".001" '
        'contype="0" conaffinity="0"/></body>'
        for patch in patches
    )
    xml = f'''<mujoco model="area-loaded-support">
    <compiler angle="radian" fusestatic="false"/>
    <option timestep="{h}" gravity="0 0 -9.81" integrator="Euler"
      solver="Newton" iterations="128" tolerance="1e-10" cone="pyramidal" jacobian="dense"/>
    <worldbody>
      <body name="pressure" pos="0 0 .002">
        <joint name="vertical" type="slide" axis="0 0 1" damping="40" armature="0"/>
        <inertial pos="0 0 0" mass=".5" diaginertia=".02 .02 .02"/>
        <geom type="sphere" size=".01" contype="0" conaffinity="0"/>
      </body>
      {fixed}
    </worldbody></mujoco>'''
    outer = np.array([[-0.12, -0.12, 0], [0.12, -0.12, 0], [0, 0.12, 0], [0, 0, 0.08]])
    pressure = TetraPressureMesh(
        ("a", "b", "c", "d", "center"),
        np.vstack((outer, outer.mean(axis=0))),
        ((0, 1, 2, 4), (0, 2, 3, 4), (0, 3, 1, 4), (1, 3, 2, 4)),
        (0, 0, 0, 0, 400000),
        ("bottom", "side-a", "side-b", "slant"),
    )
    pairs = tuple(("pressure", patch.name) for patch in patches)
    assets = []
    for patch, pair in zip(patches, pairs, strict=True):
        face = np.array([[-0.015, -0.045, 0], [0.015, -0.045, 0], [0, -0.015, 0]])
        face[:, 0] = face[:, 0] * patch.x_scale + patch.x_offset
        assets.append(
            BodyAreaPressureAsset(
                AreaPressureIdentity(*pair, "linear-pressure-full", "finite-rigid-face"),
                pressure,
                RigidPressureMesh(("r0", "r1", "r2"), face, ((0, 1, 2),), ("face",)),
            )
        )
    runtime = MuJoCoTyxRuntime(
        compile_mujoco_xml(xml, require_asset=False),
        mode=mode,
        world_keys=(13,),
        initial_states={13: MuJoCoInitialState([initial_q], [initial_v])},
        area_pressure_pairs=pairs,
        constraint_capacity=capacity,
    )
    return runtime, MuJoCoTyxAreaPressureScene(runtime, tuple(assets))


def _array(value):
    return np.asarray(value).tolist()


def record_step(before, after, geometry, stage):
    receipt = after.last_step
    patches = []
    for raw, generated in zip(receipt.raw_contact_fields["area_pressure"], geometry, strict=True):
        patches.append(
            dict(
                pair=[raw["identity"].compliant_body, raw["identity"].rigid_body],
                stamp=dict(
                    world=raw["stamp"].world_id,
                    tick=raw["stamp"].step_index,
                    time=raw["stamp"].sample_time_s,
                    configuration=raw["stamp"].configuration_id,
                ),
                polygons=generated.polygon_count,
                candidate_pairs=generated.candidate_pairs,
                sample_ids=raw["sample_ids"],
                source_ids=raw["source_ids"],
                **{
                    name: _array(raw[name])
                    for name in (
                        "body_ids",
                        "dA",
                        "points",
                        "normals",
                        "stiffness",
                        "surrogate_gap",
                        "normal_jacobian",
                        "normal_force",
                        "force_on_a",
                        "force_on_b",
                        "torque_on_a_world_origin",
                        "torque_on_b_world_origin",
                    )
                },
            )
        )
    dynamics = receipt.dynamics
    return dict(
        input_tick=before.tick,
        output_tick=after.tick,
        stage=stage,
        input_time=before.time,
        output_time=after.time,
        input_q=_array(before.q),
        input_v=_array(before.v),
        output_q=_array(after.q),
        output_v=_array(after.v),
        applied_generalized_force=_array(after.qfrc_applied),
        integration_acceleration=_array(receipt.integration_acceleration),
        generalized_contact_force=_array(receipt.qfrc_constraint),
        dynamics={
            name: _array(getattr(dynamics, name))
            for name in (
                "mass_matrix",
                "bias",
                "passive",
                "applied",
                "smooth_acceleration",
            )
        },
        patches=patches,
    )


def run_support(mode, h, patches=SINGLE):
    runtime, scene = build_support(mode, h, patches)
    records = []
    with runtime:
        stage = -1
        for step in range(round(STAGES[-1][0] / h)):
            selected = next(i for i, (end, _) in enumerate(STAGES) if step < round(end / h))
            if selected != stage:
                stage = selected
                runtime.set_controls({13: MuJoCoControl([], qfrc_applied=[STAGES[stage][1]])})
            before = runtime.integration_snapshot()[13]
            scene.step(h, expected_revision=runtime.checkpoint_revision)
            after = runtime.integration_snapshot()[13]
            records.append(record_step(before, after, scene.last_geometry[13], stage))
    return dict(
        schema="loaded-area-trajectory-v1",
        case="single" if len(patches) == 1 else "dual",
        mode=mode,
        h=h,
        pressure_domain="full",
        mass_kg=0.5,
        joint_damping_ns_m=40.0,
        gravity_m_s2=-9.81,
        body_origin_z_m=0.002,
        patch_reference_stiffness_n_m={p.name: 9000.0 * p.x_scale for p in patches},
        stages=[dict(end_s=end, force_n=force) for end, force in STAGES],
        patch_names=[patch.name for patch in patches],
        records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("interpret", "cuda"), default="cuda")
    parser.add_argument("--h", type=float, choices=(0.001, 0.0005), default=0.001)
    parser.add_argument("--case", choices=("single", "dual"), default="dual")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace a saved trajectory")
    payload = run_support(args.mode, args.h, SINGLE if args.case == "single" else DUAL)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
