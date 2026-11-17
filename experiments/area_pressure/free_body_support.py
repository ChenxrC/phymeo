"""Fixed FREE-body finite-pressure support window through production interfaces.

One actual six-DOF body per world, with no guide, pose reset, or tangential law.
Current HOST geometry drives CUDA dynamics. This producer records observations;
independent native/analytic acceptance belongs to the accompanying test.
"""

import argparse
import math
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

import numpy as np

from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.control.body_contact_wrench import BodyContactWrenchReader, BodyContactWrenchRequest

from .heterogeneous_support import _asset
from .three_body_release import _output_path, _plain, _ReadbackCapture, _save, _state

H, STEPS = 0.001, 150
WORLD_ORDER = (41, 7, 305)
MODEL = "free-cube-support"
PAIR = ("pressure", "plate")
PARAMETERS = MappingProxyType(
    dict(
        mass_kg=0.30,
        inertia_kg_m2=(0.30 / 6000, 0.30 / 6000, 0.30 / 3750),
        cube_half_size_m=(0.02, 0.02, 0.01),
        center_pressure_pa=100000.0,
        plate_half_size_m=(0.01, 0.01, 0.0025),
        plate_origin_world_m=(0.0, 0.0, -0.0025),
        gravity_m_s2=(0.0, 0.0, -9.81),
        damping=(0.0,) * 6,
        offset_m=0.001,
        tilt_pitch_rad=0.02,
        tilt_yaw_rad=0.35,
        equilibrium_depth_m=0.00073575,
        contact_capacity=16,
        constraint_capacity=64,
    )
)


def case_declarations():
    """Compile fixed declarations and return fresh assets; allocate no TYX owner."""
    p = PARAMETERS
    mass = p["mass_kg"]
    inertia = p["inertia_kg_m2"]
    xml = f'''<mujoco model="free-cube-pressure-support">
      <compiler angle="radian" fusestatic="false"/>
      <option timestep="{H}" gravity="0 0 -9.81" integrator="Euler"
        solver="Newton" iterations="128" tolerance="1e-10" cone="pyramidal" jacobian="dense"/>
      <default><geom condim="1" friction="0 0 0" contype="0" conaffinity="0"/>
        <joint damping="0" frictionloss="0" armature="0"/></default>
      <worldbody>
        <body name="pressure" pos="0 0 .011">
          <freejoint name="free"/>
          <inertial pos="0 0 0" mass="{mass}" diaginertia="{inertia[0]} {inertia[1]} {inertia[2]}"/>
          <geom name="pressure-boundary" type="box" size=".02 .02 .01"/>
        </body>
        <body name="plate" pos="0 0 -.0025">
          <geom name="plate-boundary" type="box" size=".01 .01 .0025"/>
        </body>
      </worldbody>
    </mujoco>'''
    pitch, yaw = p["tilt_pitch_rad"], p["tilt_yaw_rad"]
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    # qz(yaw) * qy(pitch), stored native WXYZ; qpos xyz is world body origin.
    tilted_quaternion = (cy * cp, -sy * sp, cy * sp, sy * cp)
    states = {
        41: dict(case="center-drop", initial_q=(0, 0, 0.011, 1, 0, 0, 0), initial_v=(0.0,) * 6),
        7: dict(
            case="tilted-support",
            initial_q=(
                0,
                0,
                (0.01 - p["equilibrium_depth_m"]) / math.cos(pitch),
                *tilted_quaternion,
            ),
            initial_v=(0.0,) * 6,
        ),
        305: dict(
            case="offset-support",
            initial_q=(p["offset_m"], 0, 0.00926425, 1, 0, 0, 0),
            initial_v=(0.0,) * 6,
        ),
    }
    compilation = compile_mujoco_xml(xml, require_asset=False)
    specs = (
        MuJoCoTyxModelSpec(
            MODEL,
            compilation,
            contact_capacity=p["contact_capacity"],
            constraint_capacity=p["constraint_capacity"],
            area_pressure_pairs=(PAIR,),
        ),
    )
    worlds = tuple(
        MuJoCoTyxWorldSpec(
            key,
            MODEL,
            MuJoCoInitialState(states[key]["initial_q"], states[key]["initial_v"]),
        )
        for key in WORLD_ORDER
    )
    # Reuse the accepted complete body-local cube mesh and finite top faces.
    assets = {MODEL: (_asset(MODEL, p),)}
    source = compilation.source
    manifest = dict(
        schema="free-body-area-support-v1",
        h=H,
        steps=STEPS,
        duration=H * STEPS,
        world_order=WORLD_ORDER,
        model_key=MODEL,
        parameters=p,
        worlds=states,
        xml=xml,
        assets=assets,
        source={
            name: getattr(source, name)
            for name in ("primary_sha256", "archive_sha256", "model_sha256", "mujoco_version")
        },
        geometry="HOST exact current full cube pressure volume / finite plate top faces",
        pressure_law="frictionless frozen-surface elastic backward Euler in shared TYX solve",
        coordinates="qpos xyz world body origin and WXYZ; qvel world linear and body-local angular",
        com_convention="declared inertial pos=0: body origin equals COM in this fixture only",
        constraints="one unrestricted FREE joint per world; no guides or native contacts",
        limitations="finite top surface, no tangential friction, no CCD or material calibration",
        qualification="producer records only; independent acceptance is a separate test audit",
    )
    return specs, worlds, assets, _plain(manifest)


def _energy(before, after, receipt):
    """Measured-state bookkeeping, not an independent dynamics oracle.

    Physical mass/inertia are constant in this explicitly centered uniform body.
    The row proxy is frozen only for this step, not a global pressure potential.
    """
    p = PARAMETERS
    v0, v1 = np.asarray(before.v, dtype=np.float64), np.asarray(after.v, dtype=np.float64)
    mass = np.diag((p["mass_kg"],) * 3 + p["inertia_kg_m2"])
    h = float(np.float32(receipt.h))
    k0, k1 = 0.5 * float(v0 @ mass @ v0), 0.5 * float(v1 @ mass @ v1)
    gravitational0 = -p["mass_kg"] * p["gravity_m_s2"][2] * float(before.q[2])
    gravitational1 = -p["mass_kg"] * p["gravity_m_s2"][2] * float(after.q[2])
    proxy0 = proxy1 = row_work = 0.0
    for area in receipt.raw_contact_fields["area_pressure"]:
        stiffness = np.asarray(area["stiffness"], dtype=np.float64)
        phi = np.asarray(area["surrogate_gap"], dtype=np.float64)
        jacobian = np.asarray(area["normal_jacobian"], dtype=np.float64)
        force = np.asarray(area["normal_force"], dtype=np.float64)
        speed = jacobian @ v1
        proxy0 += 0.5 * float(np.dot(stiffness, np.minimum(phi, 0) ** 2))
        proxy1 += 0.5 * float(np.dot(stiffness, np.minimum(phi + h * speed, 0) ** 2))
        row_work += h * float(force @ speed)
    dynamics = receipt.dynamics
    constraint_work = h * float(np.asarray(receipt.qfrc_constraint, dtype=np.float64) @ v1)
    applied_work = h * float(np.asarray(dynamics.applied, dtype=np.float64) @ v1)
    passive_work = h * float(np.asarray(dynamics.passive, dtype=np.float64) @ v1)
    bias_work = -h * float(np.asarray(dynamics.bias, dtype=np.float64) @ v1)
    dv = v1 - v0
    numerical_loss = 0.5 * float(dv @ mass @ dv)
    return dict(
        basis="declared constant physical mass/inertia and actual accepted velocities",
        integration_h_s=h,
        kinetic_entry_j=k0,
        kinetic_exit_j=k1,
        gravitational_entry_j=gravitational0,
        gravitational_exit_j=gravitational1,
        applied_work_j=applied_work,
        passive_work_j=passive_work,
        minus_bias_work_j=bias_work,
        contact_work_j=constraint_work,
        area_row_work_j=row_work,
        kinetic_numerical_loss_j=numerical_loss,
        kinetic_balance_residual_j=k1
        - k0
        - applied_work
        - passive_work
        - bias_work
        - constraint_work
        + numerical_loss,
        frozen_row_proxy_entry_j=proxy0,
        frozen_row_proxy_exit_j=proxy1,
        frozen_row_proxy_be_balance_j=proxy1 - proxy0 + row_work,
        proxy_scope="per-step fixed J/phi only; not full-trajectory physical pressure potential",
    )


def run_case(output, *, mode="cuda"):
    """Record the fixed 150-step batch and save accepted prefixes on failure."""
    path = _output_path(output)
    if type(mode) is not str or mode != "cuda":
        raise ValueError("this registered experiment admits CUDA only")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        schema="free-body-area-support-v1",
        mode=mode,
        h=H,
        steps=STEPS,
        duration=H * STEPS,
        world_order=list(WORLD_ORDER),
        status="started",
        accepted_steps=0,
        complete_records=0,
        records=[],
    )
    phase = "declarations"
    start_total = perf_counter()
    try:
        specs, worlds, assets, manifest = case_declarations()
        payload.update(manifest)
        phase = "setup"
        with ExitStack() as stack:
            batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode))
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            workspace = stack.enter_context(
                batch.prepare_device_workspace(
                    include_contacts=True,
                    include_bodies=True,
                    include_area_pressure=True,
                )
            )
            capture = _ReadbackCapture(workspace)
            reader = BodyContactWrenchReader(
                capture,
                area_identities={
                    name: tuple(asset.identity for asset in rows)
                    for name, rows in scene.assets.items()
                },
            )
            names = workspace.model_metadata[MODEL]["body_names"]
            pressure_id = names.index("pressure")
            payload["body_names"] = _plain(names)
            payload["asset_sha256"] = scene.asset_sha256
            payload["initial"] = {
                str(key): _state(value) for key, value in batch.integration_snapshot().items()
            }
            payload["world_layout"] = _plain(batch.layout)
            payload["setup_s"] = perf_counter() - start_total
            for step in range(STEPS):
                start = perf_counter()
                phase = "physics"
                before = batch.integration_snapshot()
                accepted = scene.step(H, expected_revision=batch.checkpoint_revision)
                payload["accepted_steps"] += 1
                record = dict(
                    step=step,
                    state_accepted=True,
                    observation_complete=False,
                    accepted=_plain(accepted),
                    worlds={str(key): dict(input=_state(before[key])) for key in WORLD_ORDER},
                )
                payload["records"].append(record)
                phase = "accepted-state"
                after = batch.integration_snapshot()
                requests, descriptors = [], []
                for key in WORLD_ORDER:
                    receipt = batch.last_step.receipts[key]
                    row = record["worlds"][str(key)]
                    row["output"] = _state(after[key])
                    row["receipt"] = _plain(receipt)
                    row["geometry"] = _plain(scene.last_geometry[key])
                    references = dict(
                        world=(0.0, 0.0, 0.0),
                        body_origin=receipt.dynamics.body_position[pressure_id],
                        com=receipt.dynamics.body_com[pressure_id],
                    )
                    row["reference_points_world"] = _plain(references)
                    row["body_wrenches"] = {name: {} for name in references}
                    row["energy"] = _energy(before[key], after[key], receipt)
                    for reference_name, reference in references.items():
                        for body, other in (("pressure", "plate"), ("plate", "pressure")):
                            requests.append(
                                BodyContactWrenchRequest(key, body, reference, (other,))
                            )
                            descriptors.append((key, reference_name, body))
                if dict(accepted) != dict.fromkeys(WORLD_ORDER, True):
                    raise RuntimeError("whole batch step was not accepted")
                phase = "observation"
                frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
                capture.reset()
                measured = reader.measure_many(frame, tuple(requests))
                if capture.calls != 1 or capture.arrays is None:
                    raise RuntimeError(
                        "expected one public readback for all eighteen wrench queries"
                    )
                if any(value is None for value in measured):
                    raise RuntimeError("accepted step lacks a current body measurement")
                record["arrays"] = _plain(capture.arrays)
                record["catalogs"] = _plain(frame.area_catalogs)
                record["frame"] = dict(
                    revision=frame.revision,
                    epoch=frame.epoch,
                    ticks=_plain(frame.ticks),
                    world_boundaries=_plain(frame.world_boundaries),
                )
                for (key, reference, body), value in zip(descriptors, measured, strict=True):
                    record["worlds"][str(key)]["body_wrenches"][reference][body] = _plain(value)
                record["readback_calls"] = capture.calls
                record["elapsed_s"] = perf_counter() - start
                record["observation_complete"] = True
                payload["complete_records"] += 1
            payload["final"] = {
                str(key): _state(value) for key, value in batch.integration_snapshot().items()
            }
            payload["observation_counters"] = _plain(workspace.counters)
            phase = "close"
        payload["status"] = "complete"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error"] = dict(type=type(error).__name__, message=str(error), phase=phase)
        raise
    finally:
        payload["total_s"] = perf_counter() - start_total
        _save(path, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_case(args.output)


if __name__ == "__main__":
    main()
