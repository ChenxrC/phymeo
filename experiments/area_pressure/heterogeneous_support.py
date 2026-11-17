"""Fixed heterogeneous continuous support demonstration.

Dynamics run on TYX CUDA; full pressure-volume/finite-face clipping stays HOST.
All moving bodies have vertical guides, which carry transverse forces/moments.
This is not a free-balance, controller, training or material-calibration result.
"""

import argparse
import gzip
import json
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.control.body_contact_wrench import BodyContactWrenchReader
from phymeo.geometry.area_pressure import AreaPressureIdentity
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh, TetraPressureMesh

H, DURATION, STEPS = 0.001, 0.45, 450
WORLD_ORDER = (41, 7, 305, 11)
STAGES = ((0.15, 0.0), (0.30, -1.5), (0.45, 7.0))
PAIR = ("pressure", "plate")
_FACES = (
    (0, 2, 1),
    (0, 3, 2),
    (4, 5, 6),
    (4, 6, 7),
    (0, 1, 5),
    (0, 5, 4),
    (1, 2, 6),
    (1, 6, 5),
    (2, 3, 7),
    (2, 7, 6),
    (3, 0, 4),
    (3, 4, 7),
)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("unsupported saved observation type: " + type(value).__name__)


def _xml(name, p):
    native = name == "native"
    geom = 'type="sphere" size=".01"' if native else 'type="box" size=".02 .02 .01"'
    flags = 'contype="1" conaffinity="1"' if native else 'contype="0" conaffinity="0"'
    half = p["plate_half_size_m"]
    mass = p["mass_kg"]
    inertia = (mass * 0.00004,) * 3 if native else (mass / 6000, mass / 6000, mass / 3750)
    return f'''<mujoco model="heterogeneous-support-{name}">
    <compiler angle="radian" fusestatic="false"/>
    <option timestep="{H}" gravity="0 0 -9.81" integrator="Euler"
      solver="Newton" iterations="128" tolerance="1e-10" cone="pyramidal" jacobian="dense"/>
    <default><geom condim="1" friction="0 0 0" solref=".02 1"/></default>
    <worldbody>
      <body name="pressure" pos="0 0 {p["body_origin_z_m"]}">
        <joint name="vertical" type="slide" axis="0 0 1" damping="{p["damping_ns_m"]}" armature="0"/>
        <inertial pos="0 0 0" mass="{mass}" diaginertia="{inertia[0]} {inertia[1]} {inertia[2]}"/>
        <geom name="moving-geom" {geom} {flags}/>
      </body>
      <body name="plate" pos="{p["plate_center_x_m"]} 0 {p["plate_origin_z_m"]}">
        <geom name="plate-geom" type="box" size="{half[0]} {half[1]} {half[2]}" {flags}/>
      </body>
    </worldbody></mujoco>'''


def _asset(name, p):
    vertices = np.array(
        (
            (-0.02, -0.02, -0.01),
            (0.02, -0.02, -0.01),
            (0.02, 0.02, -0.01),
            (-0.02, 0.02, -0.01),
            (-0.02, -0.02, 0.01),
            (0.02, -0.02, 0.01),
            (0.02, 0.02, 0.01),
            (-0.02, 0.02, 0.01),
            (0, 0, 0),
        )
    )
    pressure = TetraPressureMesh(
        tuple(f"corner-{i}" for i in range(8)) + ("center",),
        vertices,
        tuple((8, *face) for face in _FACES),
        (0,) * 8 + (p["center_pressure_pa"],),
        ("bottom-0", "bottom-1") + tuple(f"other-{i}" for i in range(10)),
        pressure_domain="full",
    )
    x, y, z = p["plate_half_size_m"]
    rigid = RigidPressureMesh(
        ("r0", "r1", "r2", "r3"),
        ((-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)),
        ((0, 1, 2), (0, 2, 3)),
        ("top-a", "top-b"),
    )
    return BodyAreaPressureAsset(
        AreaPressureIdentity(*PAIR, name + "-full-pressure", name + "-finite-plate"),
        pressure,
        rigid,
    )


def case_declarations():
    """Compile only the three fixed declarations; allocate no TYX runtime."""
    base = dict(body_half_height_m=0.01, plate_origin_z_m=-0.0025)
    models = {
        "soft": dict(
            base,
            mass_kg=0.30,
            damping_ns_m=70.0,
            body_origin_z_m=0.012,
            shape="cube",
            center_pressure_pa=100000.0,
            plate_half_size_m=(0.01, 0.01, 0.0025),
            plate_center_x_m=0.0,
            reference_stiffness_n_m=4000.0,
        ),
        "offset": dict(
            base,
            mass_kg=0.45,
            damping_ns_m=90.0,
            body_origin_z_m=0.013,
            shape="cube",
            center_pressure_pa=150000.0,
            plate_half_size_m=(0.01, 0.0075, 0.0025),
            plate_center_x_m=0.004,
            reference_stiffness_n_m=4500.0,
        ),
        "native": dict(
            base,
            mass_kg=0.30,
            damping_ns_m=70.0,
            body_origin_z_m=0.012,
            shape="sphere",
            center_pressure_pa=None,
            plate_half_size_m=(0.01, 0.01, 0.0025),
            plate_center_x_m=0.0,
            reference_stiffness_n_m=None,
        ),
    }
    worlds = {
        41: dict(model="soft", initial_q=[0.0], initial_v=[0.0]),
        7: dict(model="offset", initial_q=[0.0], initial_v=[0.0]),
        305: dict(model="soft", initial_q=[-0.001], initial_v=[-0.03]),
        11: dict(model="native", initial_q=[0.0], initial_v=[0.0]),
    }
    xml = {name: _xml(name, p) for name, p in models.items()}
    specs = tuple(
        MuJoCoTyxModelSpec(
            name,
            compile_mujoco_xml(xml[name], require_asset=False),
            contact_capacity=16,
            constraint_capacity=64,
            area_pressure_pairs=() if name == "native" else (PAIR,),
        )
        for name in models
    )
    world_specs = tuple(
        MuJoCoTyxWorldSpec(
            key,
            worlds[key]["model"],
            MuJoCoInitialState(worlds[key]["initial_q"], worlds[key]["initial_v"]),
        )
        for key in WORLD_ORDER
    )
    assets = {name: (_asset(name, p),) for name, p in models.items() if name != "native"}
    manifest = dict(
        schema="heterogeneous-area-support-v1",
        h=H,
        duration=DURATION,
        world_order=WORLD_ORDER,
        models=models,
        worlds=worlds,
        xml=xml,
        stages=tuple(dict(end_s=end, force_n=force) for end, force in STAGES),
        geometry="HOST exact current full volume / finite plate",
        constraints="vertical guides carry transverse forces and moments",
        area_law="frictionless pressure-derived normal rows in shared TYX solve",
        native_comparison="different geometry and contact law; no equal-physics speed claim",
    )
    return specs, world_specs, assets, _plain(manifest)


def stage_for_step(step):
    if type(step) is not int or not 0 <= step < STEPS:
        raise ValueError("step must be an integer in the registered window")
    index = 0 if step < 150 else 1 if step < 300 else 2
    return ("gravity", "loaded", "release")[index], STAGES[index][1]


def _state(snapshot):
    return dict(
        q=_plain(snapshot.q),
        v=_plain(snapshot.v),
        act=_plain(snapshot.act),
        time=float(snapshot.time),
        tick=int(snapshot.tick),
    )


def _output_path(output):
    path = Path(output)
    if not path.is_absolute() or path.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("output must be an absolute .json.gz file")
    if path.exists():
        raise FileExistsError("refusing to replace a saved trajectory")
    return path


def run_case(output, *, mode="cuda"):
    """One fixed whole batch trajectory; save the accepted prefix on failure."""
    path = _output_path(output)
    if mode != "cuda":
        raise ValueError("this fixed demonstration admits CUDA only")
    specs, worlds, assets, payload = case_declarations()
    payload.update(mode=mode, status="started", records=[])
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ExitStack() as stack:
            batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode))
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            workspace = stack.enter_context(
                batch.prepare_device_workspace(
                    include_contacts=True, include_bodies=True, include_area_pressure=True
                )
            )
            identities = {
                name: tuple(a.identity for a in values) for name, values in scene.assets.items()
            }
            reader = BodyContactWrenchReader(workspace, area_identities=identities)
            payload["asset_sha256"] = scene.asset_sha256
            payload["model_body_names"] = _plain(
                {name: values["body_names"] for name, values in workspace.model_metadata.items()}
            )
            payload["initial"] = {
                str(k): _state(s) for k, s in batch.integration_snapshot().items()
            }
            payload["world_layout"] = _plain(batch.layout)
            for step in range(STEPS):
                phase, force = stage_for_step(step)
                start = perf_counter()
                before = batch.integration_snapshot()
                controls = {
                    key: MuJoCoControl(
                        np.zeros(row.nu, np.float32),
                        qfrc_applied=np.array([force], np.float32),
                        xfrc_applied=np.zeros((row.nbody, 6), np.float32),
                    )
                    for key, row in batch.layout.items()
                }
                batch.set_controls(controls)
                accepted = scene.step(H, expected_revision=batch.checkpoint_revision)
                if dict(accepted) != dict.fromkeys(WORLD_ORDER, True):
                    raise RuntimeError("whole batch step was not accepted")
                after = batch.integration_snapshot()
                frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
                arrays = workspace.readback(frame)
                rows = {}
                for row, key in enumerate(WORLD_ORDER):
                    names = workspace.model_metadata[batch.layout[key].model_key]["body_names"]
                    reference = arrays["entry_body_position"][row, names.index("pressure")]
                    receipt = batch.last_step.receipts[key]
                    rows[str(key)] = dict(
                        input=_state(before[key]),
                        output=_state(after[key]),
                        geometry=_plain(scene.last_geometry.get(key, ())),
                        area=_plain(receipt.raw_contact_fields.get("area_pressure", ())),
                        dynamics={
                            name: _plain(getattr(receipt.dynamics, name))
                            for name in (
                                "mass_matrix",
                                "bias",
                                "passive",
                                "applied",
                                "smooth_acceleration",
                            )
                        },
                        qfrc_constraint=_plain(receipt.qfrc_constraint),
                        integration_acceleration=_plain(receipt.integration_acceleration),
                        body_wrenches={
                            name: _plain(
                                reader.measure(
                                    frame,
                                    world_key=key,
                                    body_name=name,
                                    reference_point_world=reference,
                                )
                            )
                            for name in PAIR
                        },
                    )
                payload["records"].append(
                    dict(
                        step=step,
                        phase=phase,
                        force_n=force,
                        worlds=rows,
                        arrays=_plain(arrays),
                        catalogs=_plain(frame.area_catalogs),
                        elapsed_s=perf_counter() - start,
                    )
                )
            payload["final"] = {str(k): _state(s) for k, s in batch.integration_snapshot().items()}
            payload["observation_counters"] = _plain(workspace.counters)
            payload["status"] = "complete"
    except BaseException as error:
        payload["status"] = "failed"
        payload["error"] = dict(
            type=type(error).__name__, message=str(error), accepted_records=len(payload["records"])
        )
        raise
    finally:
        raw = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        with path.open("xb") as file:
            with gzip.GzipFile(fileobj=file, mode="wb", mtime=0) as compressed:
                compressed.write(raw)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_case(args.output)


if __name__ == "__main__":
    main()
