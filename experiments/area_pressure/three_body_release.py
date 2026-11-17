"""Fixed three-body pressure-preload release through public TYX interfaces.

Three vertical guides carry transverse reactions and torques. This is not a
free-body balance, friction or material-calibration experiment. Current HOST
volume/finite-face intersection supplies the shared CUDA constraint solve.
The producer records observations only; independent acceptance is in the test.
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
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.control.body_contact_wrench import BodyContactWrenchReader, BodyContactWrenchRequest
from phymeo.geometry.area_pressure import AreaPressureIdentity
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh, TetraPressureMesh

H, STEPS, WORLD = 0.001, 40, 13
MODEL = "three-body-release"
BODY_ORDER = ("pressure", "left", "right")
MASSES = (0.5, 1.25, 0.75)
PATCHES = (("left", -0.035, 0.5), ("right", 0.035, 1.0))
INITIAL_Q, INITIAL_V = (-0.003, 0.0, 0.0), (0.0, 0.0, 0.0)


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


def case_declarations(*, reverse_pairs=False):
    """Compile the fixed declarations only; allocate no TYX runtime."""
    if type(reverse_pairs) is not bool:
        raise ValueError("reverse_pairs must be a literal bool")
    bodies = "".join(
        f'''<body name="{name}" pos="0 0 {".002" if name == "pressure" else "0"}">
          <joint name="{name}-vertical" type="slide" axis="0 0 1"
            limited="false" damping="0" frictionloss="0" armature="0"/>
          <inertial pos="0 0 0" mass="{mass}" diaginertia=".02 .02 .02"/>
          <geom name="{name}-declaration" type="sphere" size="{".01" if name == "pressure" else ".001"}"
            contype="0" conaffinity="0"/>
        </body>'''
        for name, mass in zip(BODY_ORDER, MASSES, strict=True)
    )
    xml = f'''<mujoco model="three-body-area-release">
      <compiler angle="radian" fusestatic="false"/>
      <option timestep="{H}" gravity="0 0 0" integrator="Euler" solver="Newton"
        iterations="128" tolerance="1e-10" cone="pyramidal" jacobian="dense"/>
      <worldbody>{bodies}</worldbody>
    </mujoco>'''
    outer = np.array(((-0.12, -0.12, 0), (0.12, -0.12, 0), (0, 0.12, 0), (0, 0, 0.08)))
    pressure = TetraPressureMesh(
        ("a", "b", "c", "d", "center"),
        np.vstack((outer, outer.mean(axis=0))),
        ((0, 1, 2, 4), (0, 2, 3, 4), (0, 3, 1, 4), (1, 3, 2, 4)),
        (0, 0, 0, 0, 400000),
        ("bottom", "side-a", "side-b", "slant"),
        pressure_domain="full",
    )
    patches = tuple(reversed(PATCHES)) if reverse_pairs else PATCHES
    rows = []
    for name, offset, scale in patches:
        face = np.array(((-0.015, -0.045, 0), (0.015, -0.045, 0), (0, -0.015, 0)))
        face[:, 0] = face[:, 0] * scale + offset
        rows.append(
            BodyAreaPressureAsset(
                AreaPressureIdentity("pressure", name, "linear-pressure-full", "finite-rigid-face"),
                pressure,
                RigidPressureMesh(("r0", "r1", "r2"), face, ((0, 1, 2),), ("face",)),
            )
        )
    compilation = compile_mujoco_xml(xml, require_asset=False)
    specs = (
        MuJoCoTyxModelSpec(
            MODEL,
            compilation,
            contact_capacity=16,
            constraint_capacity=64,
            area_pressure_pairs=tuple(("pressure", name) for name, _, _ in patches),
        ),
    )
    worlds = (MuJoCoTyxWorldSpec(WORLD, MODEL, MuJoCoInitialState(INITIAL_Q, INITIAL_V)),)
    assets = {MODEL: tuple(rows)}
    source = compilation.source
    manifest = dict(
        schema="three-body-area-release-v1",
        h=H,
        steps=STEPS,
        duration=H * STEPS,
        world_key=WORLD,
        model_key=MODEL,
        body_order=BODY_ORDER,
        pair_order=tuple(name for name, _, _ in patches),
        mass_kg=MASSES,
        body_origin_world=((0.0, 0.0, 0.002), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        initial_q=INITIAL_Q,
        initial_v=INITIAL_V,
        gravity_m_s2=(0.0, 0.0, 0.0),
        joint_damping_ns_m=(0.0, 0.0, 0.0),
        generalized_external_force_n=(0.0, 0.0, 0.0),
        constraint_capacity=64,
        contact_capacity=16,
        patches={
            name: dict(
                area_m2=0.00045 * scale,
                reference_stiffness_n_m=9000 * scale,
                x_offset_m=offset,
                x_scale=scale,
            )
            for name, offset, scale in PATCHES
        },
        xml=xml,
        assets=assets,
        source={
            name: getattr(source, name)
            for name in ("primary_sha256", "archive_sha256", "model_sha256", "mujoco_version")
        },
        geometry="HOST exact current full pressure volume / finite rigid triangle",
        constraints="three vertical guides carry transverse forces and moments",
        pressure_law="frictionless frozen-surface elastic backward Euler in shared TYX solve",
        physical_geometry="body-local area assets; native declaration geoms never collide",
    )
    return specs, worlds, assets, _plain(manifest)


def _state(snapshot):
    return dict(
        q=_plain(snapshot.q),
        v=_plain(snapshot.v),
        act=_plain(snapshot.act),
        time=float(snapshot.time),
        tick=int(snapshot.tick),
    )


class _ReadbackCapture:
    """Capture one real public owned readback for this frame's diagnostic log.

    This experiment-only proxy neither alters arrays nor caches across frames.
    The borrowed workspace remains responsible for all freshness checks.
    """

    def __init__(self, workspace):
        self.workspace = workspace
        self.reset()

    @property
    def binding(self):
        return self.workspace.binding

    @property
    def model_metadata(self):
        return self.workspace.model_metadata

    def reset(self):
        self.calls = 0
        self.arrays = None

    def readback(self, frame):
        self.calls += 1
        self.arrays = self.workspace.readback(frame)
        return self.arrays


def _output_path(output):
    path = Path(output)
    if not path.is_absolute() or path.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("output must be an absolute .json.gz file")
    if path.exists():
        raise FileExistsError("refusing to replace a saved trajectory")
    return path


def _save(path, payload):
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )
    with path.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
            compressed.write(raw)


def run_case(output, *, mode="cuda", reverse_first_step=False):
    """Run forty fixed steps or the single registered reversed-pair first step.

    A failure still saves the physical accepted prefix, including the latest
    partial observation. Exceptions propagate; no failure is a successful case.
    """
    path = _output_path(output)
    if type(mode) is not str or mode != "cuda":
        raise ValueError("this registered experiment admits CUDA only")
    if type(reverse_first_step) is not bool:
        raise ValueError("reverse_first_step must be a literal bool")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 1 if reverse_first_step else STEPS
    payload = dict(
        schema="three-body-area-release-v1",
        mode=mode,
        reverse_first_step=reverse_first_step,
        h=H,
        steps=count,
        duration=count * H,
        status="started",
        accepted_steps=0,
        complete_records=0,
        records=[],
    )
    phase = "declarations"
    start_total = perf_counter()
    try:
        specs, worlds, assets, manifest = case_declarations(reverse_pairs=reverse_first_step)
        payload.update(manifest)
        payload.update(steps=count, duration=count * H)
        phase = "setup"
        with ExitStack() as stack:
            batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode))
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            workspace = stack.enter_context(
                batch.prepare_device_workspace(
                    include_contacts=True, include_bodies=True, include_area_pressure=True
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
            requests = tuple(
                BodyContactWrenchRequest(
                    WORLD,
                    name,
                    (0.0, 0.0, 0.0),
                    ("left", "right") if name == "pressure" else ("pressure",),
                )
                for name in BODY_ORDER
            )
            payload["asset_sha256"] = scene.asset_sha256
            payload["body_names"] = _plain(workspace.model_metadata[MODEL]["body_names"])
            payload["initial"] = _state(batch.integration_snapshot()[WORLD])
            payload["setup_s"] = perf_counter() - start_total
            for step in range(count):
                start = perf_counter()
                phase = "physics"
                before = batch.integration_snapshot()[WORLD]
                accepted = scene.step(H, expected_revision=batch.checkpoint_revision)
                # Persist acceptance before anything that can fail in export.
                payload["accepted_steps"] += 1
                record = dict(
                    step=step,
                    state_accepted=True,
                    observation_complete=False,
                    input=_state(before),
                    accepted=_plain(accepted),
                )
                payload["records"].append(record)
                phase = "accepted-state"
                record["output"] = _state(batch.integration_snapshot()[WORLD])
                record["receipt"] = _plain(batch.last_step.receipts[WORLD])
                if dict(accepted) != {WORLD: True}:
                    raise RuntimeError("whole batch step was not accepted")
                record["geometry"] = _plain(scene.last_geometry[WORLD])
                phase = "observation"
                frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
                capture.reset()
                measured = reader.measure_many(frame, requests)
                if capture.calls != 1 or capture.arrays is None:
                    raise RuntimeError(
                        "expected exactly one public readback for three body queries"
                    )
                if any(value is None for value in measured):
                    raise RuntimeError("accepted step did not yield a current body measurement")
                record["arrays"] = _plain(capture.arrays)
                record["catalogs"] = _plain(frame.area_catalogs)
                record["body_wrenches"] = {
                    name: _plain(value) for name, value in zip(BODY_ORDER, measured, strict=True)
                }
                record["readback_calls"] = capture.calls
                record["elapsed_s"] = perf_counter() - start
                record["observation_complete"] = True
                payload["complete_records"] += 1
            payload["final"] = _state(batch.integration_snapshot()[WORLD])
            phase = "close"
        payload["status"] = "complete"
    except Exception as error:
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
    parser.add_argument("--reverse-first-step", action="store_true")
    args = parser.parse_args()
    run_case(args.output, reverse_first_step=args.reverse_first_step)


if __name__ == "__main__":
    main()
