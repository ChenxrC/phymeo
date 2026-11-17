"""Consume an installed Phymeo wheel outside every source checkout.

Run a copied script with Python -I --site ABS_TARGET --output ABS_NEW_DIRECTORY.
Exactly four mixed CUDA batch steps exercise the frozen I3t release boundary.
This is a packaging consumer, not a new physics or friction qualification.
"""

import argparse
import base64
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
import traceback
import zipfile
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

H = 0.001
WORLD_ORDER = (41, 7, 305)
AREA_MODEL, NATIVE_MODEL = "free-cube-support", "native"
MODEL_BY_WORLD = {41: AREA_MODEL, 7: NATIVE_MODEL, 305: AREA_MODEL}
np = None  # Imported only after the source/installation guards below.

AREA_XML = """<mujoco model="free-cube-pressure-support">
  <compiler angle="radian" fusestatic="false"/>
  <option timestep=".001" gravity="0 0 -9.81" integrator="Euler"
    solver="Newton" iterations="128" tolerance="1e-10" cone="pyramidal" jacobian="dense"/>
  <default><geom condim="1" friction="0 0 0" contype="0" conaffinity="0"/>
    <joint damping="0" frictionloss="0" armature="0"/></default>
  <worldbody>
    <body name="pressure" pos="0 0 .011">
      <freejoint name="free"/>
      <inertial pos="0 0 0" mass=".3" diaginertia=".00005 .00005 .00008"/>
      <geom name="pressure-boundary" type="box" size=".02 .02 .01"/>
    </body>
    <body name="plate" pos="0 0 -.0025">
      <geom name="plate-boundary" type="box" size=".01 .01 .0025"/>
    </body>
  </worldbody>
</mujoco>"""

NATIVE_XML = """<mujoco model="heterogeneous-support-native">
  <compiler angle="radian" fusestatic="false"/>
  <option timestep=".001" gravity="0 0 -9.81" integrator="Euler"
    solver="Newton" iterations="128" tolerance="1e-10" cone="pyramidal" jacobian="dense"/>
  <default><geom condim="1" friction="0 0 0" solref=".02 1"/></default>
  <worldbody>
    <body name="pressure" pos="0 0 .012">
      <joint name="vertical" type="slide" axis="0 0 1" damping="70" armature="0"/>
      <inertial pos="0 0 0" mass=".3" diaginertia=".000012 .000012 .000012"/>
      <geom name="moving-geom" type="sphere" size=".01" contype="1" conaffinity="1"/>
    </body>
    <body name="plate" pos="0 0 -.0025">
      <geom name="plate-geom" type="box" size=".01 .01 .0025" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>"""


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within(path, root):
    return path.resolve().is_relative_to(root.resolve())


def _checkout(path):
    return (path / "pyproject.toml").is_file() and (path / "src" / "phymeo").is_dir()


def _outside_checkout(path):
    _require(
        not any(_checkout(p) for p in (path, *path.parents)),
        "path inside Phymeo checkout: " + str(path),
    )


def _guard_before(site, output):
    _require(site.is_absolute() and site.is_dir(), "--site must be an existing absolute target")
    _require(
        output.is_absolute() and not output.exists(), "--output must be a fresh absolute directory"
    )
    _require(sys.flags.isolated == 1, "run the external consumer with Python -I")
    _require(not os.environ.get("PYTHONPATH"), "clear PYTHONPATH before launching")
    for path in (
        site.resolve(),
        output.resolve(),
        Path.cwd().resolve(),
        Path(__file__).resolve().parent,
    ):
        _outside_checkout(path)
    _require(
        not any(k == "phymeo" or k.startswith("phymeo.") for k in sys.modules),
        "Phymeo was preloaded",
    )
    _require(
        (site / "phymeo" / "__init__.py").is_file(), "target does not contain installed Phymeo"
    )
    before = tuple(sys.path)
    pth = {}
    for entry in before:
        path = Path(entry or os.getcwd()).resolve()
        _require(not _checkout(path), "source checkout on sys.path")
        _require(not (_checkout(path.parent) and path.name == "src"), "source src on sys.path")
        candidate = (path / "phymeo").exists() or (path / "phymeo.py").exists()
        if path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                candidate = any(
                    n == "phymeo.py" or n.startswith("phymeo/") for n in archive.namelist()
                )
        _require(not candidate or path == site.resolve(), "another Phymeo candidate: " + str(path))
        if path.is_dir():
            for item in sorted(path.glob("*.pth")):
                pth[str(item)] = dict(
                    sha256=_sha(item), text=item.read_text(encoding="utf-8", errors="replace")
                )
    sys.path.insert(0, str(site.resolve()))
    spec = importlib.util.find_spec("phymeo")
    _require(
        spec is not None and spec.origin is not None and _within(Path(spec.origin), site),
        "Phymeo spec is outside target",
    )
    return dict(
        sys_path_before=before,
        sys_path_after=tuple(sys.path),
        dependency_pth=pth,
        executable=sys.executable,
        python=sys.version,
        isolated=sys.flags.isolated,
        cwd=str(Path.cwd()),
        script=str(Path(__file__).resolve()),
        script_sha256=_sha(Path(__file__)),
        site=str(site.resolve()),
        output=str(output.resolve()),
    )


def _origins(site):
    modules, errors = {}, []
    for name, module in sorted(tuple(sys.modules.items())):
        if name == "phymeo" or name.startswith("phymeo."):
            origin = getattr(module, "__file__", None)
            if origin is None or not _within(Path(origin), site):
                errors.append(name + ": " + str(origin))
            elif not Path(origin).is_file():
                errors.append(name + ": missing origin")
            else:
                modules[name] = dict(path=str(Path(origin).resolve()), sha256=_sha(Path(origin)))
    _require(bool(modules), "no installed Phymeo modules were loaded")
    _require(not errors, "loaded Phymeo outside target: " + repr(errors))
    dependencies = {}
    for name in ("numpy", "mujoco", "tyx"):
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        direct_url = None
        try:
            distribution = importlib.metadata.distribution("tyx-compute" if name == "tyx" else name)
            version = distribution.version
            direct_url = distribution.read_text("direct_url.json")
        except importlib.metadata.PackageNotFoundError:
            version = str(getattr(module, "__version__", "unavailable"))
        dependencies[name] = dict(
            version=version,
            path=origin,
            direct_url_json=direct_url,
            sha256=_sha(Path(origin)) if origin and Path(origin).is_file() else None,
        )
    return dict(
        phymeo_modules=modules,
        phymeo_version=importlib.metadata.version("phymeo"),
        dependencies=dependencies,
        sys_path_final=tuple(sys.path),
    )


def _api():
    global np
    import numpy as numpy

    from phymeo.backends.area_pressure_assets import AreaPressureAssetBundle
    from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
    from phymeo.backends.mujoco_heterogeneous import (
        MuJoCoTyxHeterogeneousBatch,
        MuJoCoTyxModelSpec,
        MuJoCoTyxWorldSpec,
    )
    from phymeo.backends.mujoco_heterogeneous_area_scene import (
        MuJoCoTyxHeterogeneousAreaPressureScene,
        MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
    )
    from phymeo.backends.mujoco_runtime import MuJoCoInitialState
    from phymeo.backends.tyx_area_pressure_intersection import TyxAreaPressureIntersector
    from phymeo.builders.mujoco_compiler import compile_mujoco_xml
    from phymeo.control.body_contact_wrench import BodyContactWrenchReader, BodyContactWrenchRequest
    from phymeo.geometry.area_pressure import AreaPressureIdentity
    from phymeo.geometry.area_pressure_intersection import RigidPressureMesh, TetraPressureMesh

    np = numpy
    return SimpleNamespace(
        AreaPressureAssetBundle=AreaPressureAssetBundle,
        BodyAreaPressureAsset=BodyAreaPressureAsset,
        MuJoCoTyxHeterogeneousBatch=MuJoCoTyxHeterogeneousBatch,
        MuJoCoTyxModelSpec=MuJoCoTyxModelSpec,
        MuJoCoTyxWorldSpec=MuJoCoTyxWorldSpec,
        MuJoCoTyxHeterogeneousAreaPressureScene=MuJoCoTyxHeterogeneousAreaPressureScene,
        MuJoCoTyxHeterogeneousAreaSceneCheckpoint=MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
        MuJoCoInitialState=MuJoCoInitialState,
        TyxAreaPressureIntersector=TyxAreaPressureIntersector,
        compile_mujoco_xml=compile_mujoco_xml,
        BodyContactWrenchReader=BodyContactWrenchReader,
        BodyContactWrenchRequest=BodyContactWrenchRequest,
        AreaPressureIdentity=AreaPressureIdentity,
        RigidPressureMesh=RigidPressureMesh,
        TetraPressureMesh=TetraPressureMesh,
    )


def _plain(value):
    """Preserve complete numerical bytes, including dtype and signed zeros."""
    if np is not None and isinstance(value, np.ndarray):
        return dict(
            dtype=value.dtype.str,
            shape=list(value.shape),
            bytes_b64=base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        )
    if np is not None and isinstance(value, np.generic):
        return _plain(np.asarray(value))
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, bytes):
        return dict(bytes_b64=base64.b64encode(value).decode("ascii"))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("unsupported or nonfinite saved value: " + type(value).__name__)


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_plain(value), stream, ensure_ascii=True, allow_nan=False, indent=2)


def _event(stream, value):
    stream.write(
        (json.dumps(_plain(value), ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    )
    stream.flush()


def _declarations(a):
    vertices = (
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
    faces = (
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
    pressure = a.TetraPressureMesh(
        tuple(f"corner-{i}" for i in range(8)) + ("center",),
        vertices,
        tuple((8, *f) for f in faces),
        (0,) * 8 + (100000.0,),
        ("bottom-0", "bottom-1") + tuple(f"other-{i}" for i in range(10)),
        pressure_domain="full",
    )
    rigid = a.RigidPressureMesh(
        ("r0", "r1", "r2", "r3"),
        (
            (-0.01, -0.01, 0.0025),
            (0.01, -0.01, 0.0025),
            (0.01, 0.01, 0.0025),
            (-0.01, 0.01, 0.0025),
        ),
        ((0, 1, 2), (0, 2, 3)),
        ("top-a", "top-b"),
    )
    identity = a.AreaPressureIdentity(
        "pressure", "plate", AREA_MODEL + "-full-pressure", AREA_MODEL + "-finite-plate"
    )
    assets = {AREA_MODEL: (a.BodyAreaPressureAsset(identity, pressure, rigid),)}
    specs = tuple(
        a.MuJoCoTyxModelSpec(
            name,
            a.compile_mujoco_xml(xml, require_asset=False),
            contact_capacity=16,
            constraint_capacity=64,
            area_pressure_pairs=(("pressure", "plate"),) if name == AREA_MODEL else (),
        )
        for name, xml in ((AREA_MODEL, AREA_XML), (NATIVE_MODEL, NATIVE_XML))
    )
    states = {
        41: ((0, 0, 0.00926425, 1, 0, 0, 0), (0.0,) * 6),
        7: ((-0.003,), (0.0,)),
        305: ((0, 0, 0.0095, 1, 0, 0, 0), (0, 0, -0.02, 0, 0, 0)),
    }
    worlds = tuple(
        a.MuJoCoTyxWorldSpec(k, MODEL_BY_WORLD[k], a.MuJoCoInitialState(*states[k]))
        for k in WORLD_ORDER
    )
    return specs, worlds, assets


def _state(value):
    return {
        name: getattr(value, name)
        for name in (
            "q",
            "v",
            "act",
            "ctrl",
            "time",
            "tick",
            "qfrc_applied",
            "xfrc_applied",
            "held_control",
        )
    }


def _exact_array(actual, expected, name):
    _require(
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and actual.tobytes(order="C") == expected.tobytes(order="C"),
        "array bytes differ: " + name,
    )


def _exact_states(actual, expected, keys=WORLD_ORDER):
    for key in keys:
        _require(
            _plain(_state(actual[key])) == _plain(_state(expected[key])),
            "state bytes differ: " + str(key),
        )


def _exact_arrays(actual, expected, rows=None):
    _require(set(actual) == set(expected), "observation field set differs")
    for name in actual:
        left, right = actual[name], expected[name]
        if rows is not None:
            _require(
                left.shape[0] == len(WORLD_ORDER), "unexpected non-world observation field: " + name
            )
            left, right = left[list(rows)], right[list(rows)]
        _exact_array(left, right, name)


def _frame(frame):
    return dict(
        revision=frame.revision,
        epoch=frame.epoch,
        ticks=frame.ticks,
        world_boundaries=frame.world_boundaries,
        binding=frame.binding,
        area_catalogs=frame.area_catalogs,
    )


class _ReadbackCapture:
    """Retain this call's single unmodified public copy; no cross-frame cache."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.reset()

    def __getattr__(self, name):
        return getattr(self.workspace, name)

    def reset(self):
        self.calls, self.arrays = 0, None

    def readback(self, frame):
        self.calls += 1
        self.arrays = self.workspace.readback(frame)
        return self.arrays


def _observe(a, batch, workspace, capture, reader, receipts=None):
    frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
    requests, labels = [], []
    for key in WORLD_ORDER:
        names = workspace.model_metadata[MODEL_BY_WORLD[key]]["body_names"]
        body = names.index("pressure")
        references = {"world": (0.0, 0.0, 0.0)}
        if receipts is not None:
            dynamics = receipts[key].dynamics
            references.update(body_origin=dynamics.body_position[body], com=dynamics.body_com[body])
        for reference_name, reference in references.items():
            for name, other in (("pressure", "plate"), ("plate", "pressure")):
                requests.append(a.BodyContactWrenchRequest(key, name, reference, (other,)))
                labels.append((key, reference_name, name))
    capture.reset()
    measured = reader.measure_many(frame, tuple(requests))
    _require(
        capture.calls == 1 and capture.arrays is not None, "measure_many did not read exactly once"
    )
    _require(frame.revision == batch.checkpoint_revision, "frame revision differs")
    states = batch.integration_snapshot()
    _require(frame.ticks == tuple(states[k].tick for k in WORLD_ORDER), "frame ticks differ")
    for name, value in capture.arrays.items():
        _require(
            np.isfinite(value).all() and not value.flags.writeable,
            "nonfinite or mutable public array: " + name,
        )
    return dict(
        frame=frame,
        arrays=capture.arrays,
        measurements=dict(zip(labels, measured, strict=True)),
        states=states,
        readback_calls=capture.calls,
    )


def _observation_record(observation):
    return dict(
        frame=_frame(observation["frame"]),
        arrays=observation["arrays"],
        states={k: _state(v) for k, v in observation["states"].items()},
        measurements=[
            dict(world=k, reference=r, body=b, value=v)
            for (k, r, b), v in observation["measurements"].items()
        ],
        readback_calls=observation["readback_calls"],
    )


def _stale(workspace, frame):
    try:
        workspace.readback(frame)
    except ValueError:
        return
    raise AssertionError("stale frame was accepted")


def _wrench_oracle(arrays, row, body, reference):
    native_f, native_m, area_f, area_m = (np.zeros(3) for _ in range(4))
    load = 0.0
    for ids, data in zip(
        arrays["entry_contact_ids"][row], arrays["entry_contact_data"][row], strict=True
    ):
        if not ids[0] or body not in ids[4:6]:
            continue
        sign = 1 if body == int(ids[5]) else -1
        axes = np.asarray(data[4:13], dtype=np.float64).reshape(3, 3)
        force = sign * (axes.T @ np.asarray(data[13:16], dtype=np.float64))
        couple = sign * (axes.T @ np.asarray(data[16:19], dtype=np.float64))
        native_f += force
        native_m += couple + np.cross(np.asarray(data[1:4], dtype=np.float64) - reference, force)
        load += float(data[13])
    for ids, data in zip(arrays["area_ids"][row], arrays["area_data"][row], strict=True):
        if not ids[0] or body not in ids[3:5]:
            continue
        _require(ids[5] == 1, "area moment is unavailable")
        sign = 1 if body == int(ids[3]) else -1
        force = sign * float(data[8]) * np.asarray(data[3:6], dtype=np.float64)
        area_f += force
        area_m += np.cross(np.asarray(data[:3], dtype=np.float64) - reference, force)
        load += float(data[8])
    return native_f, native_m, area_f, area_m, load


def _audit_step(batch, workspace, before, receipts, geometry, configs, observed):
    frame, arrays, after = observed["frame"], observed["arrays"], observed["states"]
    result = {}
    for row, key in enumerate(WORLD_ORDER):
        old, new = before[key], after[key]
        _require(new.tick == old.tick + 1 and new.time > old.time, "physics did not advance")
        _require(
            old.q.tobytes() != new.q.tobytes() or old.v.tobytes() != new.v.tobytes(),
            "state did not move",
        )
        _require(
            tuple(arrays["entry_meta"][row, :3]) == (1, old.tick, new.tick), "entry phase differs"
        )
        _require(
            float(arrays["entry_clock"][row, 0]) - float(arrays["entry_clock"][row, 1]) == old.time,
            "input clock differs",
        )
        _require(
            float(arrays["entry_clock"][row, 2]) - float(arrays["entry_clock"][row, 3]) == new.time,
            "output clock differs",
        )
        _exact_array(arrays["entry_q"][row, : old.q.size], old.q.astype(np.float32), "entry q")
        _exact_array(arrays["entry_v"][row, : old.v.size], old.v.astype(np.float32), "entry v")
        _exact_array(arrays["q"][row, : new.q.size], new.q.astype(np.float32), "endpoint q")
        _exact_array(arrays["v"][row, : new.v.size], new.v.astype(np.float32), "endpoint v")
        catalog = frame.area_catalogs[key]
        count = int(arrays["area_meta"][row, 1])
        if key == 7:
            _require(
                catalog == () and count == 0 and not np.any(arrays["area_meta"][row]),
                "native world has area samples",
            )
            _require(arrays["area_declared_mask"][row, 0] == 0, "native world declared area")
        else:
            _require(len(catalog) == 1 and count > 0, "area catalog or samples missing")
            raw = receipts[key].raw_contact_fields["area_pressure"][0]
            cat, surface = catalog[0], geometry[key][0].surface
            _require(cat["identity"] == raw["identity"] == surface.identity, "identity differs")
            _require(cat["stamp"] == raw["stamp"] == surface.stamp, "historical stamp differs")
            stamp = cat["stamp"]
            _require(
                (stamp.world_id, stamp.step_index, stamp.sample_time_s, stamp.configuration_id)
                == (str(key), old.tick, old.time, configs[key]),
                "fresh input stamp differs",
            )
            _require(
                cat["sample_ids"] == raw["sample_ids"] and cat["source_ids"] == raw["source_ids"],
                "sample/source catalog differs",
            )
            _require(len(cat["sample_ids"]) == count, "area sample count differs")
            ids = arrays["area_ids"][row, :count]
            _require(
                np.all(ids[:, 0] == 1)
                and np.all(ids[:, 1] == 0)
                and np.array_equal(ids[:, 2], np.arange(count))
                and np.all(ids[:, 3:5] == raw["body_ids"])
                and np.all(ids[:, 5] == 1),
                "area sample/body addresses differ",
            )
            words = np.ascontiguousarray(arrays["area_measure_words"][row, :count], dtype="<i4")
            areas = np.frombuffer(words.tobytes(), dtype="<f8")
            _require(np.all(areas > 0), "nonpositive dA")
            _exact_array(areas, np.asarray(raw["source_quadrature"].dA, dtype="<f8"), "original dA")
            geometric_area = sum(
                float(
                    np.linalg.norm(
                        np.cross(t.vertices[1] - t.vertices[0], t.vertices[2] - t.vertices[0])
                    )
                )
                / 2
                for t in surface.triangles
            )
            np.testing.assert_allclose(float(areas.sum()), geometric_area, rtol=0, atol=1e-12)
            np.testing.assert_allclose(geometric_area, 0.0004, rtol=0, atol=1e-12)
            _require(int(arrays["entry_meta"][row, 3]) == 0, "duplicate native load in area model")
            _require(arrays["area_declared_mask"][row, 0] == 1, "area declaration mask differs")
            np.testing.assert_allclose(
                arrays["area_data"][row, :count, 8], raw["normal_force"], rtol=0, atol=0
            )
            result[str(key)] = dict(
                area_m2=float(areas.sum()), samples=count, input_configuration_id=configs[key]
            )
        names = workspace.model_metadata[MODEL_BY_WORLD[key]]["body_names"]
        for ref in ("world", "body_origin", "com"):
            pair = []
            for body in ("pressure", "plate"):
                measured = observed["measurements"][(key, ref, body)]
                _require(measured is not None, "accepted step lacks measurement")
                _require(
                    (measured.input_tick, measured.output_tick, measured.time_s)
                    == (old.tick, new.tick, old.time),
                    "wrench phase differs",
                )
                nf, nm, af, am, load = _wrench_oracle(
                    arrays, row, names.index(body), np.asarray(measured.reference_point_world)
                )
                for name, expected, tolerance in (
                    ("native_force_world", nf, 0.002),
                    ("native_moment_world", nm, 1e-5),
                    ("area_force_world", af, 0.002),
                    ("area_moment_world", am, 1e-5),
                    ("total_force_world", nf + af, 0.002),
                    ("total_moment_world", nm + am, 1e-5),
                ):
                    np.testing.assert_allclose(
                        getattr(measured, name), expected, rtol=0.002, atol=tolerance
                    )
                np.testing.assert_allclose(
                    measured.group_force_world, measured.total_force_world, rtol=0, atol=0
                )
                np.testing.assert_allclose(
                    measured.group_moment_world, measured.total_moment_world, rtol=0, atol=0
                )
                np.testing.assert_allclose(
                    measured.group_normal_loads, (load,), rtol=0.002, atol=0.002
                )
                _require(load > 0, "missing actual contact normal load")
                pair.append(measured)
            np.testing.assert_allclose(
                np.asarray(pair[0].total_force_world) + pair[1].total_force_world,
                0,
                rtol=0,
                atol=0.002,
            )
            np.testing.assert_allclose(
                np.asarray(pair[0].total_moment_world) + pair[1].total_moment_world,
                0,
                rtol=0,
                atol=1e-5,
            )
    return result


def _catalog_replay(actual, expected):
    """Compare identity/history fields; return only the explicit changed tokens."""
    changes = {}
    for key in WORLD_ORDER:
        left, right = actual[key], expected[key]
        _require(len(left) == len(right), "replay catalog pair count differs")
        for a, b in zip(left, right, strict=True):
            for name in ("identity", "sample_ids", "source_ids"):
                _require(a[name] == b[name], "replay catalog field differs: " + name)
            sa, sb = a["stamp"], b["stamp"]
            _require(
                (sa.world_id, sa.step_index, sa.sample_time_s)
                == (sb.world_id, sb.step_index, sb.sample_time_s),
                "replay historical phase differs",
            )
            _require(
                sa.configuration_id != sb.configuration_id,
                "replay reused a retired owner configuration ID",
            )
            changes[str(key)] = dict(
                original_configuration_id=sb.configuration_id,
                replay_configuration_id=sa.configuration_id,
            )
    return changes


def run(site, output):
    site, output = Path(site), Path(output)
    provenance = _guard_before(site, output)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        schema="phymeo-area-pressure-wheel-v1",
        status="started",
        provenance=provenance,
        h=H,
        world_order=WORLD_ORDER,
        world_models=MODEL_BY_WORLD,
        physical_calls=0,
        completed_calls=0,
        geometry="TYX FP32 candidates plus HOST exact refinement/fallback",
        friction="none",
        scope="installed production normal-area path; independent friction experiments not imported",
    )
    started, primary, origin_failure = perf_counter(), None, None
    intersector, workspace = None, None
    try:
        a = _api()
        summary["initial_loaded_origins"] = _origins(site)
        with gzip.open(output / "events.jsonl.gz", "xb") as journal, ExitStack() as stack:
            specs, worlds, assets = _declarations(a)
            summary["source"] = {
                s.key: {
                    n: getattr(s.compilation.source, n)
                    for n in ("primary_sha256", "archive_sha256", "model_sha256", "mujoco_version")
                }
                for s in specs
            }
            bundle_bytes = a.AreaPressureAssetBundle.from_assets(specs, assets).to_bytes()
            (output / "assets.bin").write_bytes(bundle_bytes)
            bundle = a.AreaPressureAssetBundle.from_bytes(bundle_bytes)
            _require(bundle.to_bytes() == bundle_bytes, "asset capsule changed")
            loaded = bundle.assets_for(specs)
            _require(_plain(loaded) == _plain(assets), "loaded assets changed numerical bytes")
            _event(
                journal,
                dict(
                    event="declarations",
                    xml={AREA_MODEL: AREA_XML, NATIVE_MODEL: NATIVE_XML},
                    assets=assets,
                    loaded_assets=loaded,
                    worlds=worlds,
                    source=summary["source"],
                ),
            )
            intersector = stack.enter_context(
                a.TyxAreaPressureIntersector(mode="cuda", capacity=64, fallback="exact_host")
            )
            batch = stack.enter_context(a.MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda"))
            scene = a.MuJoCoTyxHeterogeneousAreaPressureScene(
                batch, loaded, intersector=intersector
            )
            workspace = stack.enter_context(
                batch.prepare_device_workspace(
                    include_bodies=True, include_contacts=True, include_area_pressure=True
                )
            )
            capture = _ReadbackCapture(workspace)
            reader = a.BodyContactWrenchReader(
                capture, area_identities={AREA_MODEL: (loaded[AREA_MODEL][0].identity,)}
            )
            summary["asset_sha256"], summary["layout"] = scene.asset_sha256, batch.layout
            summary["model_metadata"] = workspace.model_metadata
            initial = _observe(a, batch, workspace, capture, reader)
            _require(
                all(v is None for v in initial["measurements"].values()),
                "initial history is not empty",
            )
            _event(journal, dict(event="initial", **_observation_record(initial)))

            def physical(label):
                begin = perf_counter()
                before = batch.integration_snapshot()
                configs = {k: batch.area_pressure_configuration_id(k) for k in (41, 305)}
                revision = batch.checkpoint_revision
                summary["physical_calls"] += 1
                accepted = scene.step(H, expected_revision=revision)
                receipts, geometry = batch.last_step.receipts, scene.last_geometry
                # Save the accepted physical prefix before the observation/audit can fail.
                _event(
                    journal,
                    dict(
                        event="accepted",
                        label=label,
                        accepted=accepted,
                        source_revision=revision,
                        output_revision=batch.checkpoint_revision,
                        input={k: _state(v) for k, v in before.items()},
                        output={k: _state(v) for k, v in batch.integration_snapshot().items()},
                        receipts=receipts,
                        geometry=geometry,
                        input_configuration_ids=configs,
                    ),
                )
                _require(
                    dict(accepted) == dict.fromkeys(WORLD_ORDER, True), "partial batch acceptance"
                )
                observed = _observe(a, batch, workspace, capture, reader, receipts)
                _event(
                    journal, dict(event="observation", label=label, **_observation_record(observed))
                )
                audit = _audit_step(batch, workspace, before, receipts, geometry, configs, observed)
                _require(
                    observed["frame"].revision == revision + 1, "step revision increment differs"
                )
                summary["completed_calls"] += 1
                _event(
                    journal,
                    dict(
                        event="audit", label=label, result=audit, elapsed_s=perf_counter() - begin
                    ),
                )
                return observed

            first = physical("step1")
            checkpoint_bytes = scene.checkpoint().to_bytes()
            (output / "scene-checkpoint.bin").write_bytes(checkpoint_bytes)
            checkpoint = a.MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(checkpoint_bytes)
            _require(
                checkpoint.to_bytes() == checkpoint_bytes
                and checkpoint.asset_sha256 == scene.asset_sha256,
                "scene capsule binding differs",
            )
            second = physical("step2")
            revision = batch.checkpoint_revision
            restored_boundary = scene.restore(checkpoint, expected_revision=revision)
            _require(
                batch.checkpoint_revision == revision + 1 and not scene.last_geometry,
                "restore publication differs",
            )
            _stale(workspace, second["frame"])
            restored = _observe(a, batch, workspace, capture, reader)
            _event(
                journal,
                dict(event="restored", boundary=restored_boundary, **_observation_record(restored)),
            )
            _exact_states(restored["states"], first["states"])
            _exact_arrays(restored["arrays"], first["arrays"])
            _require(
                _plain(restored["frame"].area_catalogs) == _plain(first["frame"].area_catalogs),
                "restore changed historical configuration IDs",
            )
            _require(
                all(
                    a0 != b0 and a0[:1] == b0[:1] and a0[1] == b0[1] + 1 and a0[2] == "restore"
                    for a0, b0 in zip(
                        restored["frame"].world_boundaries,
                        second["frame"].world_boundaries,
                        strict=True,
                    )
                ),
                "restore world boundary tokens differ",
            )
            replay = physical("step2-replay")
            _exact_states(replay["states"], second["states"])
            _exact_arrays(replay["arrays"], second["arrays"])
            summary["replay_tokens"] = _catalog_replay(
                replay["frame"].area_catalogs, second["frame"].area_catalogs
            )
            _require(
                replay["frame"].binding == second["frame"].binding
                and replay["frame"].ticks == second["frame"].ticks
                and replay["frame"].revision > second["frame"].revision
                and replay["frame"].epoch > second["frame"].epoch
                and replay["frame"].world_boundaries == restored["frame"].world_boundaries,
                "replay operational frame binding differs",
            )
            reset_boundary = scene.reset((305,))
            _stale(workspace, replay["frame"])
            reset = _observe(a, batch, workspace, capture, reader)
            _event(
                journal,
                dict(event="selected-reset", boundary=reset_boundary, **_observation_record(reset)),
            )
            _exact_states(reset["states"], replay["states"], (41, 7))
            _exact_states(reset["states"], initial["states"], (305,))
            _exact_arrays(reset["arrays"], replay["arrays"], rows=(0, 1))
            _require(reset["frame"].area_catalogs[305] == (), "reset retained selected history")
            _require(
                not np.any(reset["arrays"]["entry_meta"][2])
                and not np.any(reset["arrays"]["area_meta"][2]),
                "reset entry is still valid",
            )
            for key in (41, 7):
                _require(
                    _plain(reset["frame"].area_catalogs[key])
                    == _plain(replay["frame"].area_catalogs[key]),
                    "reset changed unselected history",
                )
            for (key, ref, body), value in reset["measurements"].items():
                _require(
                    value is None
                    if key == 305
                    else value == replay["measurements"][(key, ref, body)],
                    "reset wrench history differs",
                )
            old_boundaries, new_boundaries = (
                replay["frame"].world_boundaries,
                reset["frame"].world_boundaries,
            )
            _require(
                new_boundaries[:2] == old_boundaries[:2]
                and new_boundaries[2][0] == old_boundaries[2][0]
                and new_boundaries[2][1] == old_boundaries[2][1] + 1
                and new_boundaries[2][2] == "reset",
                "selected boundary tokens differ",
            )
            final = physical("post-reset")
            _require(final["frame"].ticks == (3, 3, 1), "final ticks differ")
            _require(
                final["frame"].world_boundaries == reset["frame"].world_boundaries,
                "ordinary step changed boundary lineage",
            )
            _require(
                summary["physical_calls"] == summary["completed_calls"] == 4,
                "wrong actual step count",
            )
            summary["intersector_stats"], summary["workspace_counters"] = (
                intersector.stats,
                workspace.counters,
            )
            summary["final_states"] = {k: _state(v) for k, v in final["states"].items()}
            summary["checks"] = dict(
                asset_bytes=True,
                replay_all_array_bytes=True,
                same_mode_state_bytes=True,
                selected_reset=True,
                fresh_stamps=True,
                one_readback_per_measure_many=True,
                independent_wrenches=True,
            )
        summary["status"] = "passed"
    except BaseException as error:
        primary = error
        summary["status"] = "failed"
        summary["error"] = dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()
        )
    finally:
        if intersector is not None:
            summary["intersector_stats_after_close"] = intersector.stats
        if workspace is not None:
            summary["workspace_counters_after_close"] = workspace.counters
        try:
            summary["loaded_origins"] = _origins(site)
        except BaseException as error:
            origin_failure = error
            summary["status"] = "failed"
            summary["origin_failure"] = dict(type=type(error).__name__, message=str(error))
        summary["total_s"] = perf_counter() - started
        summary["files"] = {
            p.name: dict(bytes=p.stat().st_size, sha256=_sha(p))
            for p in sorted(output.iterdir())
            if p.is_file()
        }
        _write_json(output / "summary.json", summary)
    if primary is not None:
        raise primary
    if origin_failure is not None:
        raise origin_failure
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.site, args.output)
    print(
        json.dumps(
            dict(
                status=result["status"],
                physical_calls=result["physical_calls"],
                output=str(args.output),
            )
        )
    )


if __name__ == "__main__":
    main()
