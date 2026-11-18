"""Portable, bounded declarations for existing body-local area-pressure assets.

This is a checksum-protected trusted artifact, not an authentication format.
Coordinates are MuJoCo body-local metres, not COM/inertial or URDF-link frames;
nodal pressures are Pa, not elastic moduli. No scaling, material inference,
native model construction, collision query or dynamics occurs here. A valid
asset can still be refused by a later query's gradient or capacity gates.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from types import MappingProxyType

from ..builders.mujoco_compiler import MuJoCoCompilation, MuJoCoSource
from ..geometry.area_pressure import AreaPressureIdentity
from ..geometry.area_pressure_intersection import (
    MAX_CELLS,
    MAX_NODES,
    MAX_RIGID_FACES,
    RigidPressureMesh,
    TetraPressureMesh,
    _owned_mesh_input,
)
from .mujoco_area_pressure_scene import BodyAreaPressureAsset
from .mujoco_checkpoint import _decode, _encode
from .mujoco_heterogeneous import MAX_MODELS, MuJoCoTyxModelSpec
from .mujoco_heterogeneous_area_scene import _asset_digest

MAGIC = b"PHYMEOBODYAREAASSETS1\n"
MAX_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 16
MAX_JSON_NODES = 262144
MAX_PAIRS = 16
SCHEMA = "phymeo-body-area-assets-v1"
LAW = "frozen-surface-elastic-backward-euler-v1"
_IDENTITY = ("compliant_body", "rigid_body", "compliant_material", "rigid_material")
_SOURCE = ("primary_sha256", "archive_sha256", "model_sha256", "mujoco_version")
_TOP = (
    "schema",
    "coordinates",
    "units",
    "pressure_domain",
    "law",
    "friction",
    "models",
    "asset_sha256",
)


def _require(condition, message):
    if not condition:
        raise ValueError("area pressure asset bundle: " + message)


def _fields(value, names, name):
    _require(type(value) is dict and set(value) == set(names), name + " fields")


def _name(value, name):
    _require(type(value) is str and 1 <= len(value) <= 128, name + " string")
    return value


def _sha(value, name):
    _require(
        type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
        name + " SHA256",
    )


def _ids(value, cap, name):
    _require(type(value) is list and 1 <= len(value) <= cap, name + " count")
    for item in value:
        _name(item, name)
    _require(len(set(value)) == len(value), name + " duplicates")


def _json_budget(raw):
    """Bound allocation before json.loads; syntax is checked by the JSON parser."""
    depth = nodes = 0
    quoted = escaped = primitive = False
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
            continue
        if byte == 34:
            quoted, primitive = True, False
            nodes += 1
        elif byte in (123, 91):
            depth += 1
            nodes += 1
            primitive = False
            _require(depth <= MAX_JSON_DEPTH, "JSON depth limit")
        elif byte in (125, 93):
            depth -= 1
            primitive = False
            _require(depth >= 0, "JSON nesting")
        elif byte in (9, 10, 13, 32, 44, 58):
            primitive = False
        elif not primitive:
            nodes += 1
            primitive = True
        _require(nodes <= MAX_JSON_NODES, "JSON node limit")
    _require(not quoted and depth == 0, "JSON nesting")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _bad_number(value):
    raise ValueError("area pressure asset bundle: nonfinite JSON " + value)


def _dumps(value):
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("ascii")


def _envelope(value):
    payload = _dumps(value)
    _require(len(MAGIC) + 65 + len(payload) <= MAX_BYTES, "serialized byte limit")
    _json_budget(payload)
    return MAGIC + hashlib.sha256(payload).hexdigest().encode("ascii") + b"\n" + payload


def _array_layout(value, dtype, shape, name):
    _fields(value, ("$array", "shape", "data"), name)
    _require(type(value["$array"]) is str and value["$array"] == dtype, name + " dtype")
    actual = value["shape"]
    _require(
        type(actual) is list and all(type(n) is int for n in actual) and actual == list(shape),
        name + " shape",
    )
    size = 1
    for dimension in shape:
        size *= dimension
    _require(
        type(value["data"]) is str and len(value["data"]) == 4 * ((8 * size + 2) // 3),
        name + " encoded byte count",
    )


def _preflight(value):
    """All fields, counts and array extents before any array or exact geometry."""
    _fields(value, _TOP, "bundle")
    fixed = {
        "schema": SCHEMA,
        "coordinates": "mujoco-body-local",
        "pressure_domain": "full-zero-boundary",
        "law": LAW,
        "friction": "none",
    }
    for key, expected in fixed.items():
        _require(type(value[key]) is str and value[key] == expected, key)
    _fields(value["units"], ("length", "pressure"), "units")
    _require(value["units"] == {"length": "m", "pressure": "Pa"}, "units")
    _sha(value["asset_sha256"], "asset")
    models = value["models"]
    _require(type(models) is dict and 1 <= len(models) <= MAX_MODELS, "model count")
    for key, model in models.items():
        _name(key, "model key")
        _fields(model, ("source", "pairs"), "model")
        _fields(model["source"], _SOURCE, "source")
        for name in _SOURCE[:-1]:
            _sha(model["source"][name], name)
        _name(model["source"]["mujoco_version"], "mujoco_version")
        pairs = model["pairs"]
        _require(type(pairs) is list and 1 <= len(pairs) <= MAX_PAIRS, "pair count")
        seen = set()
        for pair in pairs:
            _fields(pair, ("identity", "pressure", "rigid"), "pair")
            identity = pair["identity"]
            _fields(identity, _IDENTITY, "identity")
            for name in _IDENTITY:
                _name(identity[name], name)
            a, b = identity["compliant_body"], identity["rigid_body"]
            _require(a != b and frozenset((a, b)) not in seen, "duplicate or self body pair")
            seen.add(frozenset((a, b)))
            pressure, rigid = pair["pressure"], pair["rigid"]
            _fields(
                pressure,
                ("node_ids", "vertices_local", "cells", "nodal_pressure", "cell_ids"),
                "pressure",
            )
            _fields(rigid, ("node_ids", "vertices_local", "faces", "face_ids"), "rigid")
            _ids(pressure["node_ids"], MAX_NODES, "pressure node_ids")
            _ids(pressure["cell_ids"], MAX_CELLS, "cell_ids")
            _ids(rigid["node_ids"], MAX_NODES, "rigid node_ids")
            _ids(rigid["face_ids"], MAX_RIGID_FACES, "face_ids")
            na, nb = len(pressure["node_ids"]), len(rigid["node_ids"])
            _array_layout(pressure["vertices_local"], "<f8", (na, 3), "pressure vertices")
            _array_layout(pressure["nodal_pressure"], "<f8", (na,), "nodal_pressure")
            _array_layout(pressure["cells"], "<i8", (len(pressure["cell_ids"]), 4), "cells")
            _array_layout(rigid["vertices_local"], "<f8", (nb, 3), "rigid vertices")
            _array_layout(rigid["faces"], "<i8", (len(rigid["face_ids"]), 3), "faces")


def _load(raw):
    _require(isinstance(raw, (bytes, bytearray, memoryview)), "expected bytes")
    size = raw.nbytes if isinstance(raw, memoryview) else len(raw)
    _require(size <= MAX_BYTES, "serialized byte limit")
    owned = bytes(raw)
    start = len(MAGIC)
    _require(
        owned.startswith(MAGIC)
        and len(owned) > start + 65
        and owned[start + 64 : start + 65] == b"\n",
        "format header",
    )
    payload = owned[start + 65 :]
    _require(
        owned[start : start + 64] == hashlib.sha256(payload).hexdigest().encode("ascii"),
        "payload checksum",
    )
    _json_budget(payload)
    try:
        value = json.loads(payload, object_pairs_hook=_unique, parse_constant=_bad_number)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("area pressure asset bundle: invalid JSON") from error
    _preflight(value)
    return owned, value


def _restore(value):
    # Only these five preflighted array positions enter the old array decoder.
    # In particular, no untrusted mapping or $record is passed to _decode.
    decoded = {}
    for key, model in value["models"].items():
        rows = []
        for pair in model["pairs"]:
            a, b = pair["pressure"], pair["rigid"]
            rows.append(
                (
                    pair["identity"],
                    dict(
                        node_ids=tuple(a["node_ids"]),
                        world_vertices=_decode(a["vertices_local"]),
                        cells=_decode(a["cells"]),
                        nodal_pressure=_decode(a["nodal_pressure"]),
                        cell_ids=tuple(a["cell_ids"]),
                        pressure_domain="full",
                    ),
                    dict(
                        node_ids=tuple(b["node_ids"]),
                        world_vertices=_decode(b["vertices_local"]),
                        faces=_decode(b["faces"]),
                        face_ids=tuple(b["face_ids"]),
                    ),
                )
            )
        decoded[key] = rows
    # Every array has now passed decoding and finite-value checks, including the
    # last model. Public fields, never private mesh caches, define the geometry.
    result = {}
    for key, rows in decoded.items():
        result[key] = tuple(
            BodyAreaPressureAsset(
                AreaPressureIdentity(**identity), TetraPressureMesh(**a), RigidPressureMesh(**b)
            )
            for identity, a, b in rows
        )
    return MappingProxyType(result)


def _source_binding(compilation):
    _require(type(compilation) is MuJoCoCompilation, "expected MuJoCoCompilation")
    source = compilation.source
    _require(type(source) is MuJoCoSource, "expected immutable MuJoCoSource")
    result = {}
    for field, name in (
        ("primary_xml", "primary_sha256"),
        ("archive_bytes", "archive_sha256"),
        ("model_binary", "model_sha256"),
    ):
        raw, recorded = getattr(source, field), getattr(source, name)
        _require(type(raw) is bytes and bool(raw), "source bytes " + field)
        _sha(recorded, name)
        actual = hashlib.sha256(raw).hexdigest()
        _require(recorded == actual, "source bytes disagree with " + name)
        result[name] = actual
    result["mujoco_version"] = _name(source.mujoco_version, "mujoco_version")
    return result


def _models(model_specs):
    _require(
        type(model_specs) in (list, tuple) and 1 <= len(model_specs) <= MAX_MODELS,
        "model_specs must contain 1..32 models",
    )
    specs = tuple(model_specs)
    seen, area = set(), {}
    for spec in specs:
        _require(type(spec) is MuJoCoTyxModelSpec, "expected MuJoCoTyxModelSpec")
        _require(
            not spec.area_friction_laws,
            "v1 asset bundle declares friction none; use explicit live law/asset declarations",
        )
        key = _name(spec.key, "model key")
        _require(key not in seen, "duplicate model key")
        seen.add(key)
        _require(type(spec.compilation) is MuJoCoCompilation, "expected MuJoCoCompilation")
        pairs = spec.area_pressure_pairs
        _require(type(pairs) is tuple and len(pairs) <= MAX_PAIRS, "model pair count")
        unique = set()
        for pair in pairs:
            _require(type(pair) is tuple and len(pair) == 2, "model body pair")
            a, b = (_name(name, "body name") for name in pair)
            _require(a != b and frozenset((a, b)) not in unique, "duplicate or self model pair")
            unique.add(frozenset((a, b)))
        if pairs:
            area[key] = (_source_binding(spec.compilation), pairs)
    _require(bool(area), "at least one area model is required")
    return area


def _match(models, value):
    _require(set(models) == set(value["models"]), "declared area model keys")
    for key, (source, pairs) in models.items():
        saved = value["models"][key]
        _require(saved["source"] == source, "source/model/version binding for " + key)
        actual = tuple(
            (row["identity"]["compliant_body"], row["identity"]["rigid_body"])
            for row in saved["pairs"]
        )
        _require(actual == pairs, "ordered model body pairs for " + key)


def _mesh_wire(mesh, *, pressure):
    expected = TetraPressureMesh if pressure else RigidPressureMesh
    _require(isinstance(mesh, expected), "expected declared mesh type")
    fields = (
        ("node_ids", "world_vertices", "cells", "nodal_pressure", "cell_ids", "pressure_domain")
        if pressure
        else ("node_ids", "world_vertices", "faces", "face_ids")
    )
    raw = _owned_mesh_input({name: getattr(mesh, name) for name in fields}, pressure=pressure)
    if pressure:
        _require(raw["pressure_domain"] == "full", "requires full zero-boundary pressure")
    result = {"node_ids": list(raw["node_ids"]), "vertices_local": _encode(raw["world_vertices"])}
    for name in ("cells", "nodal_pressure", "cell_ids") if pressure else ("faces", "face_ids"):
        result[name] = list(raw[name]) if name.endswith("_ids") else _encode(raw[name])
    return result


@dataclass(frozen=True, init=False)
class AreaPressureAssetBundle:
    """Own bytes only; each assets_for call reconstructs new, fully checked assets.

    Loading checks complete local geometry, but does not establish a runtime
    model's body existence, query g>0, shared-face gradient or contact budget.
    Scene/runtime admission and per-pose intersection gates remain independent.
    """

    _bytes: bytes

    def __init__(self, raw):
        owned, value = _load(raw)
        assets = _restore(value)
        _require(_asset_digest(assets) == value["asset_sha256"], "asset digest")
        object.__setattr__(self, "_bytes", owned)

    @classmethod
    def from_bytes(cls, raw):
        """Validate the entire envelope and exact geometry immediately."""
        return cls(raw)

    @classmethod
    def from_assets(cls, model_specs, assets):
        """Capture current public declarations; caller caches are never trusted."""
        models = _models(model_specs)
        _require(isinstance(assets, Mapping) and len(assets) == len(models), "asset model count")
        entries = tuple(islice(assets.items(), MAX_MODELS + 1))
        _require(len(entries) == len(models), "asset model count")
        _require(all(type(key) is str for key, _ in entries), "asset model keys")
        captured = dict(entries)
        _require(len(captured) == len(entries) and set(captured) == set(models), "asset model keys")
        value = dict(
            schema=SCHEMA,
            coordinates="mujoco-body-local",
            units={"length": "m", "pressure": "Pa"},
            pressure_domain="full-zero-boundary",
            law=LAW,
            friction="none",
            models={},
            asset_sha256="0" * 64,
        )
        encoded_budget = len(MAGIC) + 65 + len(_dumps(value))
        for key in sorted(models):
            rows = captured[key]
            source, pairs = models[key]
            _require(type(rows) is tuple and len(rows) == len(pairs), "ordered asset pair count")
            saved = {"source": source, "pairs": []}
            encoded_budget += len(_dumps(key)) + len(_dumps(saved)) + 1 + bool(value["models"])
            for item in rows:
                _require(isinstance(item, BodyAreaPressureAsset), "expected BodyAreaPressureAsset")
                identity = item.identity
                _require(isinstance(identity, AreaPressureIdentity), "expected area identity")
                row = dict(
                    identity={name: _name(getattr(identity, name), name) for name in _IDENTITY},
                    pressure=_mesh_wire(item.pressure_local, pressure=True),
                    rigid=_mesh_wire(item.rigid_local, pressure=False),
                )
                encoded_budget += len(_dumps(row)) + bool(saved["pairs"])
                _require(encoded_budget <= MAX_BYTES, "serialized byte limit")
                saved["pairs"].append(row)
            value["models"][key] = saved
        # The provisional digest is a fixed-size placeholder. Serialization,
        # whole-package JSON/layout bounds and source/pair binding precede any
        # exact geometry. Only complete success can publish the owned capsule.
        _, value = _load(_envelope(value))
        _match(models, value)
        owned_assets = _restore(value)
        value["asset_sha256"] = _asset_digest(owned_assets)
        result = object.__new__(cls)
        object.__setattr__(result, "_bytes", _envelope(value))
        return result

    def to_bytes(self):
        return self._bytes

    def assets_for(self, model_specs):
        """Match area models exactly; unrelated additional native models may exist."""
        models = _models(model_specs)
        _, value = _load(self._bytes)
        _match(models, value)
        assets = _restore(value)
        _require(_asset_digest(assets) == value["asset_sha256"], "asset digest")
        return assets
