"""Bounded, shared decoding of local authored glTF 2.0 triangle geometry.

Supported: UTF-8 .gltf JSON, one scene / identity node / mesh / indexed TRIANGLES
primitive, one embedded base64 buffer, FLOAT VEC3 POSITION and unit NORMAL, and
unsigned byte/short/int indices. Vertex views may have aligned offsets/strides.
Coordinates stay in the declared metre frame; no axis conversion is performed.
Materials belong to VisualScene instances, so embedded materials are rejected.

Nonidentity transforms, extra attributes, multiple primitives, external buffers,
textures, extensions, animation, skins, morph targets and GLB are unsupported,
not silently dropped. Names/extras are descriptive metadata only. File bytes are
limited to 8 MiB, decoded data to 4 MiB, vertices and triangles to 65,536 each,
JSON nesting to 64 and structural punctuation to 4,096 tokens. The result owns
immutable values and records SHA256 of the exact file bytes, not just its buffer.
No rendering SDK, device, physics runtime, network fetch or cache is involved.
"""

import base64
import hashlib
import json
import math
import os
import stat
import struct
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .assets import Mesh

_MAX_FILE_BYTES = 8 * 1024 * 1024
_MAX_BUFFER_BYTES = 4 * 1024 * 1024
_MAX_VERTICES = 65_536
_MAX_TRIANGLES = 65_536
_MAX_JSON_DEPTH = 64
_MAX_JSON_TOKENS = 4096
_UNIT_TOLERANCE = 1e-6
_IDENTITY_MATRIX = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)


def _integer(value, label, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, not bool")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{label} outside supported range")
    return value


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{label} must be a finite number")
    try:
        valid = math.isfinite(value)
    except (OverflowError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{label} must be finite and representable")
    return value


def _vector(value, label, size, *, container=tuple):
    if not isinstance(value, container) or len(value) != size:
        raise ValueError(f"{label} must be a {container.__name__} of {size} numbers")
    for item in value:
        _finite_number(item, label)
    return value


@dataclass(frozen=True, slots=True)
class TriangleMesh:
    """Validated metre-frame geometry and its complete source-file digest."""

    positions: tuple[tuple[float, float, float], ...]
    normals: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
    source_sha256: str

    def __post_init__(self):
        for name in ("positions", "normals", "triangles"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be an immutable tuple")
        _integer(len(self.positions), "positions count", 1, _MAX_VERTICES)
        _integer(len(self.triangles), "triangles count", 1, _MAX_TRIANGLES)
        if len(self.normals) != len(self.positions):
            raise ValueError("normals count must equal positions count")
        for position in self.positions:
            _vector(position, "positions", 3)
        for normal in self.normals:
            _vector(normal, "normals", 3)
            if abs(math.hypot(*normal) - 1) > _UNIT_TOLERANCE:
                raise ValueError("normals must have unit length within 1e-6")
        for triangle in self.triangles:
            if not isinstance(triangle, tuple) or len(triangle) != 3:
                raise ValueError("triangles must contain immutable triples")
            for index in triangle:
                _integer(index, "triangle index", 0, len(self.positions) - 1)
            a, b, c = (self.positions[index] for index in triangle)
            u = tuple(y - x for x, y in zip(a, b, strict=True))
            v = tuple(y - x for x, y in zip(a, c, strict=True))
            cross = (
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            )
            area_twice = math.hypot(*cross)
            if not math.isfinite(area_twice) or area_twice == 0:
                raise ValueError("triangle is degenerate or not representable")
        digest = self.source_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("source_sha256 must be a lowercase SHA256 hex digest")


def _local_path(uri):
    if uri != uri.strip():
        raise ValueError("uri must not have surrounding whitespace")
    if uri.startswith(("//", "\\\\")):
        raise ValueError("uri must name a local file, not a network share")
    windows_drive = (
        len(uri) >= 3
        and uri[0].isascii()
        and uri[0].isalpha()
        and uri[1] == ":"
        and uri[2] in "/\\"
    )
    parsed = urlsplit(uri) if not windows_drive else None
    if parsed is not None and parsed.scheme:
        if parsed.scheme != "file" or parsed.netloc.lower() not in ("", "localhost"):
            raise ValueError("uri must use a local path or local file URI")
        if parsed.query or parsed.fragment:
            raise ValueError("file uri query and fragment are unsupported")
        for index, char in enumerate(parsed.path):
            if char == "%" and (
                index + 2 >= len(parsed.path)
                or any(
                    c not in "0123456789abcdefABCDEF" for c in parsed.path[index + 1 : index + 3]
                )
            ):
                raise ValueError("file uri contains invalid percent encoding")
        decoded = unquote(parsed.path, encoding="utf-8", errors="strict")
        if decoded.startswith(("//", "\\\\")):
            raise ValueError("file uri must not name a network share")
        if os.name == "nt" and len(decoded) >= 3 and decoded[0] == "/" and decoded[2] == ":":
            decoded = decoded[1:]
        path = Path(decoded)
        if not path.is_absolute():
            raise ValueError("file uri must name an absolute local path")
    else:
        path = Path(uri)
    if path.suffix.lower() != ".gltf":
        raise ValueError("only local .gltf JSON is supported; GLB is unsupported")
    return path


def _file_bytes(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("mesh source must be a regular file")
    if info.st_size > _MAX_FILE_BYTES:
        raise ValueError("mesh file exceeds byte limit")
    with path.open("rb") as source:
        # Also bounds a file that grows after stat; never use an unbounded read.
        raw = source.read(_MAX_FILE_BYTES + 1)
    if len(raw) > _MAX_FILE_BYTES:
        raise ValueError("mesh file exceeds byte limit")
    return raw


def _check_json_budget(raw):
    depth = tokens = 0
    quoted = escaped = False
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char in b"{}[],:":
            tokens += 1
            if tokens > _MAX_JSON_TOKENS:
                raise ValueError("JSON structure exceeds token limit")
            if char in b"{[":
                depth += 1
                if depth > _MAX_JSON_DEPTH:
                    raise ValueError("JSON nesting exceeds depth limit")
            elif char in b"}]":
                depth -= 1


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key[:80]!r}")
        result[key] = value
    return result


def _json_number(text, *, integer=False):
    if len(text) > 128:
        raise ValueError("JSON number exceeds supported length")
    if integer:
        return int(text)
    # Preserve exact identity-transform declarations rather than rounding a
    # nonidentity decimal to 0 or 1 before the supported-subset check.
    value = Decimal(text)
    converted = float(value)
    if not math.isfinite(converted) or (converted == 0 and value != 0):
        raise ValueError("JSON number is not finite and representable")
    return value


def _reject_constant(text):
    raise ValueError(f"nonfinite JSON constant {text!r} is unsupported")


def _object(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        names = ", ".join(repr(name[:80]) for name in sorted(unknown)[:3])
        raise ValueError(f"{label} has unsupported fields: {names}")
    return value


def _required(value, field, label):
    if field not in value:
        raise ValueError(f"{label}.{field} is required")
    return value[field]


def _array(value, label, minimum, maximum):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{label} must be an array with {minimum}..{maximum} entries")
    return value


def _single(document, field):
    return _array(_required(document, field, "glTF"), field, 1, 1)[0]


def _identity_node(node):
    _object(node, ("mesh", "matrix", "translation", "rotation", "scale", "name", "extras"), "node")
    _integer(_required(node, "mesh", "node"), "node.mesh", 0, 0)
    if "matrix" in node:
        if any(field in node for field in ("translation", "rotation", "scale")):
            raise ValueError("node cannot combine matrix with TRS")
        matrix = _vector(node["matrix"], "node.matrix", 16, container=list)
        if tuple(matrix) != _IDENTITY_MATRIX:
            raise ValueError("nonidentity node.matrix is unsupported")
    for field, identity in (("translation", (0, 0, 0)), ("scale", (1, 1, 1))):
        if field in node:
            value = _vector(node[field], f"node.{field}", 3, container=list)
            if tuple(value) != identity:
                raise ValueError(f"nonidentity node.{field} is unsupported")
    if "rotation" in node:
        value = tuple(_vector(node["rotation"], "node.rotation", 4, container=list))
        if value not in ((0, 0, 0, 1), (0, 0, 0, -1)):
            raise ValueError("nonidentity node.rotation is unsupported")


def _embedded_buffer(buffer):
    _object(buffer, ("byteLength", "uri", "name", "extras"), "buffer")
    length = _integer(
        _required(buffer, "byteLength", "buffer"), "buffer.byteLength", 1, _MAX_BUFFER_BYTES
    )
    uri = _required(buffer, "uri", "buffer")
    prefixes = ("data:application/octet-stream;base64,", "data:application/gltf-buffer;base64,")
    if not isinstance(uri, str) or not uri.startswith(prefixes):
        raise ValueError("buffer.uri must contain supported embedded base64 data")
    encoded = uri.split(",", 1)[1]
    if len(encoded) != 4 * ((length + 2) // 3):
        raise ValueError("buffer.byteLength disagrees with encoded data length")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != length:
        raise ValueError("buffer.byteLength disagrees with decoded data length")
    return raw


def _views(values, raw):
    views = _array(values, "bufferViews", 1, 3)
    for index, view in enumerate(views):
        label = f"bufferViews[{index}]"
        _object(
            view,
            ("buffer", "byteOffset", "byteLength", "byteStride", "target", "name", "extras"),
            label,
        )
        _integer(_required(view, "buffer", label), label + ".buffer", 0, 0)
        start = _integer(view.get("byteOffset", 0), label + ".byteOffset", 0, len(raw))
        size = _integer(_required(view, "byteLength", label), label + ".byteLength", 1, len(raw))
        if start + size > len(raw):
            raise ValueError(f"{label} range exceeds buffer")
        if "byteStride" in view:
            stride = _integer(view["byteStride"], label + ".byteStride", 4, 252)
            if stride % 4:
                raise ValueError(f"{label}.byteStride must be a multiple of 4")
        if "target" in view:
            target = _integer(view["target"], label + ".target")
            if target not in (34962, 34963):
                raise ValueError(f"{label}.target is unsupported")
    return views


def _bounds(accessor, values, *, vector, code, label):
    width = 3 if vector else 1
    for field, operation in (("min", min), ("max", max)):
        if field not in accessor:
            continue
        bounds = _vector(accessor[field], label + "." + field, width, container=list)
        actual = (
            tuple(operation(value[i] for value in values) for i in range(3))
            if vector
            else (operation(values),)
        )
        if vector:
            declared = tuple(
                struct.unpack("<f", struct.pack("<f", float(value)))[0] for value in bounds
            )
            if not all(math.isfinite(value) for value in declared):
                raise ValueError(f"{label}.{field} is outside FLOAT range")
        else:
            maximum = {"B": 255, "H": 65535, "I": 2**32 - 1}[code]
            declared = tuple(_integer(value, label + "." + field, 0, maximum) for value in bounds)
        if declared != actual:
            raise ValueError(f"{label}.{field} disagrees with actual accessor data")


def _accessor(index, accessors, views, raw, role):
    label = f"accessors[{index}] ({role})"
    accessor = _object(
        accessors[index],
        (
            "bufferView",
            "byteOffset",
            "componentType",
            "count",
            "type",
            "normalized",
            "min",
            "max",
            "name",
            "extras",
        ),
        label,
    )
    vector = role != "indices"
    component = _integer(_required(accessor, "componentType", label), label + ".componentType")
    if vector and component != 5126:
        raise ValueError(f"{label}.componentType must be FLOAT")
    if not vector and component not in (5121, 5123, 5125):
        raise ValueError(f"{label}.componentType must be an unsigned integer")
    if accessor.get("type") != ("VEC3" if vector else "SCALAR"):
        raise ValueError(f"{label}.type is unsupported")
    if accessor.get("normalized", False) is not False:
        raise ValueError(f"{label}.normalized is unsupported")
    limit = _MAX_VERTICES if vector else 3 * _MAX_TRIANGLES
    count = _integer(_required(accessor, "count", label), label + ".count", 1, limit)
    if not vector and count % 3:
        raise ValueError(f"{label}.count must be a multiple of 3")
    if role == "POSITION" and ("min" not in accessor or "max" not in accessor):
        raise ValueError(f"{label} requires POSITION min and max")
    view_id = _integer(
        _required(accessor, "bufferView", label), label + ".bufferView", 0, len(views) - 1
    )
    view = views[view_id]
    expected_target = 34962 if vector else 34963
    if view.get("target", expected_target) != expected_target:
        raise ValueError(f"{label} bufferView.target does not match accessor role")
    if not vector and "byteStride" in view:
        raise ValueError(f"{label} index bufferView.byteStride is unsupported")
    code = "f" if vector else {5121: "B", 5123: "H", 5125: "I"}[component]
    unpacker = struct.Struct("<" + ("3" if vector else "") + code)
    component_size = 4 if vector else unpacker.size
    offset = _integer(accessor.get("byteOffset", 0), label + ".byteOffset")
    start = view.get("byteOffset", 0) + offset
    stride = view.get("byteStride", unpacker.size)
    if stride < unpacker.size:
        raise ValueError(f"{label} byteStride is smaller than one element")
    if offset % component_size or start % component_size:
        raise ValueError(f"{label}.byteOffset is misaligned")
    needed = (count - 1) * stride + unpacker.size
    if offset + needed > view["byteLength"]:
        raise ValueError(f"{label} range exceeds its bufferView")
    values = tuple(unpacker.unpack_from(raw, start + i * stride) for i in range(count))
    if not vector:
        values = tuple(value[0] for value in values)
    _bounds(accessor, values, vector=vector, code=code, label=label)
    return values, view_id


def _decode(document, raw_sha):
    _object(
        document,
        (
            "asset",
            "scene",
            "scenes",
            "nodes",
            "meshes",
            "buffers",
            "bufferViews",
            "accessors",
            "extras",
        ),
        "glTF",
    )
    asset = _object(
        _required(document, "asset", "glTF"),
        ("version", "minVersion", "generator", "copyright", "extras"),
        "asset",
    )
    if asset.get("version") != "2.0" or asset.get("minVersion", "2.0") != "2.0":
        raise ValueError("asset.version/minVersion must be glTF 2.0")
    _integer(document.get("scene", 0), "scene", 0, 0)
    scene = _object(_single(document, "scenes"), ("nodes", "name", "extras"), "scene")
    roots = _array(_required(scene, "nodes", "scene"), "scene.nodes", 1, 1)
    _integer(roots[0], "scene.nodes[0]", 0, 0)
    _identity_node(_single(document, "nodes"))
    mesh = _object(_single(document, "meshes"), ("primitives", "name", "extras"), "mesh")
    primitive = _object(
        _array(_required(mesh, "primitives", "mesh"), "primitives", 1, 1)[0],
        ("attributes", "indices", "mode", "extras"),
        "primitive",
    )
    _integer(primitive.get("mode", 4), "primitive.mode", 4, 4)
    attributes = _object(
        _required(primitive, "attributes", "primitive"), ("POSITION", "NORMAL"), "attributes"
    )
    accessors = _array(_required(document, "accessors", "glTF"), "accessors", 3, 3)
    position_id = _integer(
        _required(attributes, "POSITION", "attributes"), "POSITION accessor", 0, 2
    )
    normal_id = _integer(_required(attributes, "NORMAL", "attributes"), "NORMAL accessor", 0, 2)
    index_id = _integer(_required(primitive, "indices", "primitive"), "indices accessor", 0, 2)
    if {position_id, normal_id, index_id} != {0, 1, 2}:
        raise ValueError("POSITION, NORMAL and indices must use three distinct accessors")
    raw = _embedded_buffer(_single(document, "buffers"))
    views = _views(_required(document, "bufferViews", "glTF"), raw)
    positions, position_view = _accessor(position_id, accessors, views, raw, "POSITION")
    normals, normal_view = _accessor(normal_id, accessors, views, raw, "NORMAL")
    indices, index_view = _accessor(index_id, accessors, views, raw, "indices")
    if index_view in (position_view, normal_view):
        raise ValueError("indices and vertex attributes must use separate bufferViews")
    if {position_view, normal_view, index_view} != set(range(len(views))):
        raise ValueError("unreferenced bufferViews are unsupported")
    triangles = tuple(tuple(indices[i : i + 3]) for i in range(0, len(indices), 3))
    return TriangleMesh(positions, normals, triangles, raw_sha)


def load_mesh(mesh: Mesh) -> TriangleMesh:
    """Load the documented local subset, or raise ValueError / an OS I/O error.

    Structural and unsupported-feature errors identify the Mesh and field.
    Missing/inaccessible files retain their OS error and filename. No resource
    outside the one local glTF file is opened; its buffer must be embedded.
    """
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be Mesh")
    try:
        raw = _file_bytes(_local_path(mesh.uri))
        _check_json_budget(raw)
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_int=lambda value: _json_number(value, integer=True),
            parse_float=_json_number,
            parse_constant=_reject_constant,
        )
        return _decode(document, hashlib.sha256(raw).hexdigest())
    except (ValueError, TypeError, OverflowError, struct.error) as error:
        raise ValueError(f"Mesh {mesh.id!r}: {error}") from error
