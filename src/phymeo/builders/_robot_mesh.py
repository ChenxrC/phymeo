"""Local indexed triangular OBJ data for the bounded robot collision importers."""

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from ..constraints.collision_shapes import MAX_MESH_TRIANGLES, MAX_VERTICES
from ..dynamics.rotation import quaternion_matrix
from ._robot_xml import _attributes, _child, _children, _name, _numbers


@dataclass(frozen=True)
class LocalTriangleMesh:
    """Authored position IDs and oriented faces, before the referencing link's geom pose."""

    vertices: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]


def _local(root, value, where):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{where} requires a relative local file or directory")
    text = str(value).strip().replace("\\", "/")
    if (
        ":" in text
        or "\x00" in text
        or PureWindowsPath(text).anchor
        or PurePosixPath(text).is_absolute()
    ):
        raise ValueError(f"{where} accepts only relative local paths, not URI or absolute paths")
    try:
        path = (root / text).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"{where} cannot resolve local path {text!r}") from exc
    if not path.is_relative_to(root):
        raise ValueError(f"{where} path escapes outside the local asset directory")
    return path


def _vector(value, count, where):
    try:
        data = np.asarray(value)
        if data.dtype.kind not in "iuf":
            raise ValueError
        data = np.asarray(data, dtype=float)
    except (TypeError, OverflowError, ValueError) as exc:
        raise ValueError(f"{where} requires {count} finite numbers") from exc
    if data.shape != (count,) or not np.isfinite(data).all():
        raise ValueError(f"{where} requires {count} finite numbers")
    return data


def _corner(token, counts, where):
    parts = token.split("/")
    if len(parts) > 3 or not parts[0] or (len(parts) > 1 and not parts[-1]):
        raise ValueError(f"{where} has an invalid OBJ triangle index {token!r}")
    indices = []
    for i, word in enumerate(parts):
        if not word and i == 1 and len(parts) == 3:
            continue
        if not re.fullmatch(r"[+-]?[0-9]+", word):
            raise ValueError(f"{where} has an invalid OBJ index {token!r}")
        try:
            number = int(word)
        except ValueError as exc:
            raise ValueError(f"{where} has an invalid OBJ index {token!r}") from exc
        id_ = number - 1 if number > 0 else counts[i] + number
        if number == 0 or not 0 <= id_ < counts[i]:
            raise ValueError(f"{where} OBJ index {word!r} does not address an earlier definition")
        indices.append(id_)
    return indices[0]


def _read_obj(path, where):
    vertices, faces = [], []
    counts = [0, 0, 0]
    with path.open(encoding="utf-8-sig") as source:
        for line, raw in enumerate(source, 1):
            words = raw.partition("#")[0].split()
            if not words:
                continue
            verb, *args = words
            at = f"{where} ({path.name}:{line})"
            if verb in ("v", "vt", "vn"):
                pos = {"v": 0, "vt": 1, "vn": 2}[verb]
                if (verb in ("v", "vn") and len(args) != 3) or (
                    verb == "vt" and not 1 <= len(args) <= 3
                ):
                    raise ValueError(f"{at} has unsupported OBJ {verb} coordinate count")
                try:
                    data = tuple(float(word) for word in args)
                except ValueError as exc:
                    raise ValueError(f"{at} requires finite OBJ coordinates") from exc
                if not np.isfinite(data).all():
                    raise ValueError(f"{at} requires finite OBJ coordinates")
                counts[pos] += 1
                if verb == "v":
                    if counts[0] > MAX_VERTICES:
                        raise ValueError(f"{at} exceeds the supported OBJ vertex count")
                    vertices.append(data)
            elif verb == "f":
                if len(args) != 3:
                    raise ValueError(f"{at} requires an authored triangle face")
                if len(faces) >= MAX_MESH_TRIANGLES:
                    raise ValueError(f"{at} exceeds the supported closed mesh triangle count")
                faces.append(tuple(_corner(token, counts, at) for token in args))
            elif verb not in {"o", "g", "s", "mtllib", "usemtl"}:
                raise ValueError(f"{at} contains unsupported OBJ statement {verb!r}")
    if not vertices or not faces:
        raise ValueError(f"{where} requires OBJ vertices and triangular faces")
    return np.asarray(vertices), faces


def load_obj_mesh(
    xml_path,
    file,
    scale=(1, 1, 1),
    *,
    meshdir=None,
    where,
    refpos=(0, 0, 0),
    refquat_wxyz=(1, 0, 0, 0),
):
    """Read connected-solid source indices; native registration certifies the transformed boundary.

    Local references remain inside the descriptor folder or its named meshdir. OBJ grouping,
    material and shading records have no effect on solid geometry. Geometry with separate
    face-corner texture/normal indices preserves its original position IDs.
    """
    root = Path(xml_path).resolve().parent
    directory = root if meshdir is None else _local(root, meshdir, where + " meshdir")
    path = _local(directory, file, where)
    if path.suffix.lower() != ".obj":
        raise ValueError(f"{where} supports only local triangular OBJ files")
    if not path.is_file():
        raise ValueError(f"{where} missing local OBJ file {file!r}")
    scaling = _vector(scale, 3, where + " scale")
    offset = _vector(refpos, 3, where + " refpos")
    rotation = _vector(refquat_wxyz, 4, where + " refquat")
    if np.any(scaling == 0):
        raise ValueError(f"{where} needs nonzero mesh scale on every axis")
    magnitude = np.max(np.abs(rotation))
    if magnitude == 0:
        raise ValueError(f"{where} requires nonzero refquat")
    quat = rotation[[1, 2, 3, 0]] / magnitude
    quat /= np.linalg.norm(quat)
    matrix = quaternion_matrix(quat[None, :])[0]
    try:
        source, faces = _read_obj(path, where)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{where} cannot read local UTF-8 OBJ {file!r}") from exc
    with np.errstate(over="ignore", invalid="ignore"):
        vertices = ((source - offset) @ matrix) * scaling
    if not np.isfinite(vertices).all():
        raise ValueError(f"{where} has unrepresentable scaled mesh vertices")
    if np.count_nonzero(scaling < 0) % 2:
        faces = [(a, c, b) for a, b, c in faces]
    return LocalTriangleMesh(tuple(map(tuple, vertices)), tuple(faces))


def mjcf_mesh_assets(robot, xml_path):
    """Index MJCF asset declarations and load each named collision mesh on first use."""
    compiler = _child(robot, "compiler", "mujoco", required=False)
    meshdir = None if compiler is None else compiler.get("meshdir")
    declarations = {}
    for block in robot.findall("asset"):
        _attributes(block, set(), "MJCF asset")
        _children(block, {"mesh"}, "MJCF asset")
        for mesh in block:
            if "name" in mesh.attrib:
                name = _name(mesh, "name", "MJCF mesh")
            else:
                file = _name(mesh, "file", "MJCF mesh")
                name = PurePosixPath(file.replace("\\", "/")).stem
                if not name:
                    raise ValueError("MJCF mesh needs a name or a named file")
            if name in declarations:
                raise ValueError(f"duplicate MJCF mesh asset name {name!r}")
            declarations[name] = mesh
    loaded = {}

    def resolve(name, where):
        if name not in declarations:
            raise ValueError(f"{where} refers to unknown MJCF mesh {name!r}")
        if name not in loaded:
            mesh = declarations[name]
            source = f"{where}, MJCF mesh {name!r}"
            _attributes(mesh, {"name", "file", "scale", "refpos", "refquat"}, source)
            _children(mesh, set(), source)
            loaded[name] = load_obj_mesh(
                xml_path,
                _name(mesh, "file", source),
                _numbers(mesh, "scale", 3, source, (1, 1, 1)),
                meshdir=meshdir,
                where=source,
                refpos=_numbers(mesh, "refpos", 3, source, (0, 0, 0)),
                refquat_wxyz=_numbers(mesh, "refquat", 4, source, (1, 0, 0, 0)),
            )
        return loaded[name]

    return resolve
