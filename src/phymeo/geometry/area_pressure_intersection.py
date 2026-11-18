"""Bounded HOST tetrahedral pressure/finite rigid triangle intersection.

Meshes describe one stored world-space configuration. A is compliant and B is
rigid; B's winding points toward A. This is static geometry generation, not a
collision time query, deformation model, state certificate or dynamics step.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import combinations
from numbers import Integral
from types import MappingProxyType

import numpy as np

from . import _area_pressure_clipping as exact
from ._area_pressure_clipping import AreaPressureIntersectionError
from .area_pressure import (
    MAX_TRIANGLES,
    AreaPressureSurface,
    AreaPressureTriangle,
    _array,
    _identifier,
    _identity,
    _integer,
    _stamp,
)

MAX_NODES = 128
MAX_CELLS = 32
MAX_RIGID_FACES = 64
MAX_CANDIDATE_PAIRS = 2048


def _fail(category, message):
    raise AreaPressureIntersectionError(category, message)


def _ids(value, cap, name):
    if not isinstance(value, (list, tuple)):
        _fail("invalid_input", f"{name} must be a tuple or list")
    if len(value) > cap:
        _fail("resource_limit", f"{name} exceeds {cap} entries")
    if not value:
        _fail("invalid_input", f"{name} must be nonempty")
    result = tuple(_identifier(item, name) for item in value)
    if len(set(result)) != len(result):
        _fail("invalid_input", f"{name} must be unique")
    return result


def _domain(value):
    if not isinstance(value, str) or value not in ("full", "local"):
        _fail("invalid_input", "pressure_domain must be full or local")
    return str(value)


def _connectivity(value, width, cap, count, name):
    if not isinstance(value, (list, tuple, np.ndarray)):
        _fail("invalid_input", f"{name} must be a bounded integer array")
    if isinstance(value, np.ndarray) and value.ndim != 2:
        _fail("invalid_input", f"{name} must have rank two")
    if len(value) > cap:
        _fail("resource_limit", f"{name} exceeds {cap} entries")
    if not len(value):
        _fail("invalid_input", f"{name} must be nonempty")
    rows = []
    for row in value:
        if not isinstance(row, (tuple, list, np.ndarray)):
            _fail("invalid_input", f"each {name} row needs {width} indices")
        if isinstance(row, np.ndarray) and (row.shape != (width,) or row.dtype.kind not in "iu"):
            _fail("invalid_input", f"{name} indices must be integers")
        if len(row) != width:
            _fail("invalid_input", f"each {name} row needs {width} indices")
        checked = []
        for index in row:
            if (
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, Integral)
                or not 0 <= index < count
            ):
                _fail("invalid_input", f"{name} index is not a valid node row")
            checked.append(int(index))
        if len(set(checked)) != width:
            _fail("invalid_mesh", f"{name} row repeats a node")
        rows.append(tuple(checked))
    unordered = {tuple(sorted(row)) for row in rows}
    if len(unordered) != len(rows):
        _fail("invalid_mesh", f"duplicate {name}")
    if {index for row in rows for index in row} != set(range(count)):
        _fail("invalid_mesh", "every node must be used")
    array = np.asarray(rows, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(len(rows), width)


def _vertices(value, count):
    vertices = _array(value, (count, 3), "world_vertices")
    if len({tuple(row) for row in vertices}) != count:
        _fail("invalid_mesh", "different node IDs share a world coordinate")
    coordinates = tuple(
        tuple(exact.exact_scalar(component) for component in row) for row in vertices
    )
    return vertices, coordinates


def _tet_geometry(nodes, coordinates, cells, pressures):
    cell_nodes = tuple(tuple(nodes[int(i)] for i in row) for row in cells)
    pieces = tuple(
        exact.tetrahedron(
            tuple(coordinates[int(i)] for i in row), tuple(pressures[int(i)] for i in row)
        )
        for row in cells
    )
    incidence = {}
    for cell, ids in enumerate(cell_nodes):
        for opposite in range(4):
            face = tuple(sorted(ids[:opposite] + ids[opposite + 1 :]))
            incidence.setdefault(face, []).append((cell, opposite))
    positions = dict(zip(nodes, coordinates, strict=True))
    for face, owners in incidence.items():
        if len(owners) > 2:
            _fail("invalid_mesh", "a tetrahedral face has more than two owners")
        if len(owners) == 2:
            a, b, c = (positions[node] for node in face)
            normal = exact.cross(exact.difference(b, a), exact.difference(c, a))
            heights = tuple(
                exact.dot(normal, exact.difference(pieces[cell].points[opposite], a))
                for cell, opposite in owners
            )
            if heights[0] == 0 or heights[1] == 0 or (heights[0] > 0) == (heights[1] > 0):
                _fail("invalid_mesh", "shared tetrahedral face owners must lie on opposite sides")
    for i, j in combinations(range(len(pieces)), 2):
        vertices = exact.tetrahedron_intersection_vertices(pieces[i], pieces[j])
        exact.check_shared_simplex(pieces[i], cell_nodes[i], cell_nodes[j], vertices)
    return (
        pieces,
        cell_nodes,
        MappingProxyType({face: tuple(owners) for face, owners in incidence.items()}),
    )


@dataclass(frozen=True, eq=False)
class TetraPressureMesh:
    node_ids: tuple[str, ...]
    world_vertices: np.ndarray
    cells: np.ndarray
    nodal_pressure: np.ndarray
    cell_ids: tuple[str, ...]
    pressure_domain: str = field(default="full", kw_only=True)

    def __post_init__(self):
        nodes = _ids(self.node_ids, MAX_NODES, "node_ids")
        cell_ids = _ids(self.cell_ids, MAX_CELLS, "cell_ids")
        domain = _domain(self.pressure_domain)
        vertices, coordinates = _vertices(self.world_vertices, len(nodes))
        cells = _connectivity(self.cells, 4, MAX_CELLS, len(nodes), "cells")
        if len(cell_ids) != len(cells):
            _fail("invalid_input", "cell_ids must match cells")
        pressure = _array(self.nodal_pressure, (len(nodes),), "nodal_pressure")
        if np.any(pressure < 0):
            _fail("invalid_input", "nodal_pressure must be nonnegative")
        exact_pressure = tuple(exact.exact_scalar(value) for value in pressure)
        pieces, cell_nodes, incidence = _tet_geometry(nodes, coordinates, cells, exact_pressure)
        if domain == "full":
            boundary = {
                node for face, owners in incidence.items() if len(owners) == 1 for node in face
            }
            by_node = dict(zip(nodes, exact_pressure, strict=True))
            if any(by_node[node] != 0 for node in boundary):
                _fail("invalid_mesh", "full pressure bodies require zero boundary pressure")
            if not any(by_node[node] > 0 for node in nodes if node not in boundary):
                _fail(
                    "invalid_mesh",
                    "full pressure bodies require positive pressure at an interior node",
                )
        for name, value in (
            ("node_ids", nodes),
            ("world_vertices", vertices),
            ("cells", cells),
            ("nodal_pressure", pressure),
            ("cell_ids", cell_ids),
            ("pressure_domain", domain),
            ("_exact_cells", pieces),
            ("_cell_nodes", cell_nodes),
            ("_face_owners", incidence),
        ):
            object.__setattr__(self, name, value)


def _rigid_geometry(nodes, coordinates, faces):
    face_nodes = tuple(tuple(nodes[int(i)] for i in row) for row in faces)
    pieces = tuple(exact.triangle(tuple(coordinates[int(i)] for i in row)) for row in faces)
    edges = {}
    for ids in face_nodes:
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            edges.setdefault(tuple(sorted((a, b))), []).append((a, b))
    for owners in edges.values():
        if len(owners) > 2 or (len(owners) == 2 and owners[0] != owners[1][::-1]):
            _fail("invalid_mesh", "shared rigid edges need at most two opposite incidences")
    for i, j in combinations(range(len(pieces)), 2):
        vertices = exact.triangle_intersection_vertices(pieces[i], pieces[j])
        exact.check_shared_simplex(pieces[i], face_nodes[i], face_nodes[j], vertices)
    return pieces


@dataclass(frozen=True, eq=False)
class RigidPressureMesh:
    node_ids: tuple[str, ...]
    world_vertices: np.ndarray
    faces: np.ndarray
    face_ids: tuple[str, ...]

    def __post_init__(self):
        nodes = _ids(self.node_ids, MAX_NODES, "node_ids")
        face_ids = _ids(self.face_ids, MAX_RIGID_FACES, "face_ids")
        vertices, coordinates = _vertices(self.world_vertices, len(nodes))
        faces = _connectivity(self.faces, 3, MAX_RIGID_FACES, len(nodes), "faces")
        if len(face_ids) != len(faces):
            _fail("invalid_input", "face_ids must match faces")
        pieces = _rigid_geometry(nodes, coordinates, faces)
        for name, value in (
            ("node_ids", nodes),
            ("world_vertices", vertices),
            ("faces", faces),
            ("face_ids", face_ids),
            ("_exact_faces", pieces),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class AreaPressureIntersection:
    surface: AreaPressureSurface
    pressure_domain: str
    candidate_pairs: int
    polygon_count: int
    shared_face_duplicates: int

    def __post_init__(self):
        if not isinstance(self.surface, AreaPressureSurface):
            _fail("invalid_input", "intersection requires AreaPressureSurface")
        surface = AreaPressureSurface(
            self.surface.identity, self.surface.stamp, self.surface.triangles
        )
        domain = _domain(self.pressure_domain)
        counts = tuple(
            _integer(getattr(self, name), name)
            for name in ("candidate_pairs", "polygon_count", "shared_face_duplicates")
        )
        candidates, polygons, duplicates = counts
        if candidates > MAX_CANDIDATE_PAIRS:
            _fail("resource_limit", "candidate pair capacity exceeded")
        if (
            polygons + duplicates > candidates
            or not polygons <= len(surface.triangles) <= 5 * polygons
        ):
            _fail("invalid_input", "intersection counts do not describe the surface")
        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "pressure_domain", domain)
        for name, value in zip(
            ("candidate_pairs", "polygon_count", "shared_face_duplicates"), counts, strict=True
        ):
            object.__setattr__(self, name, value)


def _raw_mesh_fields(value, *, pressure):
    """Snapshot only a fixed number of named fields, including at the build seam."""
    fields = (
        ("node_ids", "world_vertices", "cells", "nodal_pressure", "cell_ids", "pressure_domain")
        if pressure
        else ("node_ids", "world_vertices", "faces", "face_ids")
    )
    if not isinstance(value, Mapping) or len(value) != len(fields):
        _fail("invalid_input", "incomplete raw mesh declaration")
    try:
        return {name: value[name] for name in fields}
    except KeyError as error:
        raise AreaPressureIntersectionError(
            "invalid_input", "incomplete raw mesh declaration"
        ) from error


def _owned_mesh_input(value, *, pressure):
    """Bound and copy a raw declaration, without running geometric predicates."""
    value = _raw_mesh_fields(value, pressure=pressure)
    nodes = _ids(value["node_ids"], MAX_NODES, "node_ids")
    name, width, cap, ids_name = (
        ("cells", 4, MAX_CELLS, "cell_ids")
        if pressure
        else ("faces", 3, MAX_RIGID_FACES, "face_ids")
    )
    result = {
        "node_ids": nodes,
        "world_vertices": _array(value["world_vertices"], (len(nodes), 3), "world_vertices"),
        name: _connectivity(value[name], width, cap, len(nodes), name),
        ids_name: _ids(value[ids_name], cap, ids_name),
    }
    if pressure:
        result["nodal_pressure"] = _array(value["nodal_pressure"], (len(nodes),), "nodal_pressure")
        result["pressure_domain"] = _domain(value["pressure_domain"])
    return MappingProxyType(result)


@dataclass(frozen=True, init=False)
class _AreaPressureGeometryInput:
    """Owned current input, not a certificate or a reusable geometry cache."""

    pressure: object
    rigid: object
    identity: object
    stamp: object

    def __init__(self, pressure, rigid, identity, stamp):
        object.__setattr__(self, "pressure", _owned_mesh_input(pressure, pressure=True))
        object.__setattr__(self, "rigid", _owned_mesh_input(rigid, pressure=False))
        object.__setattr__(self, "identity", _identity(identity))
        object.__setattr__(self, "stamp", _stamp(stamp))


class _StepMeshBuilder:
    """Share checked mesh sides only within one ordered raw request batch."""

    def __init__(self, count):
        self.pressures = {} if count > 1 else None
        self.rigids = {} if count > 1 else None

    @staticmethod
    def _mesh(raw, body, material, cache, *, pressure):
        constructor = TetraPressureMesh if pressure else RigidPressureMesh
        if cache is None:
            return constructor(**_raw_mesh_fields(raw, pressure=pressure))

        # Bound and own inputs before taking bytes, even for a cache hit.
        fields = _owned_mesh_input(raw, pressure=pressure)
        key = (
            body,
            material,
            tuple(
                (value.dtype.str, value.shape, value.tobytes())
                if isinstance(value, np.ndarray)
                else value
                for value in fields.values()
            ),
        )
        mesh = cache.get(key)
        if mesh is None:
            mesh = constructor(**fields)
            cache[key] = mesh
        return mesh

    def build(self, request):
        if type(request) is not _AreaPressureGeometryInput:
            _fail("invalid_input", "expected owned raw geometry input")
        identity, stamp = _identity(request.identity), _stamp(request.stamp)
        pressure = self._mesh(
            request.pressure,
            identity.compliant_body,
            identity.compliant_material,
            self.pressures,
            pressure=True,
        )
        rigid = self._mesh(
            request.rigid, identity.rigid_body, identity.rigid_material, self.rigids, pressure=False
        )
        return pressure, rigid, identity, stamp


def _build_geometry_inputs(requests):
    # Lazy consumption keeps the mixed intersector's per-query capacity check
    # ahead of the next query's predicates. The iterator owns both maps.
    return map(_StepMeshBuilder(len(requests)).build, requests)


def _meshes(pressure_mesh, rigid_mesh):
    if not isinstance(pressure_mesh, TetraPressureMesh) or not isinstance(
        rigid_mesh, RigidPressureMesh
    ):
        _fail("invalid_input", "expected TetraPressureMesh and RigidPressureMesh")
    # Rebuild from public owned fields; frozen types or private caches are not
    # accepted as a certificate when an object is supplied by a caller.
    try:
        pressure = TetraPressureMesh(
            pressure_mesh.node_ids,
            pressure_mesh.world_vertices,
            pressure_mesh.cells,
            pressure_mesh.nodal_pressure,
            pressure_mesh.cell_ids,
            pressure_domain=pressure_mesh.pressure_domain,
        )
        rigid = RigidPressureMesh(
            rigid_mesh.node_ids,
            rigid_mesh.world_vertices,
            rigid_mesh.faces,
            rigid_mesh.face_ids,
        )
    except AttributeError as error:
        raise AreaPressureIntersectionError("invalid_input", "incomplete mesh object") from error
    return pressure, rigid


def _owns_polygon(mesh, cell_index, face, polygon):
    cell = mesh._exact_cells[cell_index]
    if exact.dot(cell.gradient, face.normal) <= 0:
        _fail(
            "unsupported_gradient",
            f"cell {mesh.cell_ids[cell_index]} has nonpositive effective gradient",
        )
    weights = tuple(cell.weights(point) for point in polygon)
    support = tuple(i for i in range(4) if all(weight[i] == 0 for weight in weights))
    if not support:
        return True
    if len(support) != 1:
        _fail("invalid_mesh", "positive-area polygon lies on multiple tetrahedral faces")
    ids = mesh._cell_nodes[cell_index]
    opposite = support[0]
    key = tuple(sorted(ids[:opposite] + ids[opposite + 1 :]))
    owners = mesh._face_owners[key]
    if len(owners) == 1:
        return True
    for other, _ in owners:
        if exact.dot(mesh._exact_cells[other].gradient, face.normal) <= 0:
            _fail("unsupported_gradient", "a shared-face owner has nonpositive effective gradient")
    if any(mesh._exact_cells[other].gradient != cell.gradient for other, _ in owners):
        _fail(
            "unsupported_shared_face_gradient",
            "shared-face owners have different exact world gradients",
        )
    return mesh.cell_ids[cell_index] == min(mesh.cell_ids[other] for other, _ in owners)


def _publish_triangle(cell, polygon, cell_id, face_id, fan_index):
    points = (polygon[0], polygon[fan_index + 1], polygon[fan_index + 2])
    pressure = tuple(cell.pressure(point) for point in points)
    encoded = json.dumps((cell_id, face_id, fan_index), ensure_ascii=True, separators=(",", ":"))
    triangle_id = "i1-" + sha256(encoded.encode("utf-8")).hexdigest()
    try:
        vertices = np.array(
            tuple(tuple(float(value) for value in point) for point in points), dtype=np.float64
        )
        values = np.array(tuple(float(value) for value in pressure), dtype=np.float64)
        gradient = np.array(tuple(float(value) for value in cell.gradient), dtype=np.float64)
        return AreaPressureTriangle(triangle_id, cell_id, face_id, vertices, values, gradient)
    except (ValueError, OverflowError, FloatingPointError) as error:
        raise AreaPressureIntersectionError(
            "publication_failure", "exact triangle cannot be published through I0"
        ) from error


def intersect_area_pressure(pressure_mesh, rigid_mesh, *, identity, stamp):
    """Clip every finite face/cell pair atomically and preserve the asset domain."""
    identity, stamp = _identity(identity), _stamp(stamp)
    pressure, rigid = _meshes(pressure_mesh, rigid_mesh)
    return _intersect_checked(pressure, rigid, identity=identity, stamp=stamp)


def _intersect_checked(pressure, rigid, *, identity, stamp):
    """Internal core: callers must have just constructed their owned meshes."""
    candidates = len(pressure.cells) * len(rigid.faces)
    if candidates > MAX_CANDIDATE_PAIRS:
        _fail("resource_limit", "candidate pair capacity exceeded")
    triangles = []
    polygon_count = duplicates = 0
    for ci in sorted(range(len(pressure.cells)), key=lambda i: pressure.cell_ids[i]):
        cell = pressure._exact_cells[ci]
        for fi in sorted(range(len(rigid.faces)), key=lambda i: rigid.face_ids[i]):
            face = rigid._exact_faces[fi]
            polygon = exact.clip_triangle(cell, face)
            if not polygon:
                continue
            if not _owns_polygon(pressure, ci, face, polygon):
                duplicates += 1
                continue
            count = len(polygon) - 2
            if len(triangles) + count > MAX_TRIANGLES:
                _fail("resource_limit", "intersection exceeds I0 triangle capacity")
            polygon_count += 1
            for fan in range(count):
                triangles.append(
                    _publish_triangle(cell, polygon, pressure.cell_ids[ci], rigid.face_ids[fi], fan)
                )
    try:
        surface = AreaPressureSurface(identity, stamp, tuple(triangles))
        return AreaPressureIntersection(
            surface, pressure.pressure_domain, candidates, polygon_count, duplicates
        )
    except AreaPressureIntersectionError:
        raise
    except (ValueError, OverflowError, FloatingPointError) as error:
        raise AreaPressureIntersectionError(
            "publication_failure", "intersection cannot be published through I0"
        ) from error
