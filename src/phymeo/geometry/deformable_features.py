"""Discrete, topology-aware vertex/triangle and unordered edge/edge material contacts."""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
from math import inf, nextafter
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey, CollisionModel, CollisionMotion
from ..core.validation import finite_real
from ._capsule_triangle_sweep import segment_minimum
from ._prism_pair_exact import sqrt_bounds
from ._round_sweep import _axis_features, _unit, _xyz
from ._triangle_pair_sweep import _ZERO_VECTOR, _FacePath, _Pose
from .collision_pipeline import CollisionWorkspace, _material_face_broad_pairs
from .collision_witness import ConvexContactPoint

_Node = tuple[int, int, int]
_Nodes = tuple[_Node, ...]
_Edge = tuple[_Node, _Node]
_Point = tuple[float, float, float]


@dataclass(frozen=True)
class DeformableFeatureContact:
    """Materials use (world, body, node). Two closest endpoints span all minimizing pairs."""

    kind: str
    first_nodes: _Nodes
    second_nodes: _Nodes
    support_face_pairs: tuple[tuple[CollisionKey, CollisionKey], ...]
    distance_m: float
    gap_to_force_m: float
    distance_bounds_m: tuple[float, float]
    gap_to_force_bounds_m: tuple[float, float]
    normal: _Point | None
    inside_force_distance: bool
    force_eligible: bool
    closest: tuple[ConvexContactPoint, ...]


@dataclass(frozen=True, eq=False)
class _FeatureInput:
    model: CollisionModel
    positions: tuple[bytes, ...]
    search_distance_m: float
    contact_distance_m: float
    rings: int
    contacts: tuple[DeformableFeatureContact, ...]


@dataclass(frozen=True)
class DeformableFeatureContacts:
    """All qualifying features or an explicit failure with no partial feature list."""

    status: str
    reason: str
    backend: str
    contacts: tuple[DeformableFeatureContact, ...] = ()
    broad_face_pairs: int = 0
    candidate_features: int = 0
    feature_tests: int = 0
    required_features: int | None = None
    invalid_faces: tuple[CollisionKey, ...] = ()
    narrow_backend: str = "host"


@dataclass
class _Face:
    key: CollisionKey
    nodes: _Nodes
    edges: tuple[_Edge, ...]
    pose: _Pose
    path: _FacePath


class _Geometry(Exception):
    def __init__(self, status, reason, *keys):
        self.status, self.reason, self.keys = status, reason, tuple(sorted(set(keys)))


def _topology(model, motion, *, affine=False):
    faces = []
    poses = {}
    points = {}
    introduced = {}
    graph = {}
    edge_faces = {}
    end_points = {}
    for s in model.shapes:
        m = motion[s.key]
        if not isinstance(m, CollisionMotion):
            raise TypeError("deformable feature motion must use CollisionMotion")
        if s.kind != "triangle" or s.radius:
            raise _Geometry("unsupported", "unrounded_triangle_model_required", s.key)
        if m.turn is not None or m.body_orientation_xyzw is not None or (m.moving and not affine):
            reason = "affine_material_path_required" if affine else "discrete_pose_required"
            raise _Geometry("unsupported", reason, s.key)
        if len(m.start) != 3:
            raise ValueError("triangle material positions require three vertices")
        path = _FacePath(s, m)
        pose = _Pose(path, F(0))
        if affine and not path.valid:
            raise _Geometry("unresolved", "face_degenerate_on_path", s.key)
        if not pose.den:
            raise _Geometry("invalid_geometry", "degenerate_face", s.key)
        owner = s.key.world, s.key.body
        nodes = []
        for id_, p, step in zip(s.vertex_ids, pose.p, path.step, strict=True):
            node = (*owner, id_)
            if node in points and not np.array_equal(points[node], p):
                raise _Geometry(
                    "invalid_geometry", "inconsistent_material_position", introduced[node], s.key
                )
            if affine and node in end_points and not np.array_equal(end_points[node], p + step):
                raise _Geometry(
                    "invalid_geometry", "inconsistent_material_path", introduced[node], s.key
                )
            if affine:
                end_points[node] = p + step
            points[node] = p
            introduced.setdefault(node, s.key)
            graph.setdefault(node, {node})
            nodes.append(node)
        nodes = tuple(sorted(nodes))
        if nodes in poses:
            raise _Geometry("invalid_geometry", "duplicate_material_face", poses[nodes].key, s.key)
        edges = tuple(combinations(nodes, 2))
        for edge in edges:
            incident = edge_faces.setdefault(edge, [])
            incident.append(s.key)
            if len(incident) > 2:
                raise _Geometry("invalid_geometry", "nonmanifold_material_edge", *incident)
        for a, b in edges:
            graph[a].add(b)
            graph[b].add(a)
        face = _Face(s.key, nodes, edges, pose, path)
        faces.append(face)
        poses[nodes] = face
    return faces, poses, points, graph


def _neighborhoods(graph, rings):
    # Ring one filters intrinsically shared nodes, ring two adds one graph
    # edge, then each additional level adds one graph step on the same body.
    result = {}
    for node in graph:
        seen = {node}
        frontier = {node}
        for _ in range(rings - 1):
            frontier = set().union(*(graph[x] for x in frontier)) - seen
            if not frontier:
                break
            seen.update(frontier)
        result[node] = seen
    return result


def _members(a, b, near):
    for src, dst in ((a, b), (b, a)):
        for n in src.nodes:
            if near[n].isdisjoint(dst.nodes):
                yield ("VT", (n,), dst.nodes), (src.key, dst.key)
    for e in a.edges:
        for f in b.edges:
            if near[e[0]].isdisjoint(f) and near[e[1]].isdisjoint(f):
                if e < f:
                    yield ("EE", e, f), (a.key, b.key)
                else:
                    yield ("EE", f, e), (b.key, a.key)


def _bounds(group, points):
    xyz = [points[n] for n in group]
    return tuple((min(v), max(v)) for v in zip(*xyz, strict=True))


def _lower2(a, b):
    return sum(max(0, x - t, z - y) ** 2 for (x, y), (z, t) in zip(a, b, strict=True))


def _weight(nodes, params):
    return tuple((n[2], float(w)) for n, w in zip(nodes, params, strict=True))


def _edge_minimum(a, b, c, d):
    # Existing exact segment features generate the limits of a parallel
    # minimizer interval as well as the unique skew/endpoint solution.
    features = tuple(_axis_features(b - a, d - c, a - c, _ZERO_VECTOR, F(0)))
    best = min(q.c for q in features)
    tie = sorted({(q.s[0], q.t[0]) for q in features if q.c == best})
    return best, (tie[0],) if tie[0] == tie[-1] else (tie[0], tie[-1])


def _round_out(x, up):
    value = float(x)
    if (up and F(value) < x) or (not up and F(value) > x):
        value = nextafter(value, inf if up else -inf)
    return value


def _measure(key, support, points, poses, search_sq, force, *, exact=False):
    kind, first, second = key
    p = points[first[0]]
    if kind == "VT":
        tri = poses[second].pose
        d2, (_, weights), _ = segment_minimum(p, _ZERO_VECTOR, tri)
        if search_sq is not None and d2 >= search_sq:
            return None
        other = np.asarray(weights, dtype=object) @ tri.p
        ends = (
            ConvexContactPoint(
                _xyz(p),
                _xyz(other),
                ((first[0][2], 1.0),),
                tuple((n, float(w)) for n, w in tri.address(weights)),
            ),
        )
        delta = other - p
    else:
        p1 = points[first[1]]
        q, q1 = (points[n] for n in second)
        d2, span = _edge_minimum(p, p1, q, q1)
        if search_sq is not None and d2 >= search_sq:
            return None
        out = []
        for a, b in span:
            pa, pb = p + a * (p1 - p), q + b * (q1 - q)
            delta = pb - pa
            out.append(
                ConvexContactPoint(
                    _xyz(pa), _xyz(pb), _weight(first, (1 - a, a)), _weight(second, (1 - b, b))
                )
            )
        ends = tuple(out)
    normal, _ = _unit(delta)
    low, high = sqrt_bounds(d2, 96)
    distance = float((low + high) / 2)
    inside = d2 < force * force
    contact = DeformableFeatureContact(
        kind,
        first,
        second,
        tuple(sorted(support)),
        distance,
        float((low + high) / 2 - force),
        (_round_out(low, False), _round_out(high, True)),
        (_round_out(low - force, False), _round_out(high - force, True)),
        None if normal is None else _xyz(normal),
        inside,
        inside and bool(d2),
        ends,
    )
    return (contact, d2) if exact else contact


def _capacity(value, low, high, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def query_deformable_features(
    model,
    motion,
    workspace,
    *,
    search_distance_m,
    contact_distance_m,
    topology_rings=2,
    max_feature_tests=65536,
    feature_capacity=32768,
):
    """List every in-range discrete VT and unordered EE of registered finite zero-thickness faces.

    Search and interaction use strict inequalities. A coincident feature has
    no oriented separation direction. A contact row is geometry only. Search
    uses the workspace's face broad phase. Finite closest features and their
    distance bounds use the original material coordinates on HOST.
    """
    if (
        not isinstance(model, CollisionModel)
        or not isinstance(workspace, CollisionWorkspace)
        or workspace._model is not model
    ):
        raise ValueError("workspace must belong to the registered collision model")
    if not isinstance(motion, Mapping) or set(motion) != {s.key for s in model.shapes}:
        raise ValueError("motion keys must exactly cover the collision model")
    search = F(finite_real(search_distance_m, "search_distance_m"))
    force = F(finite_real(contact_distance_m, "contact_distance_m"))
    if search <= 0 or not 0 <= force <= search:
        raise ValueError(
            "require 0 <= contact_distance_m <= search_distance_m and positive search distance"
        )
    rings = _capacity(topology_rings, 1, 8, "topology_rings")
    max_tests = _capacity(max_feature_tests, 0, 1_000_000, "max_feature_tests")
    max_results = _capacity(feature_capacity, 0, 262144, "feature_capacity")
    backend = workspace._backend
    with workspace._lock:
        if workspace._closed:
            raise RuntimeError("collision workspace is closed")
        paths = dict(motion)
        try:
            faces, poses, points, graph = _topology(model, paths)
        except _Geometry as exc:
            return DeformableFeatureContacts(
                exc.status, exc.reason, backend, invalid_faces=exc.keys
            )
        broad = (
            _material_face_broad_pairs(model, paths, workspace, float(search))
            if len(faces) > 1
            else ()
        )
    near = _neighborhoods(graph, rings)
    candidates = {}
    for i, j in broad:
        for key, support in _members(faces[i], faces[j], near):
            if key not in candidates:
                if len(candidates) == max_tests:
                    return DeformableFeatureContacts(
                        "capacity_exceeded",
                        "feature_test_capacity",
                        backend,
                        broad_face_pairs=len(broad),
                        candidate_features=len(candidates) + 1,
                    )
                candidates[key] = set()
            candidates[key].add(support)
    search_sq = search * search
    boxes = {g: _bounds(g, points) for _, a, b in candidates for g in (a, b)}
    kept = []
    found = tests = 0
    for key, support in sorted(candidates.items()):
        if _lower2(boxes[key[1]], boxes[key[2]]) >= search_sq:
            continue
        tests += 1
        contact = _measure(key, support, points, poses, search_sq, force)
        if contact is not None:
            found += 1
            if found <= max_results:
                kept.append(contact)
    info = dict(
        broad_face_pairs=len(broad),
        candidate_features=len(candidates),
        feature_tests=tests,
        required_features=found,
    )
    if found > max_results:
        return DeformableFeatureContacts("capacity_exceeded", "feature_capacity", backend, **info)
    frame = DeformableFeatureContacts("ok", "complete", backend, tuple(kept), **info)
    # A separate load consumer may reuse this geometry only for the identical
    # registration and complete pose. Keep provenance out of public row
    # serialization and dataclass replacements, which do not re-run geometry.
    source = _FeatureInput(
        model,
        tuple(paths[s.key].start.tobytes() for s in model.shapes),
        float(search),
        float(force),
        rings,
        frame.contacts,
    )
    object.__setattr__(frame, "_source", source)
    return frame


def _matches_feature_input(frame, model, paths, search, force, rings):
    source = getattr(frame, "_source", None)
    return (
        isinstance(source, _FeatureInput)
        and source.model is model
        and source.contacts is frame.contacts
        and source.search_distance_m == search
        and source.contact_distance_m == force
        and source.rings == rings
        and source.positions == tuple(paths[s.key].start.tobytes() for s in model.shapes)
    )
