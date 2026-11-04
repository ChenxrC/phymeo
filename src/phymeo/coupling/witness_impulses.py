"""Bind actual published surface witnesses to the existing multi-domain Coulomb solver."""

from collections.abc import Mapping
from dataclasses import replace
from math import fsum, hypot

import numpy as np

from ..constraints.collision_shapes import CollisionModel, CollisionMotion, CollisionPolicy
from ..constraints.contact_impulse import solve_contact_impulses
from ..core.capabilities import SolverFailure
from ..core.validation import finite_real
from ..dynamics.rotation import matrix_quaternion
from ..geometry.collision_pipeline import CollisionPair, CollisionWorkspace, detect
from .curved_manifold import hull_polygon, select_static_support
from .domain_graph import MultiDomainPointOperator


def _chart_point(frame, local, origin):
    origins, rotation, _ = frame
    arm = rotation @ local
    return np.array(
        [
            fsum([*(float(term[k]) for term in origins), float(arm[k]), -float(origin[k])])
            for k in range(3)
        ]
    )


def _material_anchor(shape, path, weights):
    """Locate the reported material address on the current registered geometry."""
    if not 1 <= len(weights) <= len(shape.vertex_ids):
        raise ValueError("collision witness requires one bounded material address")
    nodes = [node for node, _ in weights]
    values = [float(weight) for _, weight in weights]
    if (
        len(set(nodes)) != len(nodes)
        or not np.isfinite(values).all()
        or min(values) < 0
        or abs(fsum(values) - 1.0) > 1e-12
    ):
        raise ValueError("collision witness material coordinates must be a convex combination")
    vertices = dict(zip(shape.vertex_ids, path.start, strict=True))
    try:
        return np.array(
            [
                fsum(float(weight) * float(vertices[node][k]) for node, weight in weights)
                for k in range(3)
            ]
        )
    except KeyError as exc:
        raise ValueError("collision witness refers to an unregistered material vertex") from exc


def _unpack_row(row):
    if len(row) == 2:
        pair, bindings = row
        return pair, bindings, None
    if len(row) == 3:
        return row
    raise ValueError(
        "a witness row requires a collision pair, bindings and an optional support index"
    )


class _PolygonRows:
    """One public polygon, one current-pose requery and a bijection of material corners."""

    def __init__(self, pair, patch):
        self.pair = pair
        self.patch = patch
        self.requested = set()
        self.matched = set()
        self.current = None
        self.paths = None
        self.live = None

    def register(self, pair, index):
        if index is None or pair is not self.pair or index in self.requested:
            raise SolverFailure(
                "a complete convex polygon requires each indexed material corner once"
            )
        self.requested.add(index)

    def bind(self, current, paths):
        live = hull_polygon(current)
        if (
            live is None
            or len(live.points) != len(self.patch.points)
            or set(live.first_support) != set(self.patch.first_support)
            or set(live.second_support) != set(self.patch.second_support)
        ):
            raise ValueError("collision witness polygon disagrees with the live support faces")
        self.current, self.paths, self.live = current, paths, live

    def select(self, points, tol):
        # Match the two actual sites instead of assuming the live polygon starts
        # at the same corner. Ambiguous or repeated matches cannot cover a face.
        matches = [
            i
            for i, point in enumerate(self.live.points)
            if hypot(*(points[0] - point.first_point)) <= tol
            and hypot(*(points[1] - point.second_point)) <= tol
        ]
        if len(matches) != 1 or matches[0] in self.matched:
            raise ValueError("collision witness polygon has a stale or ambiguous material corner")
        self.matched.add(matches[0])
        return select_static_support(self.current, self.current.witness, matches[0])


def _current_normal(
    pair, witness, endpoints, ports, local_vertices, local_normals, tol, index, polygon, owners
):
    """Requery the immutable two-shape model in the witness-relative coordinate system."""
    if tuple(s.key for s in pair.witness_shapes) != (pair.first, pair.second):
        raise ValueError("collision witness lacks a supported live rigid shape registration")
    origin = np.asarray(
        witness.first_point if polygon is None else polygon.patch.points[0].first_point
    )
    points = []
    paths = {}
    posed_shapes = []
    for shape, (domain, row), published in zip(
        pair.witness_shapes, endpoints, (witness.first_point, witness.second_point), strict=True
    ):
        frame_reader = getattr(ports[domain], "witness_frame", None)
        if frame_reader is None:
            raise ValueError("collision witness requires material body frames")
        frame = frame_reader(row)
        actual = _chart_point(frame, frame[2], origin)
        if hypot(*(actual - (np.asarray(published) - origin))) > tol:
            raise ValueError("material port is not bound to its collision witness")
        points.append(actual)
        if polygon is not None:
            ids = getattr(ports[domain], "indices", None)
            if ids is None:
                raise ValueError("a polygon material port requires an owning body index")
            binding = (domain, int(ids[row]))
            if owners.setdefault(shape.key, binding) != binding:
                raise ValueError("collision polygon material rows refer to different rigid bodies")
            if polygon.current is not None:
                continue
        vertices = local_vertices.get(shape.key)
        if vertices is None:
            if shape.kind not in ("sphere", "plane") and shape.curved is None:
                raise ValueError("collision witness requires registered body-local shape vertices")
            vertices = ((0.0, 0.0, 0.0),)
        vertices = np.asarray(vertices, dtype=np.float64)
        if vertices.shape != (len(shape.vertex_ids), 3) or not np.isfinite(vertices).all():
            raise ValueError("witness shape vertices must match the registered material order")
        if shape.kind == "plane":
            normal = np.asarray(local_normals.get(shape.key), dtype=np.float64)
            if normal.shape != (3,) or not np.isfinite(normal).all() or not normal.any():
                raise ValueError("a plane witness requires its oriented body-local normal")
            shape = replace(shape, normal=tuple(frame[1] @ (normal / np.max(np.abs(normal)))))
        posed_shapes.append(shape)
        orientation = matrix_quaternion(frame[1]) if shape.curved is not None else None
        paths[shape.key] = CollisionMotion(
            [_chart_point(frame, vertex, origin) for vertex in vertices],
            body_orientation_xyzw=orientation,
        )
    if hypot(*(points[0] - points[1])) > tol:
        raise ValueError("collision witness ports do not meet at one material point")
    static_hulls = pair.sweep is None and all(s.kind == "hull" for s in pair.witness_shapes)
    if polygon is None or polygon.current is None:
        model = CollisionModel(posed_shapes)
        with CollisionWorkspace(model, "host") as work:
            policy = CollisionPolicy(margin=tol, proximity_patches=static_hulls)
            (current,) = detect(model, paths, policy, work).pairs
        if current.candidate and current.witness is not None and not current.witness_shapes:
            current = replace(current, witness_shapes=tuple(posed_shapes))
        if polygon is not None:
            polygon.bind(current, paths)
    else:
        current, paths = polygon.current, polygon.paths
    fresh = current.witness
    if polygon is not None:
        fresh = polygon.select(points, tol)
    elif static_hulls and hull_polygon(current) is not None:
        raise SolverFailure("the current convex polygon requires complete material rows")
    elif current.candidate and fresh is not None and index is not None:
        fresh = select_static_support(current, fresh, index)
    if (
        not current.candidate
        or fresh is None
        or fresh.normal is None
        or fresh.first_point is None
        or fresh.second_point is None
        or abs(fresh.gap_m) > tol
        or hypot(*(np.asarray(fresh.normal) - witness.normal)) > 1e-10
        or hypot(*(np.asarray(fresh.first_point) - points[0])) > tol
        or hypot(*(np.asarray(fresh.second_point) - points[1])) > tol
    ):
        raise ValueError("collision witness disagrees with the current shape pose or normal")
    for shape, old, live in zip(
        pair.witness_shapes,
        (witness.first_weights, witness.second_weights),
        (fresh.first_weights, fresh.second_weights),
        strict=True,
    ):
        path = paths[shape.key]
        if hypot(*(_material_anchor(shape, path, old) - _material_anchor(shape, path, live))) > tol:
            raise ValueError("collision witness material address disagrees with the current shape")
    return fresh.normal


def solve_witness_impulses(
    ports,
    rows,
    mu_static,
    mu_dynamic,
    *,
    sample="pose",
    local_vertices=None,
    local_normals=None,
    velocity_tolerance=1e-10,
    response_tolerance=1e-12,
    position_tolerance=1e-9,
    max_iterations=200,
):
    """Solve coincident material rows at one physical instant, with zero restitution.

    Rows contain (CollisionPair, {shape_key: (port_domain, local_row)}). A third
    entry selects a static cylinder-plane endpoint or a corner of the unique hull-hull
    contact polygon. Supply every positive-area polygon corner once, with one material
    port per owner and corner, in any row order. A static row must be touching. First-contact rows must share one reported witness
    time. The HOST impulse adapter checks both material frames and queries the current
    rigid shapes again in a common local chart. Supply local_vertices keyed by shape,
    in registered vertex order. The center of a sphere or curved solid and the
    anchor of a plane default to the body origin. A plane requires its oriented
    body-local normal in local_normals. CurvedSolid.orientation_xyzw is body-local.
    A vertex-defined shape requires its full local geometry.
    The returned world_impulses act on the second collision shape in each row.
    """
    if sample not in ("pose", "first_contact"):
        raise ValueError("select pose or first_contact witnesses")
    tol = finite_real(position_tolerance, "position_tolerance")
    if tol < 0:
        raise ValueError("position tolerance must be non-negative")
    if local_vertices is None:
        local_vertices = {}
    if local_normals is None:
        local_normals = {}
    if not isinstance(local_vertices, Mapping) or not isinstance(local_normals, Mapping):
        raise TypeError(
            "local_vertices and local_normals must be mappings keyed by collision shape"
        )
    first, second, witnesses, pairs = [], [], [], []
    event_time = None
    polygons = {}
    for row in rows:
        pair, bindings, index = _unpack_row(row)
        if sample != "pose" and index is not None:
            raise ValueError("a material support index requires a static pose witness")
        if not isinstance(pair, CollisionPair) or not pair.candidate:
            raise ValueError("an active published collision pair is required")
        if not isinstance(bindings, Mapping) or set(bindings) != {pair.first, pair.second}:
            raise ValueError("witness bindings must cover both registered shape keys")
        if sample == "first_contact":
            sweep = pair.sweep
            if (
                pair.status != "hit"
                or sweep is None
                or sweep.first_t is None
                or sweep.witness_time is None
            ):
                raise ValueError("a first-contact witness with a physical sample time is required")
            now = sweep.witness_time
            if (
                now == 0
                and pair.mesh_round is not None
                and pair.mesh_round.start.relation != "touching"
            ):
                raise ValueError("initial solid occupancy does not supply a first-impact normal")
            if not (sweep.first_t[0] <= now <= sweep.first_t[1]) or (
                event_time is not None and event_time != now
            ):
                raise ValueError("contact graph witnesses must refer to one sampled instant")
            event_time = now
            witness = sweep.witness
        else:
            if pair.status != "touching" or pair.sweep is not None:
                raise ValueError("a static touching witness is required")
            witness = pair.witness
            if witness is not None and index is not None:
                witness = select_static_support(pair, witness, index)
            patch = hull_polygon(pair)
            if patch is not None:
                key = (pair.first, pair.second)
                if key not in polygons:
                    polygons[key] = _PolygonRows(pair, patch)
                polygons[key].register(pair, index)
        if (
            witness is None
            or witness.normal is None
            or witness.first_point is None
            or witness.second_point is None
        ):
            raise ValueError("a resolved normal and both material world points are required")
        for value in (witness.first_point, witness.second_point, witness.normal):
            observed = np.asarray(value, dtype=np.float64)
            if observed.shape != (3,) or not np.isfinite(observed).all():
                raise ValueError(
                    "collision witness world vectors must be finite and three-dimensional"
                )
        if not np.isfinite(witness.gap_m) or abs(witness.gap_m) > tol:
            raise ValueError("signed witness gap exceeds the position tolerance")
        first.append(bindings[pair.first])
        second.append(bindings[pair.second])
        witnesses.append((witness, index))
        pairs.append(pair)
    # Public collision normals point from first to second. The existing graph
    # expects a normal from its right endpoint toward its left endpoint.
    if any(len(group.requested) != len(group.patch.points) for group in polygons.values()):
        raise SolverFailure("incomplete convex polygon material rows")
    graph = MultiDomainPointOperator(ports, second, first)
    if set(local_vertices) - {key for pair in pairs for key in (pair.first, pair.second)}:
        raise ValueError("witness geometry supplied for an unbound shape")
    if set(local_normals) - {s.key for p in pairs for s in p.witness_shapes if s.kind == "plane"}:
        raise ValueError("body-local normal supplied for an unbound plane")
    owners = {}
    normals = np.asarray(
        [
            _current_normal(
                pair,
                witness,
                (a, b),
                ports,
                local_vertices,
                local_normals,
                tol,
                index,
                polygons.get((pair.first, pair.second)),
                owners,
            )
            for pair, (witness, index), a, b in zip(pairs, witnesses, first, second, strict=True)
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    return solve_contact_impulses(
        graph,
        normals,
        mu_static,
        mu_dynamic,
        velocity_tolerance=velocity_tolerance,
        response_tolerance=response_tolerance,
        position_tolerance=tol,
        max_iterations=max_iterations,
    )
