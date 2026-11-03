"""Certified first closed approach and first strict entry for affine VT/EE material paths."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction as F

from ..constraints.collision_shapes import CollisionKey, CollisionModel
from ..core.validation import finite_real
from ._deformable_feature_roots import RootBudget, RootLimit, locate_threshold
from ._prism_pair_exact import sqrt_bounds
from ._sweep_polynomial import affine_vectors
from ._triangle_pair_sweep import _Pose
from .collision_pipeline import CollisionWorkspace, _material_face_broad_pairs
from .deformable_features import (
    DeformableFeatureContact,
    _capacity,
    _Geometry,
    _lower2,
    _measure,
    _members,
    _neighborhoods,
    _round_out,
    _topology,
)


@dataclass(frozen=True)
class DeformableSweepEvent:
    """A proved event-time enclosure and material geometry at the explicitly reported sample time."""

    interval: tuple[float, float]
    sample_time: float
    sample: DeformableFeatureContact
    sample_relation: str
    sample_gap_to_threshold_m: float
    distance_error_bound_m: float


@dataclass(frozen=True)
class DeformableThresholdSweep:
    threshold_m: float
    start_relation: str
    end_relation: str
    first_touch: DeformableSweepEvent | None
    first_entry: DeformableSweepEvent | None


@dataclass(frozen=True)
class DeformableFeaturePath:
    kind: str
    first_nodes: tuple
    second_nodes: tuple
    support_face_pairs: tuple
    search: DeformableThresholdSweep
    contact: DeformableThresholdSweep


@dataclass(frozen=True)
class DeformableFeatureSweeps:
    status: str
    reason: str
    backend: str
    paths: tuple[DeformableFeaturePath, ...] = ()
    broad_face_pairs: int = 0
    candidate_features: int = 0
    feature_tests: int = 0
    root_nodes: int = 0
    required_features: int | None = None
    invalid_faces: tuple[CollisionKey, ...] = ()
    narrow_backend: str = "host"

    @property
    def first_force_entry_interval(self):
        bounds = [
            p.contact.first_entry.interval for p in self.paths if p.contact.first_entry is not None
        ]
        return (min(lo for lo, _ in bounds), min(hi for _, hi in bounds)) if bounds else None


def _path_box(nodes, starts, steps):
    points = [p for n in nodes for p in (starts[n], starts[n] + steps[n])]
    return tuple((min(v), max(v)) for v in zip(*points, strict=True))


def _event(location, strict, key, support, radius, force, speed, starts, steps, faces):
    if location is None:
        return None
    cut, t = location
    time = float(t)
    actual = F(time)
    if actual != t:
        raise RootLimit("witness_time_precision")
    kind, first, second = key
    pts = {n: starts[n] + actual * steps[n] for n in (*first, *second)}
    poses = {}
    if kind == "VT":
        f = faces[second]
        poses[second] = replace(f, pose=_Pose(f.path, actual))
    row, d2 = _measure(key, support, pts, poses, None, force, exact=True)
    relation = (
        "inside" if d2 < radius * radius else "boundary" if d2 == radius * radius else "outside"
    )
    if (strict and relation != "inside") or (
        cut.lo == cut.hi == actual and not strict and relation == "outside"
    ):
        raise RootLimit("feature_witness_unresolved")
    # Distance between two convex material features changes no faster than the
    # sum of their maximum nodal speeds. The L1 speed is a rational upper bound.
    error = speed * max(abs(actual - cut.lo), abs(actual - cut.hi))
    dlo, dhi = sqrt_bounds(d2, 96)
    return DeformableSweepEvent(
        (_round_out(cut.lo, False), _round_out(cut.hi, True)),
        time,
        row,
        relation,
        float((dlo + dhi) / 2 - radius),
        _round_out(error, True),
    )


def _trajectory(key, support, search, force, starts, steps, symbols, faces, bits, budget):
    kind, first, second = key
    a, b = tuple(symbols[n] for n in first), tuple(symbols[n] for n in second)
    speed = sum(max(sum(abs(v) for v in steps[n]) for n in side) for side in (first, second))

    def layer(radius):
        start, end, touch, entry = locate_threshold(kind, a, b, radius, bits, budget)
        return DeformableThresholdSweep(
            float(radius),
            start,
            end,
            _event(touch, False, key, support, radius, force, speed, starts, steps, faces),
            _event(entry, True, key, support, radius, force, speed, starts, steps, faces),
        )

    reach = layer(search)
    if reach.first_touch is None:
        return None
    active = reach if search == force else layer(force)
    return DeformableFeaturePath(kind, first, second, tuple(sorted(support)), reach, active)


def sweep_deformable_features(
    model,
    motion,
    workspace,
    *,
    search_distance_m,
    contact_distance_m,
    topology_rings=2,
    max_feature_tests=65536,
    feature_capacity=32768,
    max_root_nodes=131072,
    time_bits=36,
):
    """Query affine material-node paths on normalized time [0,1], including isolated equality.

    first_touch is the first point in the closed threshold. first_entry
    brackets the infimum of strict membership and carries a verified interior
    material sample. An equality-only tangency has no first_entry. On a fully
    successful result, omitted features never touched the search threshold.
    """
    if (
        not isinstance(model, CollisionModel)
        or not isinstance(workspace, CollisionWorkspace)
        or workspace._model is not model
    ):
        raise ValueError("workspace must belong to the collision model")
    if not isinstance(motion, Mapping) or set(motion) != {s.key for s in model.shapes}:
        raise ValueError("motion keys must cover the collision model")
    search = F(finite_real(search_distance_m, "search_distance_m"))
    force = F(finite_real(contact_distance_m, "contact_distance_m"))
    if search <= 0 or not 0 <= force <= search:
        raise ValueError(
            "require 0 <= contact_distance_m <= search_distance_m and positive search distance"
        )
    rings = _capacity(topology_rings, 1, 8, "topology_rings")
    max_tests = _capacity(max_feature_tests, 0, 1_000_000, "max_feature_tests")
    cap = _capacity(feature_capacity, 0, 262144, "feature_capacity")
    bits = _capacity(time_bits, 8, 46, "time_bits")
    budget = RootBudget(_capacity(max_root_nodes, 0, 4_000_000, "max_root_nodes"))
    backend = workspace._backend
    with workspace._lock:
        if workspace._closed:
            raise RuntimeError("collision workspace is closed")
        motion = dict(motion)
        try:
            faces, by_nodes, starts, graph = _topology(model, motion, affine=True)
        except _Geometry as exc:
            return DeformableFeatureSweeps(exc.status, exc.reason, backend, invalid_faces=exc.keys)
        broad = (
            _material_face_broad_pairs(model, motion, workspace, float(search))
            if len(faces) > 1
            else ()
        )
    near = _neighborhoods(graph, rings)
    candidates = {}
    for i, j in broad:
        for key, support in _members(faces[i], faces[j], near):
            if key not in candidates:
                if len(candidates) == max_tests:
                    return DeformableFeatureSweeps(
                        "capacity_exceeded",
                        "feature_test_capacity",
                        backend,
                        broad_face_pairs=len(broad),
                        candidate_features=len(candidates) + 1,
                    )
                candidates[key] = set()
            candidates[key].add(support)
    steps = {
        (face.key.world, face.key.body, id_): v
        for face in faces
        for id_, v in zip(face.path.ids, face.path.step, strict=True)
    }
    symbols = {n: affine_vectors(p[None, :], steps[n][None, :])[0] for n, p in starts.items()}
    bounds = {side: _path_box(side, starts, steps) for _, a, b in candidates for side in (a, b)}
    count = tests = 0
    retained = []
    try:
        for key, support in sorted(candidates.items()):
            # Equality can be the only event, so use strict AABB rejection.
            if _lower2(bounds[key[1]], bounds[key[2]]) > search * search:
                continue
            tests += 1
            row = _trajectory(
                key, support, search, force, starts, steps, symbols, by_nodes, bits, budget
            )
            if row is not None:
                count += 1
                if count <= cap:
                    retained.append(row)
    except RootLimit as exc:
        return DeformableFeatureSweeps(
            "unresolved",
            str(exc),
            backend,
            broad_face_pairs=len(broad),
            candidate_features=len(candidates),
            feature_tests=tests,
            root_nodes=budget.used,
        )
    info = dict(
        broad_face_pairs=len(broad),
        candidate_features=len(candidates),
        feature_tests=tests,
        root_nodes=budget.used,
        required_features=count,
    )
    if count > cap:
        return DeformableFeatureSweeps("capacity_exceeded", "feature_capacity", backend, **info)
    return DeformableFeatureSweeps("ok", "complete", backend, tuple(retained), **info)
