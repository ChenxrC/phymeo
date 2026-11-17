"""Frozen native-source identities and cold, explicit exclusion planning.

Keys describe the imported MjSpec *before* static-body fusion, not native body
IDs or resident articulation keys. Plans do not alter a model or select a
runtime. A backend must explicitly qualify and consume them before execution.
"""

import json
from dataclasses import dataclass
from hashlib import sha256

from ..constraints.collision_shapes import CollisionKey


def _digest(value):
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SourceGeometry:
    key: CollisionKey
    native_geom_id: int
    body_name: str | None
    geom_name: str | None
    native_body_id: int


@dataclass(frozen=True)
class CollisionExclusionPlan:
    """Validated declaration only; no collision filtering has been applied."""

    source_identity: str
    excluded: tuple[tuple[CollisionKey, CollisionKey], ...]
    native_geom_pairs: tuple[tuple[int, int], ...]
    fingerprint: str


@dataclass(frozen=True)
class CollisionSourceMapping:
    """Snapshot-bound keys; body/shape ordinals belong to the imported spec.

    A shape ordinal is local to its precompile body, including retained visual
    geoms. It is not an ordinal in the original URDF collision XML elements.
    Names are diagnostics, never an implicit ambiguous lookup mechanism.
    """

    identity: str
    instance: str
    world: int
    geoms: tuple[SourceGeometry, ...]
    explicit_pairs: tuple[tuple[int, int], ...]
    issues: tuple[str, ...] = ()

    def plan_exclusions(self, excluded, *, source_identity):
        """Resolve owned, canonical pairs; reject stale or unsupported requests."""
        if source_identity != self.identity:
            raise ValueError("collision source identity mismatch; resolve keys again")
        if self.issues:
            raise ValueError("collision source mapping unavailable: " + "; ".join(self.issues))
        by_key = {row.key: row.native_geom_id for row in self.geoms}
        explicit = set(self.explicit_pairs)
        pairs = set()
        for item in excluded:
            pair = tuple(item)
            if len(pair) != 2 or not all(isinstance(key, CollisionKey) for key in pair):
                raise ValueError("exclusion requires two CollisionKey references")
            a, b = pair
            if a.world != self.world or b.world != self.world:
                raise ValueError("cross-world collision exclusion is not supported")
            if a == b or a not in by_key or b not in by_key:
                raise ValueError("invalid excluded shape pair: self or unknown source key")
            canonical = tuple(sorted(pair))
            if canonical in pairs:
                raise ValueError("duplicate excluded shape pair")
            native = tuple(sorted((by_key[a], by_key[b])))
            if native in explicit:
                raise ValueError("collision exclusion conflicts with an explicit contact pair")
            pairs.add(canonical)
        canonical = tuple(sorted(pairs))
        native = tuple(tuple(sorted((by_key[a], by_key[b]))) for a, b in canonical)
        fingerprint = _digest(("collision-exclusions-v1", self.identity, native))
        return CollisionExclusionPlan(self.identity, canonical, native, fingerprint)


def validate_scope(instance, world):
    if not isinstance(instance, str) or not instance.strip():
        raise ValueError("collision source instance must be a nonempty string")
    return CollisionKey(world, 0, 0).world


def capture_source_geometry(spec, world):
    """Keep native object references across compile without editing their names."""
    try:
        # MjSpec wrappers preserve object identity; native IDs are assigned by
        # compile, whereas parent links can already be fused away afterwards.
        bodies = tuple(spec.bodies)
        body_indices = {id(body): index for index, body in enumerate(bodies)}
        counts = {}
        captured = []
        for geom in spec.geoms:
            parent = geom.parent
            index = body_indices[id(parent)]
            ordinal = counts.get(index, 0)
            counts[index] = ordinal + 1
            captured.append(
                (geom, CollisionKey(world, index, ordinal), parent.name or None, geom.name or None)
            )
        return tuple(captured), ()
    except (AttributeError, KeyError) as exc:
        # Older/native importer capabilities must not change no-rule behavior.
        return (), (f"precompile source-object identity unavailable: {type(exc).__name__}",)


def freeze_source_geometry(captured, issues, model, source, instance, world):
    rows = []
    if not issues:
        try:
            for geom, key, body_name, geom_name in captured:
                native_id = int(geom.id)
                if not 0 <= native_id < model.ngeom:
                    raise ValueError("source geom has no compiled native ID")
                rows.append(
                    SourceGeometry(
                        key, native_id, body_name, geom_name, int(model.geom_bodyid[native_id])
                    )
                )
            if len(rows) != model.ngeom or len({row.native_geom_id for row in rows}) != model.ngeom:
                raise ValueError("source/native geom mapping is not bijective")
        except (AttributeError, ValueError) as exc:
            rows = []
            issues = (f"compiled source-object identity unavailable: {exc}",)
    explicit = tuple(sorted(
        tuple(sorted((int(a), int(b))))
        for a, b in zip(model.pair_geom1, model.pair_geom2, strict=True)
    ))
    identity = _digest((
        "collision-source-v1", source.primary_sha256, source.model_sha256,
        source.mujoco_version, instance, world,
        [
            (row.key.body, row.key.shape, row.native_geom_id, row.body_name, row.geom_name)
            for row in rows
        ],
        issues,
    ))
    return CollisionSourceMapping(identity, instance, world, tuple(rows), explicit, tuple(issues))
