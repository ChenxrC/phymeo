"""Explicit asset-to-resident participant assembly, without a new solver.

The existing one-articulation SceneDeclaration is unaffected. This opt-in v1
lowers original assets, controllers and material installations to the public
ResidentSceneBatch boundary. Contact sampling remains HOST/discrete-entry.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from itertools import combinations
from threading import RLock
from types import MappingProxyType

import numpy as np

from ..backends import resident_scene
from ..backends._resident_affine_binding import ResidentAffineTreeBinding
from ..backends._resident_collision_plan import ResidentStaticBinding
from ..backends._resident_floating_affine_binding import ResidentFloatingAffineTreeBinding
from ..backends._resident_floating_binding import ResidentFloatingTreeBinding
from ..backends._resident_tree_binding import ResidentTreeBinding
from ..backends.resident_scene import (
    ResidentParticipant,
    ResidentParticipantBinding,
    ResidentParticipantKey,
    ResidentParticipantLayout,
)
from ..constraints.collision_shapes import CollisionKey, CollisionModel
from ..constraints.robot_contact import RobotContactMaterial
from ..dynamics.joint_mimic import MimicRobotAsset
from ..runtime.robot_scene import _asset_snapshot
from ._urdf_frames import FoldedURDFAsset
from .scene_compiler import (
    ArticulationWorld,
    FiniteStaticShape,
    _floating,
    _integer,
    _physical,
    _validated_control,
    _world_spec,
)

_SCHEMA = "phymeo-participant-scene-v1"
_BINDINGS = dict(
    fixed=ResidentTreeBinding,
    affine=ResidentAffineTreeBinding,
    floating=ResidentFloatingTreeBinding,
    floating_affine=ResidentFloatingAffineTreeBinding,
)


def _encoded(value):
    """Deterministic typed identity, including original derived asset metadata.

    This is a digest input, not a deserializer. No object reconstruction/eval is
    supported. Infinite source metadata limits remain explicit; NaN is invalid.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or np.isnan(value).any():
            raise TypeError("scene identity requires real numeric array data without NaN")
        return ["array", value.dtype.str, value.shape, value.tobytes(order="C").hex()]
    if isinstance(value, np.generic):
        return ["scalar", value.dtype.str, _encoded(value.item())]
    if value is None or type(value) in (str, bool, int):
        return [type(value).__name__, value]
    if type(value) is float:
        if np.isnan(value):
            raise ValueError("scene identity cannot contain NaN")
        return ["float", value.hex()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if is_dataclass(value) and not isinstance(value, type):
        entries = {f.name: getattr(value, f.name) for f in fields(value)}
        entries.update(getattr(value, "__dict__", {}))
        return [
            "dataclass",
            type(value).__module__,
            type(value).__qualname__,
            [(key, _encoded(entries[key])) for key in sorted(entries)],
        ]
    if isinstance(value, Mapping):
        rows = [(_encoded(key), _encoded(item)) for key, item in value.items()]
        rows.sort(key=lambda row: json.dumps(row[0], separators=(",", ":")))
        return ["mapping", rows]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_encoded(item) for item in value]]
    if isinstance(value, (set, frozenset)):
        rows = [_encoded(item) for item in value]
        rows.sort(key=lambda row: json.dumps(row, separators=(",", ":")))
        return ["set", rows]
    raise TypeError("unsupported scene identity value: " + type(value).__name__)


def _digest(value):
    return hashlib.sha256(
        json.dumps(_encoded(value), separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _row_source(row):
    return (
        row.key,
        row.asset,
        row.initial_state,
        row.kp,
        row.kd,
        row.effort_limit,
        row.body_ids,
        row.collider_keys,
        row.initial_control,
        row._world,
    )


@dataclass(frozen=True, eq=False)
class ParticipantArticulation:
    """Original asset plus explicit full-body and collider installations."""

    key: ResidentParticipantKey
    asset: object
    initial_state: object
    kp: np.ndarray = field(kw_only=True)
    kd: np.ndarray = field(kw_only=True)
    effort_limit: np.ndarray = field(kw_only=True)
    body_ids: tuple[int, ...] = field(kw_only=True)
    collider_keys: Mapping = field(kw_only=True)
    initial_control: object = field(default=None, kw_only=True)
    _world: ArticulationWorld = field(init=False, repr=False)
    _identity: str = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.key) is not ResidentParticipantKey:
            raise TypeError("key must be a ResidentParticipantKey")
        world = ArticulationWorld(
            self.key.world,
            self.asset,
            self.initial_state,
            kp=self.kp,
            kd=self.kd,
            effort_limit=self.effort_limit,
            initial_control=self.initial_control,
        )
        physical = _physical(world.asset)
        body_ids = tuple(_integer(value, "physical body ID") for value in self.body_ids)
        count = len(physical.joint_names) + int(_floating(world.asset))
        if len(body_ids) != count or len(set(body_ids)) != count:
            raise ValueError("body_ids must cover every full physical body exactly once")
        if not isinstance(self.collider_keys, Mapping):
            raise TypeError("collider_keys must map original keys to installed keys")
        sources = [item.shape.key for item in physical.collisions]
        if len(set(sources)) != len(sources) or set(self.collider_keys) != set(sources):
            raise ValueError("collider installation must exactly cover unique original source keys")
        shapes = {shape.key: shape for shape in physical.collision_model.shapes}
        if set(shapes) != set(sources) or any(
            shapes[c.shape.key] != c.shape for c in physical.collisions
        ):
            raise ValueError("asset collider source and collision model disagree")
        installed = {}
        for source in sorted(sources):
            target = self.collider_keys[source]
            if (
                type(source) is not CollisionKey
                or type(target) is not CollisionKey
                or target.world != self.key.world
            ):
                raise ValueError(
                    "source/installed collider must be CollisionKey in the participant world"
                )
            installed[source] = target
        if len(set(installed.values())) != len(installed):
            raise ValueError("installed collider keys must be unique")
        for name in ("asset", "initial_state", "kp", "kd", "effort_limit", "initial_control"):
            object.__setattr__(self, name, getattr(world, name))
        object.__setattr__(self, "body_ids", body_ids)
        object.__setattr__(self, "collider_keys", MappingProxyType(installed))
        object.__setattr__(self, "_world", world)
        object.__setattr__(self, "_identity", _digest(_row_source(self)))

    def assert_unchanged(self):
        if _digest(_row_source(self)) != self._identity:
            raise ValueError("participant asset/control/installation source changed")


def _scene_source(scene):
    return (
        scene.schema,
        tuple(_row_source(row) for row in scene.articulations),
        scene.static_shapes,
        scene.static_body_ids,
        scene.materials_by_pair,
        scene.candidate_capacity,
        scene.site_capacity,
        scene.intra_participant_pairs,
    )


@dataclass(frozen=True, eq=False)
class ParticipantSceneDeclaration:
    """Explicit multi-articulation world assembly; pair None means exclusion."""

    articulations: tuple[ParticipantArticulation, ...]
    static_shapes: tuple[FiniteStaticShape, ...] = field(default=(), kw_only=True)
    static_body_ids: tuple[tuple[int, int], ...] = field(default=(), kw_only=True)
    materials_by_pair: Mapping | None = field(default=None, kw_only=True)
    candidate_capacity: int = field(default=4096, kw_only=True)
    site_capacity: object = field(default=128, kw_only=True)
    intra_participant_pairs: bool = field(default=False, kw_only=True)
    schema: str = field(default=_SCHEMA, kw_only=True)
    identity: str = field(init=False)

    def __post_init__(self):
        if self.schema != _SCHEMA:
            raise ValueError("unsupported participant scene schema")
        rows = tuple(self.articulations)
        if not rows or any(type(row) is not ParticipantArticulation for row in rows):
            raise TypeError("scene requires nonempty ParticipantArticulation declarations")
        if len({row.key for row in rows}) != len(rows):
            raise ValueError("duplicate logical participant identity")
        rows = tuple(sorted(rows, key=lambda row: row.key))
        for row in rows:
            row.assert_unchanged()
        if type(self.intra_participant_pairs) is not bool:
            raise TypeError("intra_participant_pairs must be bool")
        static = tuple(self.static_shapes)
        if any(type(item) is not FiniteStaticShape for item in static):
            raise TypeError("static_shapes requires FiniteStaticShape")
        if any(item.material is not None for item in static):
            raise ValueError("move static default material into explicit materials_by_pair")
        static = tuple(
            sorted((_asset_snapshot(item) for item in static), key=lambda item: item.shape.key)
        )
        if len({item.shape.key for item in static}) != len(static):
            raise ValueError("duplicate static collider identity")
        body_ids = []
        for raw in self.static_body_ids:
            pair = tuple(raw)
            if len(pair) != 2:
                raise ValueError("static body identity needs world and body")
            body_ids.append(tuple(_integer(value, "static body identity") for value in pair))
        if len(set(body_ids)) != len(body_ids):
            raise ValueError("duplicate static body identity")
        materials = {} if self.materials_by_pair is None else self.materials_by_pair
        if not isinstance(materials, Mapping):
            raise TypeError("materials_by_pair must be an explicit mapping")
        owned_laws = {}
        for pair, law in materials.items():
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or any(type(key) is not CollisionKey for key in pair)
                or not pair[0] < pair[1]
            ):
                raise ValueError("materials require canonical ordered collision pairs")
            if law is not None and not isinstance(law, RobotContactMaterial):
                raise TypeError("pair material must be RobotContactMaterial or explicit None")
            owned_laws[pair] = _asset_snapshot(law)
        capacity = _integer(self.candidate_capacity, "candidate capacity")
        if capacity > 32640:
            raise ValueError("candidate capacity exceeds the original resident bound")
        worlds = tuple(sorted({row.key.world for row in rows}))
        if isinstance(self.site_capacity, Mapping):
            if any(
                _integer(key, "site capacity world") != key for key in self.site_capacity
            ) or set(self.site_capacity) != set(worlds):
                raise ValueError("site capacities must cover all logical worlds")
            site_capacity = MappingProxyType(
                {w: _integer(self.site_capacity[w], "site capacity") for w in worlds}
            )
            caps = tuple(site_capacity.values())
        else:
            site_capacity = _integer(self.site_capacity, "site capacity")
            caps = (site_capacity,)
        if any(cap > 4096 for cap in caps):
            raise ValueError("site capacity exceeds the original resident bound")
        for name, value in (
            ("articulations", rows),
            ("static_shapes", static),
            ("static_body_ids", tuple(sorted(body_ids))),
            ("materials_by_pair", MappingProxyType(dict(sorted(owned_laws.items())))),
            ("candidate_capacity", capacity),
            ("site_capacity", site_capacity),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "identity", _digest(_scene_source(self)))

    def assert_unchanged(self):
        for row in self.articulations:
            row.assert_unchanged()
        if _digest(_scene_source(self)) != self.identity:
            raise ValueError("participant scene declaration changed")


def _fixed_group(asset, collider):
    if isinstance(asset, FoldedURDFAsset):
        site = asset.sites.get(collider.link_name)
        if site is None:
            raise ValueError("folded collider lacks original authored owner site")
        return site.link_index
    return collider.link_index


def _prepare(declaration):
    """Complete source/geometry/pair preflight before any resident allocation."""
    declaration.assert_unchanged()
    specs, participants, source_map = {}, [], {}
    for row in declaration.articulations:
        system = _world_spec(row._world)
        participant = ResidentParticipant(row.key, system, row.body_ids)
        specs[row.key] = system
        participants.append(participant)
        source_map[row.key] = row.collider_keys
    layout = ResidentParticipantLayout(
        tuple(participants), static_body_ids=declaration.static_body_ids
    )
    by_key = {participant.key: participant for participant in participants}
    shapes, bindings, owners, excluded = [], [], {}, set()
    for row in declaration.articulations:
        asset, physical = row.asset, _physical(row.asset)
        participant = by_key[row.key]
        static_paths = None
        for collider in physical.collisions:
            source, target = collider.shape.key, row.collider_keys[collider.shape.key]
            shape = replace(collider.shape, key=target)
            if not _floating(asset) and (
                _fixed_group(physical, collider) is None or _fixed_group(physical, collider) < 0
            ):
                if (target.world, target.body) not in declaration.static_body_ids:
                    raise ValueError("fixed-root collider needs explicit static_body_ids ownership")
                if static_paths is None:
                    q = (
                        asset.full_state(row.initial_state).q
                        if isinstance(asset, MimicRobotAsset)
                        else row.initial_state.q
                    )
                    static_paths = physical.collision_motion(q)
                binding = ResidentStaticBinding(target, static_paths[source])
                owner = None
            else:
                binding = _BINDINGS[participant.family].from_asset(
                    asset, source, key=target, world=row.key.world
                )
                group = binding.group_index
                if group >= len(row.body_ids) or row.body_ids[group] != target.body:
                    raise ValueError("installed collider body differs from actual physical group")
                owner = (row.key, group)
                binding = ResidentParticipantBinding(row.key, binding)
            if target in owners:
                raise ValueError("installed collider identity aliases another source")
            owners[target] = owner
            shapes.append(shape)
            bindings.append(binding)
        for pair in physical.collision_model.excluded:
            excluded.add(tuple(sorted(row.collider_keys[key] for key in pair)))
    for item in declaration.static_shapes:
        key = item.shape.key
        if key in owners:
            raise ValueError("static collider aliases an installed asset collider")
        if (key.world, key.body) not in declaration.static_body_ids:
            raise ValueError("static collider requires explicit static_body_ids ownership")
        from ..constraints.collision_shapes import CollisionMotion

        shapes.append(item.shape)
        bindings.append(ResidentStaticBinding(key, CollisionMotion(item.points_m)))
        owners[key] = None
    if not shapes:
        if declaration.materials_by_pair:
            raise ValueError("empty geometry requires an empty pair material table")
        return dict(
            participants=tuple(participants),
            specs=MappingProxyType(specs),
            model=None,
            bindings=(),
            laws=MappingProxyType({}),
            source_map=MappingProxyType(source_map),
        )
    model = CollisionModel(tuple(shapes), tuple(sorted(excluded)))
    potential = []
    adjacent = set()
    for first, second in combinations(model.shapes, 2):
        a, b = first.key, second.key
        pair = a, b
        if (
            a.world != b.world
            or a.body == b.body
            or pair in excluded
            or not (first.group & second.mask)
            or not (second.group & first.mask)
        ):
            continue
        left, right = owners[a], owners[b]
        if left is None and right is None:
            continue
        if left is not None and right is not None and left[0] == right[0]:
            if not declaration.intra_participant_pairs:
                continue
            compiled = by_key[left[0]].compiled
            parents = getattr(compiled, "physical", compiled).parents
            if parents[left[1]] == right[1] or parents[right[1]] == left[1]:
                adjacent.add(pair)
        potential.append(pair)
    if set(declaration.materials_by_pair) != set(potential):
        raise ValueError(
            "materials_by_pair must exactly cover allowed source pairs; None is explicit exclusion"
        )
    laws = {}
    for pair in potential:
        law = declaration.materials_by_pair[pair]
        if law is None:
            excluded.add(pair)
        elif pair in adjacent:
            raise ValueError("direct parent-child pair requires explicit exclusion or mask")
        else:
            laws[pair] = law
    if len(laws) > declaration.candidate_capacity:
        raise ValueError("complete candidate domain exceeds candidate capacity")
    model = CollisionModel(model.shapes, tuple(sorted(excluded)))
    # Public static registration validates every provenance token and full body
    # binding. The real batch repeats its normal admission without a new solver.
    layout.bind_geometry(model, tuple(bindings))
    declaration.assert_unchanged()
    return dict(
        participants=tuple(participants),
        specs=MappingProxyType(specs),
        model=model,
        bindings=tuple(bindings),
        laws=MappingProxyType(laws),
        source_map=MappingProxyType(source_map),
    )


def _input(row, spec, command):
    joint = command.joints if _floating(row.asset) else command
    n = len(row.asset.joint_names)
    target = np.zeros(n) if joint is None else joint.target_q
    rate = np.zeros(n) if joint is None or joint.target_velocity is None else joint.target_velocity
    effort = np.zeros(n) if joint is None or joint.feedforward is None else joint.feedforward
    if _floating(row.asset):
        return spec.inputs(np.column_stack((target, rate, effort)), external=command.external)
    if isinstance(row.asset, MimicRobotAsset):
        return spec.inputs(target, rate, feedforward=effort)
    return spec.program.inputs(target, rate, feedforward=effort)


class CompiledParticipantScene:
    """Owned public resident scene, preserving complete logical world commits.

    Checkpoints belong to this batch owner; they are not portable declarations.
    No mechanics state or controls are shadowed after restore/reset. All physical
    mutations and publication are delegated once to the existing public API.
    """

    def __init__(self, declaration, *, mode="cuda", cache_root=None):
        if type(declaration) is not ParticipantSceneDeclaration:
            raise TypeError("compile requires ParticipantSceneDeclaration")
        if mode not in ("interpret", "cuda"):
            raise ValueError("participant scene mode must be interpret or cuda")
        prepared = _prepare(declaration)
        self._lock = RLock()
        self._closed = False
        self._declaration = declaration
        self._rows = MappingProxyType({row.key: row for row in declaration.articulations})
        self._specs = prepared["specs"]
        self._source_map = prepared["source_map"]
        # Initial command conversion also completes before allocating an owner.
        initial = {
            key: _input(row, self._specs[key], row.initial_control)
            for key, row in self._rows.items()
        }
        self._batch = resident_scene.ResidentSceneBatch(
            prepared["participants"],
            mode=mode,
            cache_root=cache_root,
            static_body_ids=declaration.static_body_ids,
        )
        try:
            self._plan = self._batch.prepare_contacts(
                prepared["model"],
                prepared["bindings"],
                prepared["laws"],
                candidate_capacity=declaration.candidate_capacity,
                site_capacity=declaration.site_capacity,
                intra_participant_pairs=declaration.intra_participant_pairs,
                sampling="discrete_entry",
                narrow_phase="host_patch",
            )
            self._batch.set_controls(initial)
        except BaseException as primary:
            try:
                self._batch.close()
            except BaseException as cleanup:
                failure = BaseExceptionGroup(
                    "participant assembly and owner cleanup failed", [primary, cleanup]
                )
                # Keep explicit ownership reachable without relying on traceback or GC.
                # Unresolved does not authorize an automatic retry of an unknown cleanup.
                failure.resident_owner = self._batch
                failure.cleanup_state = "unresolved"
                raise failure from None
            self._closed = True
            raise

    def _check(self):
        if self._closed:
            raise RuntimeError("compiled participant scene is closed")
        self._declaration.assert_unchanged()

    @property
    def declaration(self):
        return self._declaration

    @property
    def identity(self):
        return self._declaration.identity

    @property
    def source_map(self):
        return self._source_map

    @property
    def layout(self):
        return self._batch.layout

    @property
    def world_keys(self):
        return self._batch.world_keys

    @property
    def participant_keys(self):
        return self._batch.participant_keys

    @property
    def state_epoch(self):
        with self._lock:
            self._check()
            return self._batch.state_epoch

    @property
    def last_accepted(self):
        with self._lock:
            self._check()
            return self._plan.last_accepted

    @property
    def last_status(self):
        with self._lock:
            self._check()
            return self._plan.last_status

    def snapshot(self, worlds=None):
        with self._lock:
            self._check()
            return self._batch.snapshot(worlds)

    def set_controls(self, controls):
        with self._lock:
            self._check()
            if not isinstance(controls, Mapping) or any(
                type(key) is not ResidentParticipantKey or key not in self._rows for key in controls
            ):
                raise ValueError("controls must map registered participant identities")
            owned = {
                key: _validated_control(self._rows[key]._world, value)
                for key, value in controls.items()
            }
            native = {
                key: _input(self._rows[key], self._specs[key], value)
                for key, value in owned.items()
            }
            self._batch.set_controls(native)

    def step(self, h):
        with self._lock:
            self._check()
            return self._batch.step_contacts(h, self._plan)

    def restore(self, checkpoints, *, expected_epoch):
        with self._lock:
            self._check()
            return self._batch.restore(checkpoints, expected_epoch=expected_epoch)

    def reset(self, worlds=None):
        with self._lock:
            self._check()
            return self._batch.reset(worlds)

    def stats(self):
        with self._lock:
            return dict(
                self._batch.stats(),
                assembly_schema=_SCHEMA,
                assembly_identity=self.identity,
                narrow_phase="host_patch",
                sampling="discrete_entry",
                assembly_closed=self._closed,
            )

    def close(self):
        with self._lock:
            if not self._closed:
                self._batch.close()
                self._closed = True

    def __enter__(self):
        with self._lock:
            self._check()
            return self

    def __exit__(self, *_):
        self.close()


def compile_participant_scene(declaration, *, mode="cuda", cache_root=None):
    """Compile the explicit versioned asset assembly; no default-route override."""
    return CompiledParticipantScene(declaration, mode=mode, cache_root=cache_root)


__all__ = [
    "ParticipantArticulation",
    "ParticipantSceneDeclaration",
    "CompiledParticipantScene",
    "compile_participant_scene",
]
