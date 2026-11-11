"""Static participant identity and owned layout; no runtime or device allocation.

This module reuses the four qualified initial-state compilers. Geometry binding
is registration only: it neither samples contacts nor advances physical state.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionModel, CollisionMotion
from ._resident_affine_binding import ResidentAffineTreeBinding, compile_affine_tree_binding
from ._resident_affine_tree import AffineRobotTree, compile_affine_tree_world
from ._resident_collision_plan import ResidentStaticBinding
from ._resident_floating_affine_binding import (
    ResidentFloatingAffineTreeBinding,
    compile_floating_affine_tree_binding,
)
from ._resident_floating_affine_tree import (
    FloatingAffineRobotTree,
    compile_floating_affine_tree_world,
)
from ._resident_floating_binding import (
    ResidentFloatingTreeBinding,
    compile_floating_tree_binding,
)
from ._resident_floating_tree import FloatingRobotTree, compile_floating_tree_world
from ._resident_tree_binding import ResidentTreeBinding, compile_tree_binding
from ._resident_tree_world import FixedRobotTree, compile_tree_world

MAX_PARTICIPANTS_PER_WORLD = 8
_I32_MAX = 2**31 - 1
_FAMILIES = {
    FixedRobotTree: ("fixed", compile_tree_world, ResidentTreeBinding, compile_tree_binding),
    AffineRobotTree: (
        "affine",
        compile_affine_tree_world,
        ResidentAffineTreeBinding,
        compile_affine_tree_binding,
    ),
    FloatingRobotTree: (
        "floating",
        compile_floating_tree_world,
        ResidentFloatingTreeBinding,
        compile_floating_tree_binding,
    ),
    FloatingAffineRobotTree: (
        "floating_affine",
        compile_floating_affine_tree_world,
        ResidentFloatingAffineTreeBinding,
        compile_floating_affine_tree_binding,
    ),
}
_BINDING_TYPES = tuple(value[2] for value in _FAMILIES.values())


def _integer(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 0 <= value <= _I32_MAX
    ):
        raise ValueError(f"{name} must be a nonnegative i32 integer")
    return int(value)


def _sequence(value, name):
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a finite list or tuple")
    return tuple(value)


def _freeze(value):
    """Exact immutable source representation, including dtype and raw array bytes."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot bind a participant")
        return ("array", value.dtype.str, value.shape, value.tobytes(order="C"))
    if isinstance(value, np.generic):
        return _freeze(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__module__,
            type(value).__qualname__,
            tuple((item.name, _freeze(getattr(value, item.name))) for item in fields(value)),
        )
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(sorted((_freeze(key), _freeze(item)) for key, item in value.items())),
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float, bytes)):
        return value
    raise TypeError(f"unsupported participant source field {type(value).__name__}")


def _source_signature(system):
    if type(system) is FixedRobotTree:
        return _freeze(
            (system.program.problem_signature(), system.state, system.control_max_iterations)
        )
    # Physical and graph sources are already retained in the compiled token.
    return _freeze(
        (
            system.state,
            system.kp,
            system.kd,
            system.effort_limit,
            system.delay_steps,
            system.control_max_iterations,
            getattr(system, "_graph_token", None),
        )
    )


def _compile(system):
    description = _FAMILIES.get(type(system))
    if description is None:
        raise TypeError("participant system must be one of the four resident tree declarations")
    return description[0], description[1](system)


def _copy_binding(binding):
    """Keep original from_asset provenance; dataclasses.replace alone loses it."""
    result = replace(binding)
    for item in fields(binding):
        if not item.init:
            object.__setattr__(result, item.name, getattr(binding, item.name))
    return result


@dataclass(frozen=True, order=True)
class ResidentParticipantKey:
    world: int
    participant: int

    def __post_init__(self):
        for name in ("world", "participant"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))


@dataclass(frozen=True, eq=False)
class ResidentParticipant:
    """Owned compiled snapshot plus an explicitly checked original declaration.

    The caller's model is not deep-frozen or modified. Its compiled data are
    bytes-backed readonly copies; require_current rejects later source changes.
    """

    key: ResidentParticipantKey
    system: object
    body_ids: tuple
    family: str = field(init=False)
    configuration_count: int = field(init=False)
    velocity_count: int = field(init=False)
    body_count: int = field(init=False)
    physical_configuration_count: int = field(init=False)
    physical_velocity_count: int = field(init=False)
    model_token: tuple = field(init=False, repr=False)
    graph_token: tuple | None = field(init=False, repr=False)
    fingerprint: tuple = field(init=False, repr=False)
    _compiled: object = field(init=False, repr=False)
    _source: tuple = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.key) is not ResidentParticipantKey:
            raise TypeError("participant needs a ResidentParticipantKey")
        ids = tuple(_integer(value, "body ID") for value in _sequence(self.body_ids, "body IDs"))
        if len(set(ids)) != len(ids):
            raise ValueError("participant body IDs must be unique")
        family, compiled = _compile(self.system)
        physical = getattr(compiled, "physical", compiled)
        floating = family.startswith("floating")
        independent = compiled.k if family in ("affine", "floating_affine") else compiled.n
        nq, nv = independent + (7 if floating else 0), independent + (6 if floating else 0)
        body_count = physical.n + int(floating)
        if len(ids) != body_count:
            raise ValueError("body IDs must cover every full physical body exactly once")
        signature = (_freeze(compiled), _source_signature(self.system))
        for name, value in (
            ("body_ids", ids),
            ("family", family),
            ("configuration_count", nq),
            ("velocity_count", nv),
            ("body_count", body_count),
            ("physical_configuration_count", physical.n + (7 if floating else 0)),
            ("physical_velocity_count", physical.n + (6 if floating else 0)),
            ("model_token", physical.model_token),
            ("graph_token", getattr(compiled, "graph_token", None)),
            ("_compiled", compiled),
            ("_source", signature),
            (
                "fingerprint",
                ("resident-participant-v1", self.key.world, self.key.participant, ids, signature),
            ),
        ):
            object.__setattr__(self, name, value)

    @property
    def compiled(self):
        """Original static compiler output; no current-state mechanics."""
        return self._compiled

    def require_current(self):
        try:
            family, compiled = _compile(self.system)
            signature = (_freeze(compiled), _source_signature(self.system))
        except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
            raise ValueError("participant source changed or became invalid") from error
        if family != self.family or signature != self._source:
            raise ValueError("participant source changed after registration")


@dataclass(frozen=True, eq=False)
class ResidentParticipantBinding:
    """Wrap an existing authored binding without changing its logical world."""

    participant: ResidentParticipantKey
    binding: object

    def __post_init__(self):
        if type(self.participant) is not ResidentParticipantKey:
            raise TypeError("binding needs a ResidentParticipantKey")
        if type(self.binding) not in _BINDING_TYPES:
            raise TypeError("participant binding requires an original resident tree binding")
        if (
            self.binding.world != self.participant.world
            or self.binding.key.world != self.participant.world
        ):
            raise ValueError("participant binding world differs from its logical owner")
        object.__setattr__(self, "binding", _copy_binding(self.binding))


@dataclass(frozen=True)
class ResidentParticipantLayoutEntry:
    key: ResidentParticipantKey
    family: str
    storage_row: int
    configuration_count: int
    velocity_count: int
    body_count: int
    physical_configuration_count: int
    physical_velocity_count: int
    configuration_offset: int
    velocity_offset: int
    body_offset: int
    topology: tuple
    model_token: tuple
    graph_token: tuple | None


@dataclass(frozen=True)
class ResidentParticipantWorldLayout:
    world: int
    participant_keys: tuple
    configuration_offset: int
    velocity_offset: int
    body_offset: int
    configuration_count: int
    velocity_count: int
    body_count: int


@dataclass(frozen=True, eq=False)
class ResidentParticipantLayout:
    """Canonical participant/state/body address tables, without a runtime.

    Offsets concatenate all participants in key order. World records delimit
    their contiguous spans; body_slot returns a world's local wrench row.
    Storage rows are per-family metadata, never caller-visible physical worlds.
    """

    participants: tuple
    static_body_ids: tuple = field(default=(), kw_only=True)
    participant_keys: tuple = field(init=False)
    world_keys: tuple = field(init=False)
    entries: Mapping = field(init=False, repr=False)
    worlds: Mapping = field(init=False, repr=False)
    family_rows: Mapping = field(init=False, repr=False)
    body_owners: Mapping = field(init=False, repr=False)
    body_keys: Mapping = field(init=False, repr=False)
    configuration_count: int = field(init=False)
    velocity_count: int = field(init=False)
    body_count: int = field(init=False)
    fingerprint: tuple = field(init=False, repr=False)

    def __post_init__(self):
        original = _sequence(self.participants, "participants")
        if not original:
            raise ValueError("participant layout must contain at least one world")
        if any(type(item) is not ResidentParticipant for item in original):
            raise TypeError("layout requires ResidentParticipant declarations")
        owners = {item.key: item for item in original}
        if len(owners) != len(original):
            raise ValueError("duplicate participant identity")
        participants = tuple(owners[key] for key in sorted(owners))
        by_world = {}
        for item in participants:
            by_world.setdefault(item.key.world, []).append(item)
        if any(len(items) > MAX_PARTICIPANTS_PER_WORLD for items in by_world.values()):
            raise ValueError("a world supports at most eight participants")

        entries, worlds, bodies, inverse = {}, {}, {}, {}
        families = {value[0]: [] for value in _FAMILIES.values()}
        q_offset = v_offset = body_offset = 0
        for world, members in by_world.items():
            start_q, start_v, start_body = q_offset, v_offset, body_offset
            for item in members:
                item.require_current()
                for group, body in enumerate(item.body_ids):
                    identity = (world, body)
                    if identity in bodies:
                        raise ValueError("body ID aliases different physical participants")
                    bodies[identity] = (item.key, group)
                    inverse[(item.key, group)] = identity
                family = families[item.family]
                entries[item.key] = ResidentParticipantLayoutEntry(
                    item.key,
                    item.family,
                    len(family),
                    item.configuration_count,
                    item.velocity_count,
                    item.body_count,
                    item.physical_configuration_count,
                    item.physical_velocity_count,
                    q_offset,
                    v_offset,
                    body_offset,
                    item.compiled.topology,
                    item.model_token,
                    item.graph_token,
                )
                family.append(item.key)
                q_offset += item.configuration_count
                v_offset += item.velocity_count
                body_offset += item.body_count
                if max(q_offset, v_offset, body_offset, len(entries)) > _I32_MAX:
                    raise ValueError("participant layout address capacity exceeds i32")
            worlds[world] = ResidentParticipantWorldLayout(
                world,
                tuple(item.key for item in members),
                start_q,
                start_v,
                start_body,
                q_offset - start_q,
                v_offset - start_v,
                body_offset - start_body,
            )

        static = []
        for raw in _sequence(self.static_body_ids, "static body IDs"):
            pair = _sequence(raw, "static body identity")
            if len(pair) != 2:
                raise ValueError("static body identity must contain world and body")
            identity = tuple(_integer(value, "static body identity") for value in pair)
            if identity[0] not in worlds or identity in bodies or identity in static:
                raise ValueError("static owner is unknown, duplicate, or aliases a dynamic body")
            static.append(identity)
        static = tuple(sorted(static))
        for name, value in (
            ("participants", participants),
            ("static_body_ids", static),
            ("participant_keys", tuple(entries)),
            ("world_keys", tuple(worlds)),
            ("entries", MappingProxyType(entries)),
            ("worlds", MappingProxyType(worlds)),
            (
                "family_rows",
                MappingProxyType({key: tuple(value) for key, value in families.items()}),
            ),
            ("body_owners", MappingProxyType(bodies)),
            ("body_keys", MappingProxyType(inverse)),
            ("configuration_count", q_offset),
            ("velocity_count", v_offset),
            ("body_count", body_offset),
            (
                "fingerprint",
                (
                    "resident-participant-layout-v1",
                    tuple(item.fingerprint for item in participants),
                    static,
                ),
            ),
        ):
            object.__setattr__(self, name, value)

    def require_current(self):
        for participant in self.participants:
            participant.require_current()

    def body_slot(self, participant, local_body):
        if type(participant) is not ResidentParticipantKey or participant not in self.entries:
            raise ValueError("unknown participant identity")
        local_body = _integer(local_body, "local body")
        row = self.entries[participant]
        if local_body >= row.body_count:
            raise ValueError("local body is outside the actual participant")
        return row.body_offset - self.worlds[participant.world].body_offset + local_body

    def bind_geometry(self, model, bindings):
        """Validate complete source ownership; no candidate query or contact law."""
        return _bind_geometry(self, model, bindings)


@dataclass(frozen=True, eq=False, init=False)
class ResidentParticipantGeometry:
    """Immutable geometry preflight output, not an accepted device query."""

    model: CollisionModel
    bindings: tuple
    shape_keys: tuple
    owners: Mapping
    compiled_bindings: Mapping
    fingerprint: tuple
    _layout: ResidentParticipantLayout = field(repr=False)
    _model_source: CollisionModel = field(repr=False)
    _binding_sources: tuple = field(repr=False)
    _source_signature: tuple = field(repr=False)

    def __init__(self):
        raise TypeError("geometry registration is constructed by layout.bind_geometry")

    def require_current(self, layout):
        if layout is not self._layout:
            raise ValueError("geometry registration belongs to another layout owner")
        layout.require_current()
        signature = (_freeze(self._model_source), _freeze(self._binding_sources))
        if signature != self._source_signature:
            raise ValueError("geometry registration source changed after preflight")


def _bind_geometry(layout, model, bindings):
    layout.require_current()
    if not isinstance(model, CollisionModel):
        raise TypeError("geometry registration requires a CollisionModel")
    source_bindings = _sequence(bindings, "geometry bindings")
    shapes = tuple(replace(shape) for shape in model.shapes)
    owned_model = CollisionModel(shapes, model.excluded)
    shapes = owned_model.shapes
    by_key = {}
    for value in source_bindings:
        if type(value) is ResidentParticipantBinding:
            key = value.binding.key
        elif type(value) is ResidentStaticBinding:
            key = value.key
        else:
            raise TypeError("geometry requires participant wrappers or explicit static bindings")
        if key in by_key:
            raise ValueError("duplicate geometry binding key")
        by_key[key] = value
    shape_keys = tuple(shape.key for shape in shapes)
    if set(by_key) != set(shape_keys):
        raise ValueError("geometry bindings must cover every shape exactly once")

    participants = {item.key: item for item in layout.participants}
    owners, compiled, registered = {}, {}, []
    for shape in shapes:
        key, value = shape.key, by_key[shape.key]
        if key.world not in layout.worlds:
            raise ValueError("shape world has no registered participants")
        if (
            shape.kind not in ("sphere", "capsule", "hull")
            or not shape.rigid
            or (shape.kind == "hull" and shape.radius != 0)
        ):
            raise ValueError("geometry requires finite sphere, capsule or unrounded hull")
        if type(value) is ResidentStaticBinding:
            if (key.world, key.body) not in layout.static_body_ids:
                raise ValueError("static shape requires an explicit non-dynamic static owner")
            motion = value.motion
            if (
                motion.end is not None
                or motion.turn is not None
                or len(motion.start) != len(shape.vertex_ids)
            ):
                raise ValueError("static binding must contain every vertex at one fixed pose")
            owned_motion = CollisionMotion(
                motion.start, body_orientation_xyzw=motion.body_orientation_xyzw
            )
            registered.append(ResidentStaticBinding(key, owned_motion))
            owners[key], compiled[key] = None, owned_motion
            continue
        if value.participant not in participants:
            raise ValueError("geometry binding refers to an unknown participant")
        participant = participants[value.participant]
        binding = value.binding
        description = _FAMILIES[type(participant.system)]
        if type(binding) is not description[2]:
            raise TypeError("geometry binding family differs from its physical participant")
        group = binding.group_index
        if group >= participant.body_count:
            raise ValueError("binding group exceeds the actual full physical body list")
        if participant.body_ids[group] != key.body:
            raise ValueError("binding body ID differs from the participant/local-body bijection")
        compiled[key] = description[3](shape, binding, participant.compiled)
        owners[key] = (participant.key, group)
        registered.append(ResidentParticipantBinding(participant.key, binding))

    signature = (_freeze(model), _freeze(source_bindings))
    result = object.__new__(ResidentParticipantGeometry)
    for name, value in (
        ("model", owned_model),
        ("bindings", tuple(registered)),
        ("shape_keys", shape_keys),
        ("owners", MappingProxyType(owners)),
        ("compiled_bindings", MappingProxyType(compiled)),
        (
            "fingerprint",
            (
                "resident-participant-geometry-v1",
                layout.fingerprint,
                _freeze(owned_model),
                _freeze(tuple(registered)),
            ),
        ),
        ("_layout", layout),
        ("_model_source", model),
        ("_binding_sources", source_bindings),
        ("_source_signature", signature),
    ):
        object.__setattr__(result, name, value)
    return result
