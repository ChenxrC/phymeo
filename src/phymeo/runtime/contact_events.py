"""Lossless, immutable views of accepted ordinary and robot contact loads.

This module observes positive-normal-load episodes; it adds no friction state or
restitution law. A record is a shape pair. Its ordered components are the actual
sampler rows, not persistent feature IDs and not an averaged contact normal.
All forces, application points and material coordinates belong to sample_time_s.
"""

from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..constraints.robot_contact import RobotContactMaterial
from ..core.validation import finite_real
from ..dynamics.rotation import quaternion_matrix
from ..solvers.collision_loads import (
    NormalLayer,
    PairLoad,
    ParticleAttachment,
    RigidAttachment,
    SurfaceLoads,
)
from .contact_history import (
    ContactHistory,
    ContactIdentity,
    ContactMaterialAddress,
    ContactMaterialSite,
    _owned_load,
    _support,
    _values,
    _vector,
)


@dataclass(frozen=True)
class ContactOwner:
    """Registered physical owner and the frame used by its material coordinates.

    A robot's collision body ID may be remapped. body is its registered scene
    body ID; link_index/link_name identify the actual computational rigid link.
    Folded fixed descendants use that link's frame, not their authored URDF frame.
    """

    world: int
    body: int
    kind: str
    local_frame: str
    link_index: int | None = None
    link_name: str | None = None


@dataclass(frozen=True)
class ContactComponent:
    """One unchanged source PairLoad and its complete bilateral material sites.

    material_kind identifies the original law and its units. normal_layer uses
    an opening width and rate factor; robot_contact_material uses penetration
    and a dashpot. Their gap/force/energy values are never converted here.
    """

    load: PairLoad
    material: tuple[ContactMaterialSite, ...]
    material_kind: str
    law: NormalLayer | RobotContactMaterial


@dataclass(frozen=True)
class SceneContactRecord:
    identity: ContactIdentity
    first_owner: ContactOwner
    second_owner: ContactOwner
    components: tuple[ContactComponent, ...]
    material_changed: bool = False
    support_changed: bool = False


@dataclass(frozen=True)
class SceneContactEvent:
    kind: str
    identity: ContactIdentity
    record: SceneContactRecord | None
    reason: str | None = None


@dataclass(frozen=True)
class SceneContactEvents:
    """One committed sample and bounded per-pair generation tombstones.

    Any component with positive normal load activates its pair; every current
    component of that pair is retained, including any zero-force component.
    End events carry no stale force. An empty initial snapshot has zero times.
    """

    step_index: int = 0
    sample_time_s: float = 0.0
    commit_time_s: float = 0.0
    active: tuple[SceneContactRecord, ...] = ()
    events: tuple[SceneContactEvent, ...] = ()
    _generations: tuple[ContactIdentity, ...] = field(default=(), repr=False)

    @property
    def generations(self):
        return self._generations


def _ordinary_owner(key, bindings):
    binding = bindings[key]
    if isinstance(binding, RigidAttachment):
        return ContactOwner(key.world, key.body, "rigid", "rigid_com")
    if isinstance(binding, ParticleAttachment):
        return ContactOwner(key.world, key.body, "particle", "material_vertices")
    raise TypeError("expected rigid or particle material attachment")


def contact_events_from_history(history, *, layers, bindings):
    """Read an already committed B1 history without resampling or advancing it."""
    if not isinstance(history, ContactHistory):
        raise TypeError("expected ContactHistory")
    records = {}
    for old in history.active:
        first, second = old.identity.first, old.identity.second
        law = layers[first, second]
        if not isinstance(law, NormalLayer):
            raise TypeError("ordinary history requires its original NormalLayer")
        records[old.identity] = SceneContactRecord(
            old.identity,
            _ordinary_owner(first, bindings),
            _ordinary_owner(second, bindings),
            (ContactComponent(old.load, old.material, "normal_layer", law),),
            old.material_changed,
            old.support_changed,
        )
    return SceneContactEvents(
        history.step_index,
        history.sample_time_s,
        history.commit_time_s,
        tuple(records[r.identity] for r in history.active),
        tuple(
            SceneContactEvent(
                e.kind, e.identity, None if e.record is None else records[e.identity], e.reason
            )
            for e in history.events
        ),
        history.generations,
    )


@dataclass(frozen=True)
class _OwnerFrame:
    owner: ContactOwner
    vertex_ids: frozenset[int]
    row: int | None = None


def _index(value, name, *, minimum=0):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < minimum
    ):
        raise ValueError("invalid " + name)
    return int(value)


def _rotation(value):
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (3, 3)
        or not np.isfinite(result).all()
        or not np.allclose(result.T @ result, np.eye(3), atol=1e-10, rtol=0)
        or not np.isclose(np.linalg.det(result), 1.0, atol=1e-10, rtol=0)
    ):
        raise ValueError("invalid material frame rotation")
    return result


def _component_values(components, function):
    # Enumeration is local to this sample; retain multiplicity, ignore row order
    # only when reporting whether the sampled material set changed.
    return tuple(sorted(function(c.material) for c in components))


class RobotContactEventAdapter:
    """Immutable registration lookup and pure staging for same-input robot rows.

    No robot_scene import, solver call, publication, mutable history or geometry
    cache is hidden here. The owning scene publishes the staged result together
    with its accepted state, after every allocation and validation succeeds.
    """

    def __init__(self, robot_addr, tree, rigid_items, rigid_rows, pairs, key_map):
        world, body = (_index(v, "robot owner") for v in robot_addr)
        frames = {}
        for addr, item in rigid_items.items():
            key = item.shape.key
            if (key.world, key.body) != addr or key.world != world:
                raise ValueError("rigid shape and physical owner differ")
            frames[key] = _OwnerFrame(
                ContactOwner(*addr, "rigid", "rigid_com"),
                frozenset(item.shape.vertex_ids),
                _index(rigid_rows[addr], "rigid row"),
            )
        for col in tree.collisions:
            key = key_map[col.shape.key]
            if not isinstance(key, CollisionKey) or key.world != world or key in frames:
                raise ValueError("robot collision remap must be unique in its world")
            link = _index(col.link_index, "robot link", minimum=-1)
            name = tree.root_link if link == -1 else tree.link_names[link]
            kind, frame = (
                ("robot_root", "robot_root")
                if link == -1
                else ("robot_link", "robot_model_link")
            )
            frames[key] = _OwnerFrame(
                ContactOwner(world, body, kind, frame, link, name),
                frozenset(col.shape.vertex_ids),
            )
        laws = {}
        for pair, law in pairs.items():
            if tuple(sorted(pair)) != pair or len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("material pairs require two ordered physical owners")
            if law is not None and not isinstance(law, RobotContactMaterial):
                raise TypeError("robot event material is not converted from another law")
            laws[pair] = law
        self._frames = MappingProxyType(frames)
        self._laws = MappingProxyType(laws)
        self._root_position = _vector(tree.root_xyz, "root position")
        self._root_rotation = tuple(
            map(tuple, quaternion_matrix(np.asarray([tree.root_xyzw], dtype=float))[0])
        )

    def _pair_law(self, first, second):
        if (
            not isinstance(first, CollisionKey)
            or not isinstance(second, CollisionKey)
            or not first < second
            or first.world != second.world
        ):
            raise ValueError("loaded pairs require ordered same-world shape keys")
        if first not in self._frames or second not in self._frames:
            raise ValueError("load references an unregistered shape")
        a, b = self._frames[first].owner, self._frames[second].owner
        owners = tuple(sorted(((a.world, a.body), (b.world, b.body))))
        law = self._laws.get(owners)
        if law is None:
            raise ValueError("load references an excluded or unregistered owner pair")
        return law

    def _address(self, key, point, weights, geo, rigid, rotations):
        metadata = self._frames[key]
        if any(node not in metadata.vertex_ids or w < 0 for node, w in weights):
            raise ValueError("material weights reference an invalid registered vertex")
        if not np.isclose(sum(w for _, w in weights), 1.0, atol=1e-10, rtol=0):
            raise ValueError("material weights must sum to one")
        owner = metadata.owner
        if owner.kind == "rigid":
            position, rotation = rigid.position[metadata.row], rotations[metadata.row]
        elif owner.kind == "robot_root":
            position, rotation = self._root_position, self._root_rotation
        else:
            position = geo.position[owner.link_index]
            rotation = geo.rotation[owner.link_index]
        position = _vector(position, "material frame position")
        local = (np.asarray(point) - position) @ _rotation(rotation)
        return ContactMaterialAddress(
            owner.kind, weights, _vector(local, "local surface point")
        )

    def _material(self, load, geo, rigid, rotations):
        return tuple(
            ContactMaterialSite(
                self._address(
                    load.first, s.material.first_point, s.material.first_weights,
                    geo, rigid, rotations,
                ),
                self._address(
                    load.second, s.material.second_point, s.material.second_weights,
                    geo, rigid, rotations,
                ),
                s.fraction,
            )
            for s in load.sites
        )

    def stage(self, previous, loads, *, geo, rigid, sample_time_s, commit_time_s):
        """Return a new snapshot; failures cannot change previous or the adapter."""
        if not isinstance(previous, SceneContactEvents) or not isinstance(loads, SurfaceLoads):
            raise TypeError("expected SceneContactEvents and SurfaceLoads")
        sample = finite_real(sample_time_s, "load sample time")
        committed = finite_real(commit_time_s, "state commit time")
        if sample != previous.commit_time_s or committed <= sample:
            raise ValueError("event times must follow the accepted physical step")
        old = {(r.identity.first, r.identity.second): r for r in previous.active}
        generations = {(i.first, i.second): i for i in previous.generations}
        for pair in generations.keys() | old.keys():
            self._pair_law(*pair)
        rotations = quaternion_matrix(rigid.orientation)
        current = {}
        for raw in loads.contacts:
            if not isinstance(raw, PairLoad):
                raise TypeError("expected original PairLoad components")
            law = self._pair_law(raw.first, raw.second)
            owned = _owned_load(raw)
            if owned.force_n < 0 or any(s.force_n < 0 or s.fraction <= 0 for s in owned.sites):
                raise ValueError("normal loads and site fractions must be nonnegative/positive")
            component = ContactComponent(
                owned, self._material(owned, geo, rigid, rotations),
                "robot_contact_material", law,
            )
            current.setdefault((raw.first, raw.second), []).append(component)
        active, events = [], []
        for pair in sorted(old.keys() | current.keys()):
            components, prior = tuple(current.get(pair, ())), old.get(pair)
            if not any(c.load.force_n > 0 for c in components):
                if prior is not None:
                    events.append(SceneContactEvent(
                        "end", prior.identity, None, "zero_force" if components else "absent"
                    ))
                continue
            if prior is None:
                tombstone = generations.get(pair)
                identity = ContactIdentity(*pair, 1 if tombstone is None else tombstone.generation + 1)
                generations[pair] = identity
            else:
                identity = prior.identity
            record = SceneContactRecord(
                identity, self._frames[pair[0]].owner, self._frames[pair[1]].owner, components,
                prior is not None and _component_values(components, _values)
                != _component_values(prior.components, _values),
                prior is not None and _component_values(components, _support)
                != _component_values(prior.components, _support),
            )
            active.append(record)
            events.append(SceneContactEvent("begin" if prior is None else "persist", identity, record))
        return SceneContactEvents(
            previous.step_index + 1, sample, committed, tuple(active), tuple(events),
            tuple(generations[pair] for pair in sorted(generations)),
        )
