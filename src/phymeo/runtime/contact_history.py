"""Immutable lifecycle observations for loads applied by an accepted scene step.

This is a record of positive normal loads, not a contact solver or geometric
overlap tracker. A zero-force PairLoad can still occupy the material layer.
The load and both surface witnesses belong to sample_time_s; commit_time_s is
the time of the accepted state obtained by applying that load for the step.
"""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ..constraints.collision_shapes import CollisionKey
from ..core.validation import finite_real
from ..dynamics.rotation import quaternion_matrix
from ..geometry.collision_witness import ConvexContactPoint
from ..solvers.collision_loads import (
    CollisionLoadStep,
    LoadSite,
    PairLoad,
    ParticleAttachment,
    RigidAttachment,
)


@dataclass(frozen=True, order=True)
class ContactIdentity:
    """World-local shape pair and its positive-load episode; no coordinate hash."""

    first: CollisionKey
    second: CollisionKey
    generation: int


@dataclass(frozen=True)
class ContactMaterialAddress:
    """Original vertex weights and, for a rigid owner, its body-local surface point."""

    kind: str
    weights: tuple[tuple[int, float], ...]
    local_point_m: tuple[float, float, float] | None


@dataclass(frozen=True)
class ContactMaterialSite:
    """Two surface addresses, distinct from the force's possibly shared application point."""

    first: ContactMaterialAddress
    second: ContactMaterialAddress
    fraction: float


@dataclass(frozen=True)
class ContactRecord:
    """Actual current load and sampled material witnesses for one loaded shape pair.

    material_changed compares the exact recorded material values, including
    possible floating-point last-bit changes. support_changed compares discrete
    nonzero vertex support sets. Neither flag creates a new contact identity.
    No stable per-site or solver feature ID is inferred from these observations.
    """

    identity: ContactIdentity
    load: PairLoad
    material: tuple[ContactMaterialSite, ...]
    material_changed: bool = False
    support_changed: bool = False


@dataclass(frozen=True)
class ContactEvent:
    """Begin/persist carries the current record; end carries no stale force."""

    kind: str
    identity: ContactIdentity
    record: ContactRecord | None
    reason: str | None = None


@dataclass(frozen=True)
class ContactHistory:
    """Latest committed load observation and bounded per-pair generation tombstones.

    Only active records and events from this step are retained. Generations keep
    one small identity per previously loaded registered pair, not a frame log.
    Sample/geometry calls never advance this snapshot. An empty initial snapshot
    has both times and its step_index equal to zero.
    """

    step_index: int = 0
    sample_time_s: float = 0.0
    commit_time_s: float = 0.0
    active: tuple[ContactRecord, ...] = ()
    events: tuple[ContactEvent, ...] = ()
    _generations: tuple[ContactIdentity, ...] = field(default=(), repr=False)

    @property
    def generations(self):
        """Last positive-load generation for each pair, including inactive ones."""
        return self._generations


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("invalid " + name)
    return tuple(map(float, result))


def _weights(values):
    out = []
    seen = set()
    for node, weight in values:
        if (
            isinstance(node, (bool, np.bool_))
            or not isinstance(node, Integral)
            or node < 0
            or node in seen
        ):
            raise ValueError("invalid material vertex ID")
        seen.add(node)
        out.append((int(node), finite_real(weight, "material weight")))
    if not out:
        raise ValueError("missing material weights")
    return tuple(sorted(out))


def _owned_load(load):
    """Freeze even array-like data supplied through the reusable staging entry."""
    sites = []
    for site in load.sites:
        p = site.material
        material = ConvexContactPoint(
            _vector(p.first_point, "first material point"),
            _vector(p.second_point, "second material point"),
            _weights(p.first_weights),
            _weights(p.second_weights),
        )
        sites.append(
            LoadSite(
                material,
                finite_real(site.fraction, "site fraction"),
                finite_real(site.rate_m_per_s, "site separation rate"),
                finite_real(site.force_n, "site force"),
                _vector(site.application_point_first_m, "first application point"),
                _vector(site.application_point_second_m, "second application point"),
                _vector(site.force_on_second_n, "site traction"),
                finite_real(site.power_w, "site power"),
            )
        )
    if not sites:
        raise ValueError("positive load requires material sites")
    return PairLoad(
        load.first,
        load.second,
        _vector(load.normal, "contact normal"),
        finite_real(load.gap_m, "gap"),
        finite_real(load.force_n, "normal load"),
        finite_real(load.potential_j, "layer energy"),
        finite_real(load.power_w, "layer power"),
        tuple(sites),
    )


def _address(key, point, weights, state, bindings, rotations):
    binding = bindings[key]
    if isinstance(binding, RigidAttachment):
        row = binding.row
        # Use the sampled surface point, not the midpoint chosen for friction.
        local = (np.asarray(point) - state.rigid.position[row]) @ rotations[row]
        return ContactMaterialAddress("rigid", weights, _vector(local, "local surface point"))
    if isinstance(binding, ParticleAttachment):
        return ContactMaterialAddress("particle", weights, None)
    raise TypeError("expected rigid or particle material attachment")


def _material(load, entry, bindings, rotations):
    return tuple(
        ContactMaterialSite(
            _address(
                load.first,
                s.material.first_point,
                s.material.first_weights,
                entry,
                bindings,
                rotations,
            ),
            _address(
                load.second,
                s.material.second_point,
                s.material.second_weights,
                entry,
                bindings,
                rotations,
            ),
            s.fraction,
        )
        for s in load.sites
    )


def _values(material):
    # Ignore the enumerator's site order; retain every actual sampled coordinate.
    def side(address):
        return address.kind, address.weights, address.local_point_m or ()

    return tuple(sorted((side(s.first), side(s.second), s.fraction) for s in material))


def _support(material):
    def side(address):
        return address.kind, tuple(node for node, weight in address.weights if weight != 0.0)

    return tuple(sorted((side(s.first), side(s.second)) for s in material))


def advance_contact_history(
    previous,
    frame,
    *,
    sample_time_s,
    commit_time_s,
    bindings,
):
    """Stage a new immutable snapshot without mutating previous or publishing.

    The scene calls this only after obtaining its owned physical result and
    before its enclosing workspace transaction commits. Any error leaves the
    caller's prior state, time, and history available for the same retry.
    """
    if not isinstance(previous, ContactHistory) or not isinstance(frame, CollisionLoadStep):
        raise TypeError("expected ContactHistory and an accepted CollisionLoadStep")
    sample = finite_real(sample_time_s, "load sample time")
    committed = finite_real(commit_time_s, "state commit time")
    if sample != previous.commit_time_s or committed <= sample:
        raise ValueError("contact history times must follow the accepted physical step")
    for report in (frame.particle_report, frame.rigid_report):
        if report is not None and not report.accepted:
            raise ValueError("contact history requires an accepted physical step")
    if frame.path.status != "ok":
        raise ValueError("contact history requires an accepted collision path")

    old = {(r.identity.first, r.identity.second): r for r in previous.active}
    generations = {(i.first, i.second): i for i in previous.generations}
    current = {}
    for load in frame.loads.contacts:
        if (
            not isinstance(load, PairLoad)
            or not isinstance(load.first, CollisionKey)
            or not isinstance(load.second, CollisionKey)
            or not load.first < load.second
            or load.first.world != load.second.world
        ):
            raise ValueError("loaded pairs require ordered same-world shape keys")
        pair = load.first, load.second
        if pair in current:
            raise ValueError("duplicate loaded shape pair")
        if any(key not in bindings for key in pair):
            raise ValueError("load references an unbound shape")
        force = finite_real(load.force_n, "normal load")
        if force < 0:
            raise ValueError("normal load must be nonnegative")
        current[pair] = load

    rotations = quaternion_matrix(frame.entry.rigid.orientation)
    active = []
    events = []
    for pair in sorted(old.keys() | current.keys()):
        load = current.get(pair)
        prior = old.get(pair)
        if load is None or load.force_n == 0.0:
            if prior is not None:
                events.append(
                    ContactEvent(
                        "end",
                        prior.identity,
                        None,
                        "absent" if load is None else "zero_force",
                    )
                )
            continue
        if prior is None:
            tombstone = generations.get(pair)
            identity = ContactIdentity(*pair, 1 if tombstone is None else tombstone.generation + 1)
            generations[pair] = identity
        else:
            identity = prior.identity
        owned = _owned_load(load)
        material = _material(owned, frame.entry, bindings, rotations)
        record = ContactRecord(
            identity,
            owned,
            material,
            prior is not None and _values(material) != _values(prior.material),
            prior is not None and _support(material) != _support(prior.material),
        )
        active.append(record)
        events.append(ContactEvent("begin" if prior is None else "persist", identity, record))
    return ContactHistory(
        previous.step_index + 1,
        sample,
        committed,
        tuple(active),
        tuple(events),
        tuple(generations[pair] for pair in sorted(generations)),
    )
