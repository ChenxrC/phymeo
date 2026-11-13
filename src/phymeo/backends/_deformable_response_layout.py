"""Stable static body addresses for bounded M-B1 and M-B2 response domains.

This compiler allocates only owned HOST metadata. Source/model/topology currency
remains the responsibility of the original resident scene and query registration.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ._resident_participant_layout import ResidentParticipant, ResidentParticipantKey


@dataclass(frozen=True, eq=False, init=False)
class DeformableResponseLayout:
    """Immutable canonical world/participant/body table, without a Runtime."""

    worlds: tuple
    participant_keys: tuple
    world_participants: Mapping
    participant_slots: Mapping
    body_addresses: tuple
    body_slots: Mapping
    participant_body_counts: Mapping
    body_offsets: np.ndarray
    multibody: bool
    articulated: bool

    def __init__(self):
        raise TypeError("use compile_deformable_response_layout(participants)")


def _identity(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 0 <= value <= 2**31 - 1
    ):
        raise ValueError(name + " must be a nonnegative i32 identity")
    return int(value)


def compile_deformable_response_layout(participants):
    """Compile 1..8 ordinary floating participants in canonical body order.

    A world may hold several independent single bodies or one articulated tree,
    never a tree together with another participant. Every physical body gets a
    slot, including bodies without colliders. Different worlds may reuse body
    IDs; within a world each ID has exactly one owner. No device allocation or
    current-state mechanics occurs here.
    """
    if not isinstance(participants, (tuple, list)):
        raise TypeError("response participants must be a finite tuple or list")
    if not 1 <= len(participants) <= 8:
        raise ValueError("M-B1/M-B2 requires 1..8 participants in the complete batch")
    if any(type(item) is not ResidentParticipant for item in participants):
        raise TypeError("response layout requires ResidentParticipant declarations")
    by_key = {}
    body_owners = set()
    for item in participants:
        key = item.key
        if type(key) is not ResidentParticipantKey:
            raise TypeError("response participant key must be ResidentParticipantKey")
        world = _identity(key.world, "participant world")
        _identity(key.participant, "participant ID")
        if key in by_key:
            raise ValueError("duplicate response participant identity")
        if item.family != "floating":
            raise ValueError("M-B1/M-B2 requires ordinary floating participants")
        count = _identity(item.body_count, "participant body count")
        if not 1 <= count <= 33:
            raise ValueError("response participant requires 1..33 physical bodies")
        if type(item.body_ids) is not tuple or len(item.body_ids) != count:
            raise ValueError("participant body IDs must cover every physical body")
        spec = item.compiled
        if spec.n + 1 != count or spec.b != count:
            raise ValueError("participant body count differs from its floating model")
        for raw_body in item.body_ids:
            body = _identity(raw_body, "participant body ID")
            identity = (world, body)
            if identity in body_owners:
                raise ValueError("body ID aliases participants in the same response world")
            body_owners.add(identity)
        by_key[key] = item

    keys = tuple(sorted(by_key))
    grouped = {}
    for key in keys:
        grouped.setdefault(key.world, []).append(key)
    worlds = tuple(grouped)
    if not 1 <= len(worlds) <= 8:
        raise ValueError("response layout requires 1..8 worlds")
    world_participants = {world: tuple(grouped[world]) for world in worlds}
    counts = {key: int(by_key[key].body_count) for key in keys}
    slots, body_slots, addresses = {}, {}, []
    offsets = [0]
    for world in worlds:
        members = world_participants[world]
        if len(members) > 1 and any(counts[key] > 1 for key in members):
            raise ValueError("an articulated response world requires exactly one participant")
        if sum(counts[key] for key in members) > 33:
            raise ValueError("a response world supports at most 33 physical bodies")
        for key in members:
            slots[key] = len(addresses)
            for group, body in enumerate(by_key[key].body_ids):
                body_slots[(key, group)] = len(addresses)
                addresses.append((key, group, int(body)))
        offsets.append(len(addresses))
    # Immutable bytes owner prevents setflags(write=True), unlike a readonly
    # view of a still-mutable caller array. Inputs themselves are not retained.
    body_offsets = np.frombuffer(np.asarray(offsets, dtype=np.int32).tobytes(), dtype=np.int32)
    articulated = any(count > 1 for count in counts.values())
    result = object.__new__(DeformableResponseLayout)
    for name, value in (
        ("worlds", worlds),
        ("participant_keys", keys),
        ("world_participants", MappingProxyType(world_participants)),
        ("participant_slots", MappingProxyType(slots)),
        ("body_addresses", tuple(addresses)),
        ("body_slots", MappingProxyType(body_slots)),
        ("participant_body_counts", MappingProxyType(counts)),
        ("body_offsets", body_offsets),
        ("multibody", articulated or any(len(m) > 1 for m in world_participants.values())),
        ("articulated", articulated),
    ):
        object.__setattr__(result, name, value)
    return result
