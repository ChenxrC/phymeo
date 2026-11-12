"""Stable static body addresses for the bounded M-B1 response domain.

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
    """Immutable canonical world/participant address table, without a Runtime."""

    worlds: tuple
    participant_keys: tuple
    world_participants: Mapping
    participant_slots: Mapping
    body_offsets: np.ndarray
    multibody: bool

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
    """Compile 1..8 independent ordinary floating bodies in stable key order.

    Each participant owns one physical body. Different worlds may reuse a body
    ID, but two participants in one world may not. No accepted state, device
    address, source recompilation, or physics execution is involved here.
    """
    if not isinstance(participants, (tuple, list)):
        raise TypeError("response participants must be a finite tuple or list")
    if not 1 <= len(participants) <= 8:
        raise ValueError("M-B1 requires 1..8 participants in the complete batch")
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
        if item.family != "floating" or item.body_count != 1:
            raise ValueError("M-B1 requires ordinary floating single-body participants")
        if type(item.body_ids) is not tuple or len(item.body_ids) != 1:
            raise ValueError("single-body participant requires exactly one body ID")
        body = _identity(item.body_ids[0], "participant body ID")
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
    world_participants = {world: tuple(grouped[world]) for world in worlds}
    slots = {key: slot for slot, key in enumerate(keys)}
    offsets = [0]
    for world in worlds:
        offsets.append(offsets[-1] + len(world_participants[world]))
    # Immutable bytes owner prevents setflags(write=True), unlike a readonly
    # view of a still-mutable caller array. Inputs themselves are not retained.
    body_offsets = np.frombuffer(np.asarray(offsets, dtype=np.int32).tobytes(), dtype=np.int32)
    result = object.__new__(DeformableResponseLayout)
    for name, value in (
        ("worlds", worlds),
        ("participant_keys", keys),
        ("world_participants", MappingProxyType(world_participants)),
        ("participant_slots", MappingProxyType(slots)),
        ("body_offsets", body_offsets),
        ("multibody", any(len(members) > 1 for members in world_participants.values())),
    ):
        object.__setattr__(result, name, value)
    return result
