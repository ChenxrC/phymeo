"""Common-entry contact assembly for bounded resident participants.

HOST finite patch sampling is explicit. Device FK/J/M and the existing four-family
solvers remain the sole mechanical source. This module does not integrate state.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.robot_contact import RobotContactMaterial
from ._resident_contact_bridge import (
    ContactShapeFrame,
    ContactSiteCapacityExceeded,
    finish_contact_loads,
    sample_tree_contacts,
)
from ._resident_participant_layout import ResidentParticipantKey


def _owned(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("participant receipt array must be finite real data")
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _integer(value, name, low=0, high=2**31 - 1):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _map(value):
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class ResidentSceneCheckpoint:
    world: int
    tick: int
    registry_fingerprint: tuple
    participants: Mapping
    owner: object = field(repr=False)

    def __post_init__(self):
        world = _integer(self.world, "world")
        tick = _integer(self.tick, "tick")
        members = dict(self.participants)
        if not members or any(
            type(key) is not ResidentParticipantKey
            or key.world != world
            or point.world != world
            or point.tick != tick
            for key, point in members.items()
        ):
            raise ValueError("world checkpoint requires complete same-world participant state")
        object.__setattr__(self, "world", world)
        object.__setattr__(self, "tick", tick)
        object.__setattr__(
            self, "participants", _map({key: members[key] for key in sorted(members)})
        )


@dataclass(frozen=True, eq=False)
class ResidentParticipantReceipt:
    participant: ResidentParticipantKey
    dt_s: float
    entry_state: object
    accepted_state: object
    entry_geometry: object
    accepted_geometry: object
    entry_body_twist: np.ndarray
    accepted_body_twist: np.ndarray
    entry_mass: np.ndarray
    entry_bias: np.ndarray
    entry_linear: np.ndarray
    entry_angular: np.ndarray
    input_external: np.ndarray
    law_wrench_fp64: np.ndarray
    uploaded_wrench_fp32: np.ndarray
    contact_generalized: np.ndarray
    diagnostic: object

    def __post_init__(self):
        for name in (
            "entry_body_twist",
            "accepted_body_twist",
            "entry_mass",
            "entry_bias",
            "entry_linear",
            "entry_angular",
            "input_external",
            "law_wrench_fp64",
            "contact_generalized",
        ):
            object.__setattr__(self, name, _owned(getattr(self, name), np.float64))
        count = len(self.entry_linear)
        if self.entry_body_twist.shape != (count, 6) or self.accepted_body_twist.shape != (count, 6):
            raise ValueError("participant twists require every full physical body")
        object.__setattr__(
            self, "uploaded_wrench_fp32", _owned(self.uploaded_wrench_fp32, np.float32)
        )


@dataclass(frozen=True, eq=False)
class ResidentSceneReceipt:
    world: int
    entry_tick: int
    exit_tick: int
    dt_s: float
    state_epoch: int
    entry_bank: int
    participants: Mapping
    sites: tuple
    static_reactions: Mapping
    entry_frames: Mapping
    accepted_frames: Mapping
    geometry_fingerprint: tuple | None
    candidate_pairs: tuple
    owner: object = field(repr=False)

    def __post_init__(self):
        for name in ("participants", "static_reactions", "entry_frames", "accepted_frames"):
            object.__setattr__(self, name, _map(getattr(self, name)))
        object.__setattr__(self, "sites", tuple(self.sites))
        object.__setattr__(self, "candidate_pairs", tuple(self.candidate_pairs))


@dataclass(frozen=True, eq=False)
class _ScenePublication:
    accepted: Mapping
    status: Mapping
    receipts: Mapping
    states: Mapping
    rejected_sites: Mapping

    def __post_init__(self):
        for name in ("accepted", "status", "receipts", "states", "rejected_sites"):
            object.__setattr__(self, name, _map(getattr(self, name)))


class ResidentParticipantContactPlan:
    """One common-entry HOST patch plan; closing it never closes its scene."""

    def __init__(
        self,
        scene,
        model=None,
        bindings=(),
        materials_by_pair=None,
        *,
        candidate_capacity=4096,
        site_capacity=128,
        sampling="discrete_entry",
        intra_participant_pairs=False,
        narrow_phase="host_patch",
        device_patch_site_capacity=None,
    ):
        if sampling != "discrete_entry":
            raise ValueError("ROB1b requires discrete_entry contact sampling")
        if narrow_phase != "host_patch" or device_patch_site_capacity is not None:
            raise ValueError("ROB1b supports explicit host_patch only")
        if type(intra_participant_pairs) is not bool:
            raise TypeError("intra_participant_pairs must be bool")
        capacity = _integer(candidate_capacity, "candidate capacity", 0, 32640)
        self._scene = scene
        self._owner = scene._owner
        self._closed = False
        self._geometry = None if model is None else scene.layout.bind_geometry(model, bindings)
        self._bindings_by_key = (
            {}
            if self._geometry is None
            else {
                item.binding.key if hasattr(item, "participant") else item.key: item
                for item in self._geometry.bindings
            }
        )
        self._intra = intra_participant_pairs
        self._allowed = self._allowed_pairs()
        if len(self._allowed) > capacity:
            raise ValueError("complete candidate domain exceeds candidate capacity")
        self._candidate_capacity = capacity
        laws = {} if materials_by_pair is None else materials_by_pair
        if not isinstance(laws, Mapping) or set(laws) != set(self._allowed):
            raise ValueError("materials must exactly cover allowed canonical participant pairs")
        if any(not isinstance(value, RobotContactMaterial) for value in laws.values()):
            raise TypeError("participant contacts require original RobotContactMaterial")
        self._laws = _map(
            {
                pair: RobotContactMaterial(
                    law.stiffness, law.damping, law.friction, law.slip_speed_m_s
                )
                for pair, law in laws.items()
            }
        )
        worlds = scene.world_keys
        if isinstance(site_capacity, Mapping):
            if set(site_capacity) != set(worlds) or any(
                isinstance(key, (bool, np.bool_)) or not isinstance(key, Integral)
                for key in site_capacity
            ):
                raise ValueError("site capacities must cover every complete logical world")
            caps = {
                world: _integer(site_capacity[world], "site capacity", 0, 4096) for world in worlds
            }
        else:
            cap = _integer(site_capacity, "site capacity", 0, 4096)
            caps = dict.fromkeys(worlds, cap)
        self._site_caps = _map(caps)
        self._world_pairs = _map(
            {world: tuple(p for p in self._allowed if p[0].world == world) for world in worlds}
        )
        self._last_accepted = _map({})
        self._last_status = _map({})
        self._last_publication = None
        self._counts = dict(
            sampling_attempts=0, candidate_pairs_evaluated=0, site_capacity_failures=0
        )

    def _allowed_pairs(self):
        if self._geometry is None:
            return ()
        geometry = self._geometry
        excluded = set(geometry.model.excluded)
        allowed = []
        participants = {item.key: item for item in self._scene.layout.participants}
        for first, second in combinations(geometry.model.shapes, 2):
            pair = first.key, second.key
            if (
                pair[0].world != pair[1].world
                or pair[0].body == pair[1].body
                or pair in excluded
                or not (first.group & second.mask)
                or not (second.group & first.mask)
            ):
                continue
            left, right = geometry.owners[first.key], geometry.owners[second.key]
            if left is None and right is None:
                continue
            if left is not None and right is not None and left[0] == right[0]:
                if not self._intra:
                    continue
                compiled = participants[left[0]].compiled
                physical = getattr(compiled, "physical", compiled)
                parents = physical.parents
                if parents[left[1]] == right[1] or parents[right[1]] == left[1]:
                    raise ValueError("direct parent-child pair requires exclusion or mask")
            allowed.append(pair)
        return tuple(allowed)

    def _check(self):
        self._scene._check_open()
        if self._closed:
            raise RuntimeError("resident participant contact plan is closed")
        if self._geometry is not None:
            self._geometry.require_current(self._scene.layout)

    def _frames(self, packets):
        result = {world: {} for world in self._scene.world_keys}
        if self._geometry is None:
            return result
        for shape in self._geometry.model.shapes:
            owner = self._geometry.owners[shape.key]
            compiled = self._geometry.compiled_bindings[shape.key]
            if owner is None:
                points = compiled.start
                frame = ContactShapeFrame(
                    shape,
                    points,
                    None,
                    np.zeros(3),
                    np.eye(3),
                    np.mean(points, axis=0),
                    np.zeros(3),
                    np.zeros(3),
                )
            else:
                participant, local = owner
                actual = packets[participant]["geometry"]
                position = actual.link_position_m[local]
                rotation = actual.link_rotation[local].astype(np.float64)
                # Only transform registered body-local material with device FK;
                # no HOST kinematics or mechanical reconstruction is performed.
                points = (
                    position + self._bindings_by_key[shape.key].binding.local_points_m @ rotation.T
                )
                frame = ContactShapeFrame(
                    shape,
                    points,
                    self._scene.layout.body_slot(participant, local),
                    position,
                    rotation,
                    actual.com_position_m[local],
                    packets[participant]["body_twist"][local, :3],
                    packets[participant]["body_twist"][local, 3:],
                )
            result[shape.key.world][shape.key] = frame
        return {world: _map(frames) for world, frames in result.items()}

    @property
    def last_accepted(self):
        with self._scene._service.locked():
            self._check()
            return self._last_accepted

    @property
    def last_status(self):
        with self._scene._service.locked():
            self._check()
            return self._last_status

    def stats(self):
        return dict(
            self._counts,
            closed=self._closed,
            candidate_capacity=self._candidate_capacity,
            candidate_count=len(self._allowed),
            site_capacities=dict(self._site_caps),
            owned_bytes=0,
            narrow_phase="host_patch",
            candidate_source="host_full_allowed_pairs",
        )

    def close(self):
        if self._closed:
            return
        with self._scene._service.closing():
            if not self._closed:
                self._closed = True
                self._scene._plans.discard(self)

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()

    def sample(self, entry, h):
        self._check()
        if set(entry) != set(self._scene.participant_keys):
            raise ValueError("entry mechanics must contain every registered participant")
        self._counts["sampling_attempts"] += 1
        frames = self._frames(entry)
        loads, rejected_sites = {}, {}
        rejected = dict.fromkeys(self._scene.world_keys, False)
        wrenches = {
            key: np.zeros((self._scene.layout.entries[key].body_count, 6))
            for key in self._scene.participant_keys
        }
        for world in self._scene.world_keys:
            pairs = self._world_pairs[world]
            self._counts["candidate_pairs_evaluated"] += len(pairs)
            try:
                loads[world] = sample_tree_contacts(
                    world=world,
                    frames=frames[world],
                    materials_by_pair={pair: self._laws[pair] for pair in pairs},
                    candidates=pairs,
                    group_count=self._scene.layout.worlds[world].body_count,
                    site_capacity=self._site_caps[world],
                    tree_dynamic_pairs=True,
                )
            except ContactSiteCapacityExceeded as error:
                rejected[world] = True
                rejected_sites[world] = (error.count, error.limit)
                self._counts["site_capacity_failures"] += 1
                continue
            for key in self._scene.layout.worlds[world].participant_keys:
                row = self._scene.layout.body_slot(key, 0)
                count = self._scene.layout.entries[key].body_count
                wrenches[key] = _owned(loads[world].wrench_fp64[row : row + count])
        return dict(
            wrenches=_map(wrenches),
            rejected=_map(rejected),
            payload=dict(
                loads=_map(loads),
                frames=_map(frames),
                wrenches=_map(wrenches),
                rejected_sites=_map(rejected_sites),
                entry_epoch=self._scene.state_epoch,
                entry_bank=self._scene.stats()["current_buffer"],
            ),
        )

    def prepare_publication(
        self, entry, candidate, diagnostics, projected, accepted, status, payload, h
    ):
        self._check()
        if set(candidate) != set(entry) or set(accepted) != set(self._scene.world_keys):
            raise ValueError("candidate publication has incomplete participant/world membership")
        if payload["entry_epoch"] != self._scene.state_epoch:
            raise ValueError("contact entry epoch changed during candidate construction")
        frames = self._frames(candidate)
        states = self._scene._capsules({key: packet["state"] for key, packet in candidate.items()})
        receipts = dict(self._last_accepted)
        for world in self._scene.world_keys:
            keys = self._scene.layout.worlds[world].participant_keys
            if not accepted[world]:
                if any(candidate[key]["state"] != entry[key]["state"] for key in keys):
                    raise RuntimeError("world rejection failed complete participant retention")
                continue
            if world not in payload["loads"]:
                raise RuntimeError("site-rejected world cannot become accepted")
            complete = finish_contact_loads(payload["loads"][world], frames[world])
            members = {}
            for key in keys:
                before, after = entry[key], candidate[key]
                if after["state"].tick != before["state"].tick + 1:
                    raise RuntimeError("participant publication must advance exactly once")
                if key not in diagnostics or key not in projected:
                    raise ValueError("accepted participant lacks its actual device diagnostics")
                wrench = payload["wrenches"][key]
                members[key] = ResidentParticipantReceipt(
                    key,
                    h,
                    before["state"],
                    after["state"],
                    before["geometry"],
                    after["geometry"],
                    before["body_twist"],
                    after["body_twist"],
                    before["mass"],
                    before["bias"],
                    before["linear"],
                    before["angular"],
                    before["external"],
                    wrench,
                    np.asarray(wrench, dtype=np.float32),
                    projected[key],
                    diagnostics[key],
                )
            receipts[world] = ResidentSceneReceipt(
                world,
                entry[keys[0]]["state"].tick,
                candidate[keys[0]]["state"].tick,
                h,
                payload["entry_epoch"] + 1,
                payload["entry_bank"],
                members,
                complete.sites,
                complete.static_reactions,
                payload["frames"][world],
                frames[world],
                None if self._geometry is None else self._geometry.fingerprint,
                self._world_pairs[world],
                self._owner,
            )
        return _ScenePublication(accepted, status, receipts, states, payload["rejected_sites"])

    def _publish(self, publication):
        self._last_publication = publication
        self._last_accepted = publication.receipts
        self._last_status = publication.status

    def _prepare_invalidation(self, worlds):
        selected = set(worlds)
        return (
            _map({w: r for w, r in self._last_accepted.items() if w not in selected}),
            _map({w: s for w, s in self._last_status.items() if w not in selected}),
        )

    def _invalidate(self, prepared):
        self._last_accepted, self._last_status = prepared
        self._last_publication = None

    def require_current(self, receipt):
        with self._scene._service.locked():
            self._check()
            if (
                type(receipt) is not ResidentSceneReceipt
                or receipt.owner is not self._owner
                or receipt.world not in self._scene.layout.worlds
                or receipt.state_epoch != self._scene.state_epoch
                or self._last_accepted.get(receipt.world) is not receipt
            ):
                raise ValueError("receipt belongs to another owner or stale scene epoch")
            return receipt
