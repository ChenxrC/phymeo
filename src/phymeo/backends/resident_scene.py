"""Resident same-world participant registration and bounded common response.

Four tree families use their existing resident mechanics and explicit HOST patch
sampling. All mechanical blocks share one entry and world publication; this is
not an implicit contact-force solve across independent mechanical trees.
The shared transaction owns every device bank, solve and atomic world publish.
"""

from collections.abc import Mapping
from types import MappingProxyType

from ._resident_participant_contacts import (
    ResidentParticipantContactPlan,
    ResidentParticipantReceipt,
    ResidentSceneCheckpoint,
    ResidentSceneReceipt,
    _integer,
)
from ._resident_participant_layout import (
    MAX_PARTICIPANTS_PER_WORLD,
    ResidentParticipant,
    ResidentParticipantBinding,
    ResidentParticipantGeometry,
    ResidentParticipantKey,
    ResidentParticipantLayout,
    ResidentParticipantLayoutEntry,
    ResidentParticipantWorldLayout,
)


class ResidentSceneBatch:
    """One owner for all independent mechanical blocks of each logical world."""

    def __init__(self, participants, *, mode="cuda", cache_root=None, static_body_ids=()):
        if mode not in ("interpret", "cuda"):
            raise ValueError("resident scene mode must be interpret or cuda")
        self._layout = ResidentParticipantLayout(
            tuple(participants), static_body_ids=tuple(static_body_ids)
        )
        self._owner = object()
        self._closed = False
        self._plans = set()
        self._last_publication = None
        from ._resident_batch_transaction import ResidentBatchTransaction

        self._service = ResidentBatchTransaction(self.layout, mode, cache_root=cache_root)
        try:
            self._free_plan = ResidentParticipantContactPlan(self)
        except BaseException:
            self._service.close()
            self._closed = True
            raise

    @property
    def layout(self):
        return self._layout

    @property
    def participant_keys(self):
        return self.layout.participant_keys

    @property
    def world_keys(self):
        return self.layout.world_keys

    @property
    def state_epoch(self):
        self._check_open()
        return self._service.state_epoch

    def _check_open(self):
        if self._closed:
            raise RuntimeError("resident scene is closed")
        self.layout.require_current()

    def _worlds(self, worlds):
        if worlds is None:
            return self.world_keys
        selected = tuple(_integer(world, "world") for world in worlds)
        if len(set(selected)) != len(selected) or any(
            world not in self.layout.worlds for world in selected
        ):
            raise ValueError("unknown or duplicate logical world")
        return tuple(world for world in self.world_keys if world in selected)

    def _capsules(self, states):
        result = {}
        for world in self.world_keys:
            keys = self.layout.worlds[world].participant_keys
            selected = {key: states[key] for key in keys if key in states}
            if not selected:
                continue
            if set(selected) != set(keys):
                raise ValueError("world snapshot must include every participant")
            ticks = {point.tick for point in selected.values()}
            if len(ticks) != 1:
                raise RuntimeError("participant ticks differ inside a logical world")
            result[world] = ResidentSceneCheckpoint(
                world, ticks.pop(), self.layout.fingerprint, selected, self._owner
            )
        if set(states) != {key for packet in result.values() for key in packet.participants}:
            raise ValueError("snapshot contains an unknown participant")
        return MappingProxyType(result)

    def snapshot(self, worlds=None):
        with self._service.locked():
            self._check_open()
            return self._capsules(self._service.snapshot(self._worlds(worlds)))

    def set_controls(self, controls):
        with self._service.locked():
            self._check_open()
            if not isinstance(controls, Mapping) or any(
                type(key) is not ResidentParticipantKey or key not in self.layout.entries
                for key in controls
            ):
                raise ValueError("controls must map registered participant identities")
            self._service.set_controls(controls)

    def prepare_contacts(
        self,
        model,
        bindings,
        materials_by_pair,
        *,
        candidate_capacity=4096,
        site_capacity=128,
        sampling="discrete_entry",
        intra_participant_pairs=False,
        narrow_phase="host_patch",
        device_patch_site_capacity=None,
    ):
        with self._service.locked():
            self._check_open()
            plan = ResidentParticipantContactPlan(
                self,
                model,
                tuple(bindings),
                materials_by_pair,
                candidate_capacity=candidate_capacity,
                site_capacity=site_capacity,
                sampling=sampling,
                intra_participant_pairs=intra_participant_pairs,
                narrow_phase=narrow_phase,
                device_patch_site_capacity=device_patch_site_capacity,
            )
            self._plans.add(plan)
            return plan

    def _step(self, h, plan):
        with self._service.locked():
            self._check_open()
            if type(plan) is not ResidentParticipantContactPlan:
                raise TypeError("scene step requires a ResidentParticipantContactPlan")
            if plan._scene is not self or plan._owner is not self._owner:
                raise ValueError("contact plan belongs to another scene owner")
            plan._check()
            publication = self._service.step(h, plan)
            # All objects were prepared before the shared service's bank swap.
            # These assignments run under the same owner lock and cannot fail.
            plan._publish(publication)
            self._last_publication = publication
            return publication.accepted

    def step(self, h):
        return self._step(h, self._free_plan)

    def step_contacts(self, h, plan):
        return self._step(h, plan)

    def restore(self, checkpoints, *, expected_epoch):
        with self._service.locked():
            self._check_open()
            epoch = _integer(expected_epoch, "expected epoch")
            if epoch != self.state_epoch:
                raise ValueError("stale scene restore epoch")
            if not isinstance(checkpoints, Mapping):
                raise TypeError("scene restore requires complete world checkpoints")
            worlds = self._worlds(checkpoints)
            staged = {}
            for world in worlds:
                packet = checkpoints[world]
                if (
                    type(packet) is not ResidentSceneCheckpoint
                    or packet.owner is not self._owner
                    or packet.registry_fingerprint != self.layout.fingerprint
                    or packet.world != world
                ):
                    raise ValueError("checkpoint belongs to another owner or registry source")
                keys = self.layout.worlds[world].participant_keys
                if set(packet.participants) != set(keys):
                    raise ValueError("world restore requires complete participant membership")
                if any(
                    point.world != world or point.tick != packet.tick
                    for point in packet.participants.values()
                ):
                    raise ValueError("checkpoint participant world or tick differs")
                staged.update(packet.participants)
            invalidated = [
                (plan, plan._prepare_invalidation(worlds))
                for plan in (*self._plans, self._free_plan)
            ]
            self._service.restore(staged, expected_epoch=epoch)
            for plan, prepared in invalidated:
                plan._invalidate(prepared)
            if worlds:
                self._last_publication = None

    def reset(self, worlds=None):
        with self._service.locked():
            self._check_open()
            selected = self._worlds(worlds)
            invalidated = [
                (plan, plan._prepare_invalidation(selected))
                for plan in (*self._plans, self._free_plan)
            ]
            self._service.reset(selected)
            for plan, prepared in invalidated:
                plan._invalidate(prepared)
            if selected:
                self._last_publication = None

    def stats(self):
        return dict(
            self._service.stats(), scene_closed=self._closed, contact_plan_count=len(self._plans)
        )

    def close(self):
        if self._closed:
            return
        with self._service.closing():
            for plan in tuple(self._plans):
                plan.close()
            self._free_plan.close()
            self._service.close()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()


__all__ = [
    "MAX_PARTICIPANTS_PER_WORLD",
    "ResidentParticipant",
    "ResidentParticipantBinding",
    "ResidentParticipantGeometry",
    "ResidentParticipantKey",
    "ResidentParticipantLayout",
    "ResidentParticipantLayoutEntry",
    "ResidentParticipantWorldLayout",
    "ResidentParticipantContactPlan",
    "ResidentParticipantReceipt",
    "ResidentSceneBatch",
    "ResidentSceneCheckpoint",
    "ResidentSceneReceipt",
]
