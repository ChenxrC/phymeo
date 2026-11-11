"""Shared multi-participant transaction over the existing floating TYX storage.

Mechanical row IDs are private storage addresses. Public identities and failure
domains remain logical worlds. This service never duplicates the integrator.
The first domain is ordinary floating trees, at most 33 physical bodies/world.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType

import numpy as np

from ._resident_floating_contact import (
    floating_contact_diagnostics,
    read_floating_contact_packet,
)
from ._resident_participant_layout import ResidentParticipantLayout
from ._world_step_layout import as_fp32, world_key
from .tyx_world_batch import TyxWorldBatch


def _owned(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


class ResidentBatchTransaction:
    """Internal service; scene/contact adapters cannot mutate physical banks.

    ``adapter.sample`` consumes a single common-entry packet. Its bounded HOST
    law outputs become device COM wrenches, projected by actual device Jv/Jw.
    Publication is fully constructed before the original batch swaps its bank.
    """

    def __init__(self, layout, mode="cuda", *, cache_root=None):
        if type(layout) is not ResidentParticipantLayout:
            raise TypeError("transaction requires a ResidentParticipantLayout")
        layout.require_current()
        if any(item.family != "floating" for item in layout.participants):
            raise ValueError("ROB1b currently requires ordinary floating participants")
        if any(world.body_count > 33 for world in layout.worlds.values()):
            raise ValueError("ROB1b supports at most 33 physical bodies per logical world")
        self.layout = layout
        self._mechanical = {key: index for index, key in enumerate(layout.participant_keys)}
        self._keys = {index: key for key, index in self._mechanical.items()}
        self._batch = TyxWorldBatch(
            tuple((self._mechanical[item.key], item.system) for item in layout.participants),
            mode,
            cache_root=cache_root,
        )
        self._closed = False
        self._in_step = False
        self.last_publication = None
        self._logical_accepted_steps = 0
        self._logical_rejected_steps = 0
        self._stages = dict.fromkeys(
            ("entry_mechanics_stages", "candidate_solve_stages", "candidate_finalize_stages"),
            0,
        )
        self._transfers = dict(downloaded_bytes=0, uploaded_bytes=0)
        # Existing readback helper supplies actual device state and geometry.
        self._contact_worlds = tuple(self._keys)
        self._contact_shapes = {world: () for world in self._contact_worlds}
        self._contact_bindings = {}
        storage = self._batch._floating_trees
        self._storage = storage
        self._rows = {key: storage.rows[index] for key, index in self._mechanical.items()}
        try:
            from ._tyx_participant_transaction import make_participant_transaction_kernels
            from ._tyx_resident_floating_contact import make_floating_contact_kernels

            self._project = make_floating_contact_kernels()["project"]
            self._reduce = make_participant_transaction_kernels().reduce
            offsets, rows = [0], []
            for world in layout.worlds.values():
                rows.extend(self._rows[key] for key in world.participant_keys)
                offsets.append(len(rows))
            with self._batch._device():
                self._world_offsets = self._batch._buffer(
                    (len(offsets),), np.int32, offsets, "topology"
                )
                self._world_rows = self._batch._buffer((len(rows),), np.int32, rows, "topology")
                self._wrench = self._batch._buffer((storage.count, storage.bmax, 6), np.float32)
                self._projected = self._batch._buffer((storage.count, storage.dmax), np.float32)
        except Exception:
            self._batch.close()
            raise

    @contextmanager
    def locked(self):
        with self._batch._lock:
            self._check_open()
            yield self

    @contextmanager
    def closing(self):
        """Release resources even when source validation or the driver failed."""
        with self._batch._lock:
            if self._in_step:
                raise RuntimeError("cannot close during scene publication")
            yield self

    def _check_open(self):
        if self._closed:
            raise RuntimeError("resident participant transaction is closed")
        self._batch._check_open()
        self.layout.require_current()

    def _writer(self):
        self._check_open()
        if self._in_step:
            raise RuntimeError("cannot mutate a resident scene during its transaction")

    @property
    def state_epoch(self):
        with self.locked():
            return self._batch._state_epoch

    def _worlds(self, keys):
        values = self.layout.world_keys if keys is None else tuple(world_key(key) for key in keys)
        if len(set(values)) != len(values) or any(
            isinstance(value, (bool, np.bool_)) or value not in self.layout.worlds
            for value in values
        ):
            raise ValueError("unknown or duplicate logical world")
        return tuple(values)

    def _selected(self, worlds):
        return tuple(key for world in worlds for key in self.layout.worlds[world].participant_keys)

    def snapshot(self, world_keys=None):
        with self.locked():
            keys = self._selected(self._worlds(world_keys))
            states = self._batch.snapshot([self._mechanical[key] for key in keys])
            return {key: replace(states[self._mechanical[key]], world=key.world) for key in keys}

    def geometry(self, world_keys=None):
        with self.locked():
            keys = self._selected(self._worlds(world_keys))
            packets = self._read_packets(self._batch._active)
            return {key: packets[key]["geometry"] for key in keys}

    def set_controls(self, controls):
        with self.locked():
            self._writer()
            if not isinstance(controls, Mapping) or any(
                key not in self._mechanical for key in controls
            ):
                raise ValueError("controls must map registered participant keys")
            self._batch.set_controls(
                {self._mechanical[key]: value for key, value in controls.items()}
            )

    def restore(self, checkpoints, *, expected_epoch=None):
        with self.locked():
            self._writer()
            if expected_epoch is not None and (
                type(expected_epoch) is not int or expected_epoch < 0
            ):
                raise ValueError("restore epoch must be a nonnegative integer")
            if expected_epoch is not None and expected_epoch != self._batch._state_epoch:
                raise ValueError("stale scene restore epoch")
            if not isinstance(checkpoints, Mapping) or any(
                key not in self._mechanical for key in checkpoints
            ):
                raise ValueError("restore must map registered participant keys")
            worlds = {key.world for key in checkpoints}
            expected = set(self._selected(worlds))
            if set(checkpoints) != expected:
                raise ValueError("restore must cover every participant of each selected world")
            for world in worlds:
                values = [checkpoints[key] for key in self.layout.worlds[world].participant_keys]
                if (
                    any(value.world != world for value in values)
                    or len({value.tick for value in values}) != 1
                ):
                    raise ValueError("restore logical world or common tick differs")
            self._batch.restore(
                {
                    self._mechanical[key]: replace(value, world=self._mechanical[key])
                    for key, value in checkpoints.items()
                }
            )
            if checkpoints:
                self.last_publication = None

    def reset(self, world_keys=None):
        with self.locked():
            self._writer()
            keys = self._selected(self._worlds(world_keys))
            self._batch.reset([self._mechanical[key] for key in keys])
            if keys:
                self.last_publication = None

    def stats(self):
        with self._batch._lock:
            original = self._batch.stats()
            return {
                **original,
                "mechanical_row_count": original["world_count"],
                "accepted_mechanical_row_steps": original["accepted_world_steps"],
                "rejected_mechanical_row_steps": original["rejected_world_steps"],
                "accepted_world_steps": self._logical_accepted_steps,
                "rejected_world_steps": self._logical_rejected_steps,
                "world_count": len(self.layout.world_keys),
                "participant_count": len(self.layout.participant_keys),
                **self._stages,
                "participant_packet_downloaded_bytes": self._transfers["downloaded_bytes"],
                "participant_wrench_uploaded_bytes": self._transfers["uploaded_bytes"],
            }

    def close(self):
        with self._batch._lock:
            if self._in_step:
                raise RuntimeError("cannot close during scene publication")
            if not self._closed:
                self._batch.close()
                self._closed = True

    def step(self, h, adapter):
        with self.locked():
            self._writer()
            self._in_step = True
            self._adapter = adapter
            try:
                self._batch._step(h, contacts=self)
                return self.last_publication
            finally:
                self._adapter = None
                self._in_step = False

    def _read_contact(self, buffer):
        value = self._batch._read(buffer, "snapshot")
        self._transfers["downloaded_bytes"] += value.nbytes
        return value

    def _read_packets(self, bank):
        raw = read_floating_contact_packet(self, self._storage, self._contact_worlds, bank)
        result = {}
        for index, packet in raw.items():
            key = self._keys[index]
            result[key] = {
                "state": replace(packet["state"], world=key.world),
                "geometry": packet["geometry"],
                "energy": packet["energy"],
            }
        for world in self.layout.worlds.values():
            if len({result[key]["state"].tick for key in world.participant_keys}) != 1:
                raise RuntimeError("logical world participants have inconsistent ticks")
        return result

    def _handles(self, storage):
        return storage is self._storage

    def _begin_attempt(self, dt, src, dst):
        storage = self._storage
        held = tuple(
            getattr(storage, name).copy() for name in ("_held_drive", "_held_effort", "_held_loads")
        )
        storage.stage_mechanics(src)
        self._stages["entry_mechanics_stages"] += 1
        entry = self._read_packets(src)
        arrays = {
            name: self._read_contact(getattr(storage, name))
            for name in ("linear", "angular", "mass", "vector")
        }
        for key, packet in entry.items():
            row = self._rows[key]
            spec = storage.specs[self._mechanical[key]]
            packet.update(
                linear=_owned(arrays["linear"][row, : spec.b, :, : spec.d]),
                angular=_owned(arrays["angular"][row, : spec.b, :, : spec.d]),
                mass=_owned(arrays["mass"][row, : spec.d, : spec.d]),
                bias=_owned(arrays["vector"][row, : spec.d, 0]),
                external=_owned(arrays["vector"][row, : spec.d, 1]),
            )
        entry = MappingProxyType({key: MappingProxyType(value) for key, value in entry.items()})
        sample = self._adapter.sample(entry, dt)
        if not isinstance(sample, Mapping) or set(sample) != {"wrenches", "rejected", "payload"}:
            raise ValueError("contact sampler returned an incomplete transaction")
        if set(sample["wrenches"]) != set(self._mechanical):
            raise ValueError("contact wrenches must cover every participant")
        if set(sample["rejected"]) != set(self.layout.world_keys) or any(
            type(value) is not bool for value in sample["rejected"].values()
        ):
            raise ValueError("contact rejection must cover every logical world")
        wrench = np.zeros((storage.count, storage.bmax, 6), np.float32)
        for key, value in sample["wrenches"].items():
            count = self.layout.entries[key].body_count
            wrench[self._rows[key], :count] = as_fp32(
                value, "participant contact wrench", (count, 6)
            )
        self._batch._upload(self._wrench, wrench, "control")
        self._transfers["uploaded_bytes"] += wrench.nbytes
        return dict(
            dt=dt,
            src=src,
            dst=dst,
            epoch=self._batch._state_epoch,
            held=held,
            entry=entry,
            arrays=arrays,
            sample=sample,
            rows=MappingProxyType(self._rows),
        )

    def _before_finalize(self, attempt):
        """Internal fault seam: tests may return participant -> original failure code."""
        return None

    def _stage_tree(self, storage, attempt, dt, src, dst):
        batch = self._batch
        batch._launch(
            self._project[src],
            storage.count,
            storage.meta,
            storage.linear,
            storage.angular,
            self._wrench,
            storage.vector,
            self._projected,
        )
        storage.stage_solve(dt, src, dst, batch._ticks[src])
        self._stages["candidate_solve_stages"] += 1
        storage.stage_candidate_geometry(dst)
        failures = self._before_finalize(attempt)
        forced = {key: 7 for key in self._mechanical if attempt["sample"]["rejected"][key.world]}
        if failures is not None:
            if not isinstance(failures, Mapping) or any(
                key not in self._mechanical or type(code) is not int or code != 6
                for key, code in failures.items()
            ):
                raise ValueError("invalid internal participant failure injection")
            forced.update(failures)
        if forced:
            info = self._read_contact(storage.info)
            for key, code in forced.items():
                if info[self._rows[key], 2] == 0:
                    info[self._rows[key], 2] = code
            batch._upload(storage.info, info, "control")
        batch._launch(
            self._reduce,
            len(self.layout.world_keys),
            self._world_offsets,
            self._world_rows,
            storage.info,
            storage.flags,
        )
        storage.stage_finalize(src, dst, batch._ticks[src], batch._ticks[dst], batch._status)
        self._stages["candidate_finalize_stages"] += 1

    def _read_candidate(self, attempt, dst):
        storage = self._storage
        packet = self._read_packets(dst)
        arrays = {name: attempt["arrays"][name] for name in ("linear", "angular", "mass")}
        arrays.update(
            {
                name: self._read_contact(getattr(storage, name))
                for name in ("vector", "entry_pose", "info", "diag")
            }
        )
        selected = [index for index in self._keys if storage._pending_info[0][index]]
        diagnostics = floating_contact_diagnostics(storage, selected, arrays)
        projected = self._read_contact(self._projected)
        return dict(
            packet=packet,
            diagnostics={self._keys[index]: value for index, value in diagnostics.items()},
            projected={
                key: _owned(projected[self._rows[key], : self.layout.entries[key].velocity_count])
                for key in self._mechanical
            },
        )

    def _finish_attempt(self, attempt, candidate, status):
        accepted, names = {}, {}
        for world in self.layout.worlds.values():
            flags = {
                bool(status[self._batch._slots[self._mechanical[key]]])
                for key in world.participant_keys
            }
            if len(flags) != 1:
                raise RuntimeError("partial logical-world acceptance")
            accepted[world.world] = flags.pop()
            names[world.world] = "ACCEPTED" if accepted[world.world] else "REJECTED"
            for key in world.participant_keys:
                before, after = attempt["entry"][key]["state"], candidate["packet"][key]["state"]
                if accepted[world.world]:
                    if after.tick != before.tick + 1:
                        raise RuntimeError("participant tick is not one common step")
                elif after != before:
                    raise RuntimeError("rejected participant state was not retained")
        publication = self._prepare_publication(
            attempt,
            candidate["packet"],
            candidate["diagnostics"],
            candidate["projected"],
            MappingProxyType(accepted),
            MappingProxyType(names),
        )
        count = sum(accepted.values())
        return (
            publication,
            self._logical_accepted_steps + count,
            self._logical_rejected_steps + len(accepted) - count,
        )

    def _prepare_publication(self, attempt, candidate, diagnostics, projected, accepted, status):
        return self._adapter.prepare_publication(
            attempt["entry"],
            candidate,
            diagnostics,
            projected,
            accepted,
            status,
            attempt["sample"]["payload"],
            attempt["dt"],
        )

    def _check_attempt(self, attempt):
        self._check_open()
        if self._batch._state_epoch != attempt["epoch"] or self._batch._active != attempt["src"]:
            raise ValueError("stale participant transaction")
        for name, value in zip(
            ("_held_drive", "_held_effort", "_held_loads"), attempt["held"], strict=True
        ):
            if not np.array_equal(getattr(self._storage, name), value):
                raise ValueError("held controls changed during participant transaction")

    def _publish(self, commit):
        # No allocation, validation or user callback after the accepted-bank swap.
        self.last_publication, self._logical_accepted_steps, self._logical_rejected_steps = commit
