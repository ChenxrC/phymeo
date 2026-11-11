"""Shared multi-participant transaction over the existing four TYX tree storages.

Mechanical row IDs are private storage addresses. Public identities and failure
domains remain logical worlds. This service never duplicates the integrator.
The bounded domain retains at most 33 physical bodies per logical world.
"""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType

import numpy as np

from ._resident_contact_plan import _ResidentContactMechanics
from ._resident_floating_contact import floating_contact_diagnostics
from ._resident_participant_layout import ResidentParticipantLayout
from ._resident_tree_world import TreeDiagnostic
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
        if any(world.body_count > 33 for world in layout.worlds.values()):
            raise ValueError("resident scene supports at most 33 physical bodies per logical world")
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
        self._groups = {}
        self._storage_stages = {}
        self._key_groups = {}
        self._rows = {}
        try:
            from ._tyx_participant_transaction import make_participant_transaction_kernels
            from ._tyx_resident_contact_kernels import make_contact_kernels
            from ._tyx_resident_floating_contact import make_floating_contact_kernels

            self._contact_kernels = make_contact_kernels()
            self._floating_contact_kernels = make_floating_contact_kernels()
            self._transaction_kernels = make_participant_transaction_kernels()
            offsets, slots = [0], []
            for world in layout.worlds.values():
                slots.extend(
                    self._batch._slots[self._mechanical[key]] for key in world.participant_keys
                )
                offsets.append(len(slots))
            with self._batch._device():
                self._world_offsets = self._batch._buffer(
                    (len(offsets),), np.int32, offsets, "topology"
                )
                self._world_slots = self._batch._buffer((len(slots),), np.int32, slots, "topology")
                self._failed_slots = self._batch._buffer((len(slots),), np.int32)
                for _, storage in self._batch._tree_storages():
                    floating = storage in (
                        self._batch._floating_trees,
                        self._batch._floating_affine_trees,
                    )
                    affine = storage in (
                        self._batch._affine_trees,
                        self._batch._floating_affine_trees,
                    )
                    physical = storage.physical if affine else storage
                    body_max = physical.bmax if floating else physical.nmax
                    dof_max = physical.dmax if floating else physical.nmax
                    row_slots = [0] * physical.count
                    group = dict(
                        storage=storage,
                        physical=physical,
                        floating=floating,
                        affine=affine,
                        body_max=body_max,
                        dof_max=dof_max,
                        worlds=tuple(storage.rows),
                        wrench=self._batch._buffer((physical.count, body_max, 6), np.float32),
                        projected=self._batch._buffer((physical.count, dof_max), np.float32),
                        velocity=None
                        if floating
                        else self._batch._buffer((physical.count, body_max, 6), np.float32),
                    )
                    group["family"] = ("floating_" if floating else "") + (
                        "affine" if affine else "tree"
                    )
                    self._storage_stages[group["family"]] = dict.fromkeys(self._stages, 0)
                    self._groups[storage] = group
                    for index, row in storage.rows.items():
                        key = self._keys[index]
                        self._rows[key] = row
                        self._key_groups[key] = group
                        row_slots[row] = self._batch._slots[index]
                    group["row_slots"] = self._batch._buffer(
                        (physical.count,), np.int32, row_slots, "topology"
                    )
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
            self._batch._restore(
                {
                    self._mechanical[key]: replace(value, world=self._mechanical[key])
                    for key, value in checkpoints.items()
                },
                before_commit=self._before_restore_commit,
            )
            if checkpoints:
                self.last_publication = None

    def _before_restore_commit(self):
        """Fault seam after all restore checks, before any accepted bank changes."""
        return None

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
                "storage_stage_counts": {
                    key: dict(value) for key, value in self._storage_stages.items()
                },
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
        raw = {}
        with self._batch._device():
            for group in self._groups.values():
                raw.update(_ResidentContactMechanics._read_storage_packet(self, group, bank))
        result = {}
        for index, packet in raw.items():
            key = self._keys[index]
            result[key] = {
                "state": replace(packet["state"], world=key.world),
                "geometry": packet["geometry"],
                "energy": packet["energy"],
                "body_twist": packet["body_twist"],
            }
        for world in self.layout.worlds.values():
            if len({result[key]["state"].tick for key in world.participant_keys}) != 1:
                raise RuntimeError("logical world participants have inconsistent ticks")
        return result

    def _handles(self, storage):
        return storage in self._groups

    def _launch_contact(self, name, bank, *args):
        self._batch._launch(self._contact_kernels[name][bank], *args)

    def _record_stage(self, group, name):
        self._stages[name] += 1
        self._storage_stages[group["family"]][name] += 1

    def _begin_attempt(self, dt, src, dst):
        held = _ResidentContactMechanics._control_snapshot(self)
        for group in self._groups.values():
            group["physical"].stage_mechanics(src)
            self._record_stage(group, "entry_mechanics_stages")
        entry = self._read_packets(src)
        arrays = {}
        for storage, group in self._groups.items():
            arrays[storage] = {
                name: self._read_contact(getattr(group["physical"], name))
                for name in ("linear", "angular", "mass", "vector")
            }
        for key, packet in entry.items():
            row = self._rows[key]
            group = self._key_groups[key]
            spec = group["physical"].specs[self._mechanical[key]]
            b, d = (spec.b, spec.d) if group["floating"] else (spec.n, spec.n)
            data = arrays[group["storage"]]
            packet.update(
                linear=_owned(data["linear"][row, :b, :, :d]),
                angular=_owned(data["angular"][row, :b, :, :d]),
                mass=_owned(data["mass"][row, :d, :d]),
                bias=_owned(data["vector"][row, :d, 0]),
                external=_owned(data["vector"][row, :d, 1]),
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
        wrenches = {
            storage: np.zeros((group["physical"].count, group["body_max"], 6), np.float32)
            for storage, group in self._groups.items()
        }
        for key, value in sample["wrenches"].items():
            count = self.layout.entries[key].body_count
            group = self._key_groups[key]
            wrenches[group["storage"]][self._rows[key], :count] = as_fp32(
                value, "participant contact wrench", (count, 6)
            )
        for storage, wrench in wrenches.items():
            self._batch._upload(self._groups[storage]["wrench"], wrench, "control")
            self._transfers["uploaded_bytes"] += wrench.nbytes
        return dict(
            dt=dt,
            src=src,
            dst=dst,
            epoch=self._batch._state_epoch,
            held=held,
            entry=entry,
            groups=arrays,
            staged=set(),
            sample=sample,
            rows=MappingProxyType(self._rows),
        )

    def _before_finalize(self, attempt):
        """Internal fault seam: tests may return participant -> original failure code."""
        return None

    def _stage_tree(self, storage, attempt, dt, src, dst):
        batch = self._batch
        group = self._groups[storage]
        physical = group["physical"]
        if storage in attempt["staged"]:
            raise RuntimeError("participant storage was solved twice")
        kernels = self._floating_contact_kernels if group["floating"] else self._contact_kernels
        batch._launch(
            kernels["project"][src],
            physical.count,
            physical.meta,
            physical.linear,
            physical.angular,
            group["wrench"],
            physical.vector,
            group["projected"],
        )
        if group["affine"]:
            storage.stage_solve_from_mechanics(dt, src, dst, batch._ticks[src])
        else:
            storage.stage_solve(dt, src, dst, batch._ticks[src])
        self._record_stage(group, "candidate_solve_stages")
        physical.stage_candidate_geometry(dst)
        attempt["staged"].add(storage)

    def _finalize_trees(self, attempt, dt, src, dst):
        """Reduce every physical family's status before retaining any candidate."""
        if attempt["staged"] != set(self._groups):
            raise RuntimeError("participant storages did not all produce candidates")
        batch = self._batch
        failures = self._before_finalize(attempt)
        forced = {
            key: 9 if self._key_groups[key]["affine"] else 7
            for key in self._mechanical
            if attempt["sample"]["rejected"][key.world]
        }
        if failures is not None:
            if not isinstance(failures, Mapping) or any(
                key not in self._mechanical or type(code) is not int or code != 6
                for key, code in failures.items()
            ):
                raise ValueError("invalid internal participant failure injection")
            forced.update(failures)
        for storage, group in self._groups.items():
            physical = group["physical"]
            local = {key: code for key, code in forced.items() if self._key_groups[key] is group}
            if local:
                info = self._read_contact(physical.info)
                for key, code in local.items():
                    if info[self._rows[key], 2] == 0:
                        info[self._rows[key], 2] = code
                batch._upload(physical.info, info, "control")
            batch._launch(
                self._transaction_kernels.collect,
                physical.count,
                group["row_slots"],
                physical.info,
                physical.flags,
                self._failed_slots,
            )
        batch._launch(
            self._transaction_kernels.reduce_slots,
            len(self.layout.world_keys),
            self._world_offsets,
            self._world_slots,
            self._failed_slots,
        )
        for storage, group in self._groups.items():
            physical = group["physical"]
            batch._launch(
                self._transaction_kernels.scatter,
                physical.count,
                group["row_slots"],
                self._failed_slots,
                physical.info,
            )
            if group["affine"]:
                storage.stage_finalize_and_retain(
                    src, dst, batch._ticks[src], batch._ticks[dst], batch._status
                )
            else:
                storage.stage_finalize(
                    src, dst, batch._ticks[src], batch._ticks[dst], batch._status
                )
            self._record_stage(group, "candidate_finalize_stages")

    def _read_candidate(self, attempt, dst):
        packet = self._read_packets(dst)
        diagnostics, projected = {}, {}
        for storage, group in self._groups.items():
            physical = group["physical"]
            selected = [index for index in group["worlds"] if storage._pending_info[0][index]]
            if group["affine"]:
                local_diagnostics = storage._read_diagnostics(selected)
            else:
                arrays = {
                    name: attempt["groups"][storage][name] for name in ("linear", "angular", "mass")
                }
                arrays.update(
                    {
                        name: self._read_contact(getattr(physical, name))
                        for name in ("vector", "info", "diag")
                    }
                )
                if group["floating"]:
                    arrays["entry_pose"] = self._read_contact(physical.entry_pose)
                    local_diagnostics = floating_contact_diagnostics(storage, selected, arrays)
                else:
                    local_diagnostics = {}
                    for index in selected:
                        row, n = physical.rows[index], physical.specs[index].n
                        v = arrays["vector"][row, :n]
                        info, diag = arrays["info"][row], arrays["diag"][row]
                        local_diagnostics[index] = TreeDiagnostic(
                            int(info[0]),
                            int(info[1]),
                            float(diag[0]),
                            v[:, 4],
                            v[:, 5],
                            arrays["linear"][row, :n, :, :n],
                            arrays["angular"][row, :n, :, :n],
                            arrays["mass"][row, :n, :n],
                            v[:, 0],
                            v[:, 1],
                            v[:, 2],
                            v[:, 3],
                            int(info[3]),
                            float(diag[1]),
                            float(diag[2]),
                        )
            diagnostics.update(
                {self._keys[index]: value for index, value in local_diagnostics.items()}
            )
            values = self._read_contact(group["projected"])
            for index in group["worlds"]:
                key = self._keys[index]
                projected[key] = _owned(
                    values[self._rows[key], : self.layout.entries[key].physical_velocity_count]
                )
        return dict(
            packet=packet,
            diagnostics=diagnostics,
            projected=projected,
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
        for storage, name, value in attempt["held"]:
            if not np.array_equal(getattr(storage, name), value):
                raise ValueError("held controls changed during participant transaction")

    def _publish(self, commit):
        # No allocation, validation or user callback after the accepted-bank swap.
        self.last_publication, self._logical_accepted_steps, self._logical_rejected_steps = commit
