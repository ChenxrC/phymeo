"""Entry-sampled finite static contact in the existing resident batch transaction.

HOST narrow phase is explicit. Device state, full mechanics, controls and the
single accepted-bank swap remain owned by TyxWorldBatch.
"""

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..constraints.robot_contact import RobotContactMaterial
from ._resident_collision_plan import ResidentCollisionPlan, ResidentStaticBinding
from ._resident_contact_bridge import (
    ContactShapeFrame,
    ContactSiteCapacityExceeded,
    finish_contact_loads,
    sample_tree_contacts,
)
from ._resident_tree_binding import ResidentTreeBinding
from ._resident_tree_world import TreeKinematics, _owned
from ._world_step_layout import as_fp32


@dataclass(frozen=True, eq=False)
class ResidentContactReceipt:
    world: int
    entry_tick: int
    exit_tick: int
    dt_s: float
    entry_state: object
    accepted_state: object
    entry_geometry: TreeKinematics
    accepted_geometry: TreeKinematics
    input_external: np.ndarray
    selected_drive: np.ndarray
    law_wrench_fp64: np.ndarray
    uploaded_wrench_fp32: np.ndarray
    contact_generalized: np.ndarray
    actual_drive: np.ndarray
    sites: tuple
    static_reactions: Mapping
    diagnostics: Mapping

    def __post_init__(self):
        for name in (
            "input_external",
            "selected_drive",
            "law_wrench_fp64",
            "contact_generalized",
            "actual_drive",
        ):
            object.__setattr__(self, name, _owned(getattr(self, name), np.float64))
        object.__setattr__(
            self, "uploaded_wrench_fp32", _owned(self.uploaded_wrench_fp32, np.float32)
        )
        object.__setattr__(self, "sites", tuple(self.sites))
        object.__setattr__(self, "static_reactions", MappingProxyType(dict(self.static_reactions)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def _site_limit(value):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not 0 <= value <= 4096
    ):
        raise ValueError("site capacity must be an integer from 0 to 4096")
    return int(value)


class ResidentContactPlan(ResidentCollisionPlan):
    """Immutable registration plus owned temporary load/velocity storage."""

    def __init__(
        self,
        batch,
        model,
        bindings,
        materials_by_pair,
        *,
        candidate_capacity=4096,
        site_capacity=128,
        sampling="discrete_entry",
    ):
        if sampling != "discrete_entry":
            raise ValueError("resident contact supports only discrete_entry sampling, not CCD")
        if not isinstance(materials_by_pair, Mapping):
            raise TypeError("contact materials must be a complete pair mapping")
        bindings = tuple(bindings)
        if not all(type(b) in (ResidentTreeBinding, ResidentStaticBinding) for b in bindings):
            raise ValueError("contact binding requires an ordinary tree or finite static shape")
        self._contact_bindings = MappingProxyType({b.key: b for b in bindings})
        self._materials = MappingProxyType(dict(materials_by_pair))
        self._site_config = site_capacity
        self._contact_resources = ExitStack()
        self._contact_counts = dict(
            device_allocations=0,
            device_owned_bytes=0,
            uploaded_bytes=0,
            downloaded_bytes=0,
            host_read_bytes=0,
            kernel_launches=0,
        )
        self._contact_kernels = None
        self._wrench = self._reject = self._projected = self._velocity = None
        self._last_accepted = MappingProxyType({})
        self._last_status = MappingProxyType({})
        super().__init__(batch, model, bindings, candidate_capacity=candidate_capacity)

    @property
    def last_accepted(self):
        with self._batch._lock:
            return self._last_accepted

    @property
    def last_status(self):
        with self._batch._lock:
            return self._last_status

    def _prepare_topology(self, slots):
        super()._prepare_topology(slots)
        if any(
            not s.rigid
            or s.kind not in ("sphere", "capsule", "hull")
            or (s.kind == "hull" and s.radius != 0)
            for s in self._model.shapes
        ):
            raise ValueError("contact geometry must be a finite sphere, capsule or unrounded hull")
        allowed = (
            ()
            if self._space is None
            else tuple(
                (self._shape_keys[i], self._shape_keys[j])
                for (i, j, _), flag in zip(self._pairs, self._space.defaults, strict=True)
                if flag == 5
            )
        )
        if set(self._materials) != set(allowed):
            raise ValueError(
                "contact materials must exactly cover the allowed canonical pair domain"
            )
        if any(
            not isinstance(m, RobotContactMaterial) or m.friction != 0
            for m in self._materials.values()
        ):
            raise ValueError("resident contact requires explicit frictionless RobotContactMaterial")
        worlds = tuple(sorted(self._world_shape_rows))
        if isinstance(self._site_config, Mapping):
            if any(
                isinstance(w, (bool, np.bool_)) or not isinstance(w, Integral)
                for w in self._site_config
            ) or set(self._site_config) != set(worlds):
                raise ValueError("site capacities must cover exactly the contact-bound worlds")
            caps = {w: _site_limit(self._site_config[w]) for w in worlds}
        else:
            cap = _site_limit(self._site_config)
            caps = dict.fromkeys(worlds, cap)
        self._site_caps = MappingProxyType(caps)
        self._contact_worlds = worlds
        self._world_materials = {
            w: MappingProxyType(
                {pair: law for pair, law in self._materials.items() if pair[0].world == w}
            )
            for w in worlds
        }

    def _prepare_device(self, view, slots, paths):
        super()._prepare_device(view, slots, paths)
        from ._tyx_resident_contact_kernels import make_contact_kernels

        storage = self._batch._trees
        t, n = storage.count, storage.nmax
        self._contact_kernels = make_contact_kernels()
        self._wrench = self._buffer((t, n, 6), np.float32)
        self._reject = self._buffer((t,), np.int32)
        self._projected = self._buffer((t, n), np.float32)
        self._velocity = self._buffer((t, n, 6), np.float32)
        for suffix in ("uploaded_bytes", "downloaded_bytes"):
            self._batch._counts.setdefault("contact_" + suffix, 0)

    def _buffer(self, shape, dtype, initial=None):
        """Register actual allocation before returning, including failure injection."""
        batch = self._batch
        data = (
            np.zeros(shape, dtype)
            if initial is None
            else np.asarray(initial, dtype=dtype).reshape(shape)
        )
        if batch._mode == "interpret":
            return data.copy()
        result = batch._tx.cuda_array(data)
        self._contact_resources.callback(result.free)
        self._contact_counts["device_allocations"] += 1
        self._contact_counts["device_owned_bytes"] += result.nbytes
        self._contact_counts["uploaded_bytes"] += data.nbytes
        return result

    def _upload_contact(self, buffer, data):
        self._batch._upload(buffer, data, "contact")
        if self._batch._mode == "cuda":
            self._contact_counts["uploaded_bytes"] += data.nbytes

    def _read_contact(self, buffer):
        data = self._batch._read(buffer, "contact")
        key = "downloaded_bytes" if self._batch._mode == "cuda" else "host_read_bytes"
        self._contact_counts[key] += data.nbytes
        return data

    def _launch_contact(self, name, bank, *args):
        self._batch._launch(self._contact_kernels[name][bank], *args)
        self._contact_counts["kernel_launches"] += 1

    def _handles(self, storage):
        return bool(self._contact_worlds) and storage is self._batch._trees

    def _read_packet(self, bank):
        batch, storage = self._batch, self._batch._trees
        self._launch_contact(
            "velocity",
            bank,
            storage.count,
            storage.meta,
            storage.parents,
            storage.kinds,
            storage.states[bank],
            storage.fk[bank],
            self._velocity,
        )
        batch._sync()
        state, fk, ticks, energy, velocity = [
            self._read_contact(a)
            for a in (
                storage.states[bank],
                storage.fk[bank],
                batch._ticks[bank],
                storage.energy[bank],
                self._velocity,
            )
        ]
        packets = {}
        for world in self._contact_worlds:
            row, spec = storage.rows[world], storage.specs[world]
            n, tick = spec.n, int(ticks[batch._slots[world]])
            raw = fk[row, :n].astype(np.float64)
            position, rotation = raw[:, :3] + spec.anchor, raw[:, 3:12].reshape(n, 3, 3)
            com = raw[:, 15:18] + spec.anchor
            if not np.isfinite(raw).all() or not np.isfinite(velocity[row, :n]).all():
                raise FloatingPointError("nonfinite contact geometry/velocity readback")
            checkpoint = spec.checkpoint(world, state[row], tick, storage._held[row])
            geometry = TreeKinematics(
                tick,
                spec.joint_types,
                state[row, :n, 0],
                state[row, :n, 1],
                position,
                rotation,
                com,
            )
            frames = {}
            for shape in self._model.shapes:
                if shape.key.world != world:
                    continue
                binding = self._contact_bindings[shape.key]
                if isinstance(binding, ResidentTreeBinding):
                    group = binding.group_index
                    p, r = position[group], rotation[group]
                    points = p + binding.local_points_m @ r.T
                    frame = ContactShapeFrame(
                        shape,
                        points,
                        group,
                        p,
                        r,
                        com[group],
                        velocity[row, group, :3],
                        velocity[row, group, 3:],
                    )
                else:
                    points = binding.motion.start
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
                frames[shape.key] = frame
            packets[world] = dict(
                state=checkpoint,
                geometry=geometry,
                frames=MappingProxyType(frames),
                energy=float(energy[row]),
            )
        return packets

    def _control_snapshot(self):
        records = []
        for _, storage in self._batch._tree_storages():
            if storage is self._batch._trees:
                targets = ((storage, "_held"),)
            else:
                targets = (
                    (storage, "_held_drive"),
                    (storage, "_held_loads"),
                    (storage, "_held_commands"),
                    (storage.physical, "_held"),
                )
            records.extend((target, name, getattr(target, name).copy()) for target, name in targets)
        return tuple(records)

    def _begin_attempt(self, dt, src, dst):
        self._check_open()
        result, pending = self._stage_query()
        batch = self._batch
        attempt = dict(
            dt=dt,
            src=src,
            dst=dst,
            epoch=batch._state_epoch,
            result=result,
            pending=pending,
            loads={},
            rejected={},
            held=batch._held.copy(),
            tree_held=self._control_snapshot(),
        )
        if not self._contact_worlds:
            return attempt
        storage = batch._trees
        storage.stage_mechanics(src)
        entry = self._read_packet(src)
        vectors, mass, linear, angular = [
            self._read_contact(a)
            for a in (storage.vector, storage.mass, storage.linear, storage.angular)
        ]
        attempt.update(
            entry=entry, entry_vectors=vectors, mass=mass, linear=linear, angular=angular
        )
        wrench = np.zeros((storage.count, storage.nmax, 6), np.float32)
        reject = np.zeros(storage.count, np.int32)
        for world in self._contact_worlds:
            row, spec = storage.rows[world], storage.specs[world]
            try:
                loads = sample_tree_contacts(
                    world=world,
                    frames=entry[world]["frames"],
                    materials_by_pair=self._world_materials[world],
                    candidates=tuple(pair for pair in result.candidates if pair[0].world == world),
                    group_count=spec.n,
                    site_capacity=self._site_caps[world],
                )
            except ContactSiteCapacityExceeded as error:
                reject[row] = 1
                attempt["rejected"][world] = (error.count, error.limit)
                continue
            attempt["loads"][world] = loads
            wrench[row, : spec.n] = as_fp32(loads.wrench_fp64, "contact COM wrench", (spec.n, 6))
        self._upload_contact(self._wrench, wrench)
        self._upload_contact(self._reject, reject)
        attempt["uploaded"] = wrench
        return attempt

    def _stage_tree(self, attempt, dt, src, dst):
        batch, storage = self._batch, self._batch._trees
        # Full mechanics was assembled at the entry before sampling the law.
        self._launch_contact(
            "project",
            src,
            storage.count,
            storage.meta,
            storage.linear,
            storage.angular,
            self._wrench,
            storage.vector,
            self._projected,
        )
        storage.stage_solve(dt, src, dst, batch._ticks[src])
        storage.stage_candidate_geometry(dst)
        self._launch_contact("reject", src, storage.count, self._reject, storage.info)
        storage.stage_finalize(src, dst, batch._ticks[src], batch._ticks[dst], batch._status)

    def _read_candidate(self, attempt, dst):
        if not self._contact_worlds:
            return {}
        storage = self._batch._trees
        packet = self._read_packet(dst)
        return dict(
            packet=packet,
            vectors=self._read_contact(storage.vector),
            projected=self._read_contact(self._projected),
            info=self._read_contact(storage.info),
            diag=self._read_contact(storage.diag),
        )

    def _check_attempt(self, attempt):
        batch = self._batch
        self.require_current(attempt["result"])
        if (
            batch._state_epoch != attempt["epoch"]
            or batch._active != attempt["src"]
            or not np.array_equal(batch._held, attempt["held"])
            or any(
                not np.array_equal(getattr(storage, name), held)
                for storage, name, held in attempt["tree_held"]
            )
        ):
            raise ValueError("stale contact attempt state, epoch or held commands")

    def _finish_attempt(self, attempt, candidate, status):
        self._check_attempt(attempt)
        batch = self._batch
        accepted = dict(self.last_accepted)
        names = {
            w: ("ACCEPTED" if status[slot] else "REJECTED") for w, slot in batch._slots.items()
        }
        for _, storage in batch._tree_storages():
            names.update(storage._pending_info[1])
        for world in self._contact_worlds:
            entry, final = attempt["entry"][world], candidate["packet"][world]
            if not status[batch._slots[world]]:
                if final["state"] != entry["state"]:
                    raise RuntimeError("rejected contact world failed complete state retention")
                continue
            if final["state"].tick != entry["state"].tick + 1:
                raise RuntimeError("contact candidate tick did not advance once")
            loads = finish_contact_loads(attempt["loads"][world], final["frames"])
            storage = batch._trees
            row, n = storage.rows[world], storage.specs[world].n
            vector = candidate["vectors"][row, :n]
            wrench = attempt["uploaded"][row, :n]
            force_power = sum(
                float(
                    np.dot(site.first_force_n, site.first_velocity_m_s)
                    + np.dot(site.second_force_n, site.second_velocity_m_s)
                )
                for site in loads.sites
            )
            diagnostics = dict(
                sampling="discrete_entry",
                entry_frames=entry["frames"],
                accepted_frames=final["frames"],
                entry_kinetic_energy_j=entry["energy"],
                accepted_kinetic_energy_j=final["energy"],
                kinetic_energy_change_j=final["energy"] - entry["energy"],
                entry_power_times_h_j=force_power * attempt["dt"],
                application_work_j=sum(site.application_work_j for site in loads.sites),
                material_work_j=sum(site.material_work_j for site in loads.sites),
                wrench_quantization_error=_owned(
                    wrench.astype(float) - loads.wrench_fp64, np.float64
                ),
                mass=_owned(attempt["mass"][row, :n, :n]),
                linear_jacobian=_owned(attempt["linear"][row, :n, :, :n]),
                angular_jacobian=_owned(attempt["angular"][row, :n, :, :n]),
                bias=_owned(attempt["entry_vectors"][row, :n, 0]),
                total_external=_owned(vector[:, 1]),
                acceleration=_owned(vector[:, 3]),
                control_iterations=int(candidate["info"][row, 3]),
                control_residual=float(candidate["diag"][row, 1]),
                equation_residual=float(candidate["diag"][row, 2]),
            )
            accepted[world] = ResidentContactReceipt(
                world,
                entry["state"].tick,
                final["state"].tick,
                attempt["dt"],
                entry["state"],
                final["state"],
                entry["geometry"],
                final["geometry"],
                attempt["entry_vectors"][row, :n, 1],
                vector[:, 11:14],
                loads.wrench_fp64,
                wrench,
                candidate["projected"][row, :n],
                vector[:, 2],
                loads.sites,
                loads.static_reactions,
                diagnostics,
            )
        commit = (
            MappingProxyType(accepted),
            MappingProxyType(names),
            attempt["result"],
            attempt["pending"],
        )
        return commit

    def _publish(self, commit):
        self._last_accepted, self._last_status, result, pending = commit
        self._publish_query(result, pending)

    def stats(self):
        result = super().stats()
        counts = self._contact_counts
        result["device_owned_bytes"] += counts["device_owned_bytes"]
        result["device_allocations"] += counts["device_allocations"]
        result.update({("contact_" + key): value for key, value in counts.items()})
        return result

    def close(self):
        try:
            super().close()
        finally:
            self._contact_resources.close()
            self._contact_counts["device_owned_bytes"] = 0
            self._wrench = self._reject = self._projected = self._velocity = None
            self._contact_kernels = None
