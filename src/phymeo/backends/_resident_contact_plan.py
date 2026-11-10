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
from ._resident_affine_binding import ResidentAffineTreeBinding
from ._resident_affine_tree import AffineTreeKinematics
from ._resident_collision_plan import ResidentCollisionPlan, ResidentStaticBinding
from ._resident_contact_bridge import (
    ContactShapeFrame,
    ContactSiteCapacityExceeded,
    finish_contact_loads,
    sample_tree_contacts,
)
from ._resident_floating_affine_binding import ResidentFloatingAffineTreeBinding
from ._resident_floating_binding import ResidentFloatingTreeBinding
from ._resident_floating_contact import (
    floating_contact_diagnostics,
    read_floating_contact_packet,
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


class _ResidentContactMechanics:
    """Shared one-entry/one-solve contact transaction for plans and page bundles."""

    def _initialize_contact_mechanics(self):
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
        self._floating_contact_kernels = None
        self._contact_groups = {}
        self._world_groups = {}
        self._last_accepted = MappingProxyType({})
        self._last_status = MappingProxyType({})

    @property
    def last_accepted(self):
        with self._batch._lock:
            return self._last_accepted

    @property
    def last_status(self):
        with self._batch._lock:
            return self._last_status

    def _prepare_contact_registration(self):
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
        if any(not isinstance(m, RobotContactMaterial) for m in self._materials.values()):
            raise ValueError("resident contact requires explicit RobotContactMaterial")
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
        self._contact_shapes = MappingProxyType({
            w: tuple(s for s in self._model.shapes if s.key.world == w) for w in worlds
        })
        self._world_materials = {
            w: MappingProxyType(
                {pair: law for pair, law in self._materials.items() if pair[0].world == w}
            )
            for w in worlds
        }

    def _prepare_contact_device(self):
        from ._tyx_resident_contact_kernels import make_contact_kernels

        self._contact_kernels = make_contact_kernels()
        for _, storage in self._batch._tree_storages():
            worlds = tuple(w for w in self._contact_worlds if w in storage.rows)
            if not worlds:
                continue
            floating = storage in (self._batch._floating_trees, self._batch._floating_affine_trees)
            affine = storage in (self._batch._affine_trees, self._batch._floating_affine_trees)
            physical = storage.physical if affine else storage
            t = physical.count
            body_max = physical.bmax if floating else physical.nmax
            velocity_max = physical.dmax if floating else physical.nmax
            if floating and self._floating_contact_kernels is None:
                from ._tyx_resident_floating_contact import make_floating_contact_kernels

                self._floating_contact_kernels = make_floating_contact_kernels()
            group = dict(
                storage=storage,
                physical=physical,
                affine=affine,
                floating=floating,
                body_max=body_max,
                worlds=worlds,
                wrench=self._buffer((t, body_max, 6), np.float32),
                reject=self._buffer((t,), np.int32),
                projected=self._buffer((t, velocity_max), np.float32),
                velocity=None if floating else self._buffer((t, body_max, 6), np.float32),
            )
            self._contact_groups[storage] = group
            self._world_groups.update({w: group for w in worlds})
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
        return storage in self._contact_groups

    def _read_packet(self, bank):
        packets = {}
        for group in self._contact_groups.values():
            packets.update(self._read_storage_packet(group, bank))
        return packets

    def _read_storage_packet(self, group, bank):
        batch, storage = self._batch, group["physical"]
        owner = group["storage"]
        if group["floating"]:
            return read_floating_contact_packet(self, owner, group["worlds"], bank)
        self._launch_contact(
            "velocity",
            bank,
            storage.count,
            storage.meta,
            storage.parents,
            storage.kinds,
            storage.states[bank],
            storage.fk[bank],
            group["velocity"],
        )
        batch._sync()
        state, fk, ticks, energy, velocity = [
            self._read_contact(a)
            for a in (
                storage.states[bank],
                storage.fk[bank],
                batch._ticks[bank],
                storage.energy[bank],
                group["velocity"],
            )
        ]
        reduced = self._read_contact(owner.states[bank]) if group["affine"] else None
        packets = {}
        for world in group["worlds"]:
            row, spec = storage.rows[world], storage.specs[world]
            n, tick = spec.n, int(ticks[batch._slots[world]])
            raw = fk[row, :n].astype(np.float64)
            position, rotation = raw[:, :3] + spec.anchor, raw[:, 3:12].reshape(n, 3, 3)
            com = raw[:, 15:18] + spec.anchor
            if not np.isfinite(raw).all() or not np.isfinite(velocity[row, :n]).all():
                raise FloatingPointError("nonfinite contact geometry/velocity readback")
            geometry_args = (
                tick,
                spec.joint_types,
                state[row, :n, 0],
                state[row, :n, 1],
                position,
                rotation,
                com,
            )
            if group["affine"]:
                affine_spec = owner.specs[world]
                checkpoint = affine_spec.checkpoint(
                    world,
                    reduced[row],
                    state[row],
                    tick,
                    owner._held_drive[row],
                    owner._held_loads[row],
                )
                geometry = AffineTreeKinematics(
                    *geometry_args,
                    reduced[row, : affine_spec.k, 0],
                    reduced[row, : affine_spec.k, 1],
                    affine_spec.joint_names,
                    affine_spec.full_joint_names,
                )
            else:
                checkpoint = spec.checkpoint(world, state[row], tick, storage._held[row])
                geometry = TreeKinematics(*geometry_args)
            frames = {}
            for shape in self._contact_shapes[world]:
                binding = self._contact_bindings[shape.key]
                if isinstance(binding, ResidentTreeBinding):
                    group_index = binding.group_index
                    p, r = position[group_index], rotation[group_index]
                    points = p + binding.local_points_m @ r.T
                    frame = ContactShapeFrame(
                        shape,
                        points,
                        group_index,
                        p,
                        r,
                        com[group_index],
                        velocity[row, group_index, :3],
                        velocity[row, group_index, 3:],
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
            elif storage is self._batch._floating_affine_trees:
                targets = tuple(
                    (target, name)
                    for target in (storage, storage.physical)
                    for name in ("_held_drive", "_held_effort", "_held_loads")
                )
            elif storage is self._batch._floating_trees:
                targets = (
                    (storage, "_held_drive"),
                    (storage, "_held_effort"),
                    (storage, "_held_loads"),
                )
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
        attempt["groups"] = {}
        for group in self._contact_groups.values():
            owner, physical = group["storage"], group["physical"]
            owner.invalidate_diagnostics()
            physical.stage_mechanics(src)
        entry = self._read_packet(src)
        attempt["entry"] = entry
        for group in self._contact_groups.values():
            storage, physical = group["storage"], group["physical"]
            vectors, mass, linear, angular = [
                self._read_contact(a)
                for a in (physical.vector, physical.mass, physical.linear, physical.angular)
            ]
            data = dict(entry_vectors=vectors, mass=mass, linear=linear, angular=angular)
            wrench = np.zeros((physical.count, group["body_max"], 6), np.float32)
            reject = np.zeros(physical.count, np.int32)
            for world in group["worlds"]:
                row, spec = physical.rows[world], physical.specs[world]
                body_count = spec.b if group["floating"] else spec.n
                try:
                    loads = sample_tree_contacts(
                        world=world,
                        frames=entry[world]["frames"],
                        materials_by_pair=self._world_materials[world],
                        candidates=tuple(
                            pair for pair in result.candidates if pair[0].world == world
                        ),
                        group_count=body_count,
                        site_capacity=self._site_caps[world],
                        tree_dynamic_pairs=self._tree_dynamic_pairs,
                    )
                except ContactSiteCapacityExceeded as error:
                    reject[row] = 1
                    attempt["rejected"][world] = (error.count, error.limit)
                    continue
                attempt["loads"][world] = loads
                wrench[row, :body_count] = as_fp32(
                    loads.wrench_fp64, "contact COM wrench", (body_count, 6)
                )
            self._upload_contact(group["wrench"], wrench)
            self._upload_contact(group["reject"], reject)
            data["uploaded"] = wrench
            attempt["groups"][storage] = data
        return attempt

    def _stage_tree(self, storage, attempt, dt, src, dst):
        batch, group = self._batch, self._contact_groups[storage]
        physical = group["physical"]
        # Entry mechanics is already complete; do not erase contact with a rebuild.
        projection_args = (
            physical.count,
            physical.meta,
            physical.linear,
            physical.angular,
            group["wrench"],
            physical.vector,
            group["projected"],
        )
        if group["floating"]:
            batch._launch(self._floating_contact_kernels["project"][src], *projection_args)
            self._contact_counts["kernel_launches"] += 1
        else:
            self._launch_contact("project", src, *projection_args)
        if group["affine"]:
            storage.stage_solve_from_mechanics(dt, src, dst, batch._ticks[src])
        else:
            storage.stage_solve(dt, src, dst, batch._ticks[src])
        physical.stage_candidate_geometry(dst)
        self._launch_contact(
            "reject",
            src,
            physical.count,
            9 if group["affine"] else 7,
            group["reject"],
            physical.info,
        )
        if group["affine"]:
            storage.stage_finalize_and_retain(
                src, dst, batch._ticks[src], batch._ticks[dst], batch._status
            )
        else:
            storage.stage_finalize(src, dst, batch._ticks[src], batch._ticks[dst], batch._status)

    def _read_candidate(self, attempt, dst):
        if not self._contact_worlds:
            return {}
        candidate = dict(packet=self._read_packet(dst), groups={})
        for group in self._contact_groups.values():
            storage, physical = group["storage"], group["physical"]
            data = dict(
                vectors=self._read_contact(physical.vector),
                projected=self._read_contact(group["projected"]),
                info=self._read_contact(physical.info),
                diag=self._read_contact(physical.diag),
            )
            if group["affine"]:
                data["reduced_vectors"] = self._read_contact(storage.vector)
                selected = [w for w in group["worlds"] if storage._pending_info[0][w]]
                data["affine_diagnostics"] = storage._read_diagnostics(selected)
            elif group["floating"]:
                selected = [w for w in group["worlds"] if storage._pending_info[0][w]]
                entry_data = attempt["groups"][storage]
                arrays = {key: entry_data[key] for key in ("linear", "angular", "mass")}
                arrays.update(
                    vector=data["vectors"],
                    entry_pose=self._read_contact(storage.entry_pose),
                    info=data["info"],
                    diag=data["diag"],
                )
                data["floating_diagnostics"] = floating_contact_diagnostics(
                    storage, selected, arrays
                )
            candidate["groups"][storage] = data
        return candidate

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
            group = self._world_groups[world]
            storage, physical = group["storage"], group["physical"]
            row, spec = physical.rows[world], physical.specs[world]
            body_count = spec.b if group["floating"] else spec.n
            velocity_count = spec.d if group["floating"] else spec.n
            entry_data, final_data = attempt["groups"][storage], candidate["groups"][storage]
            vector = final_data["vectors"][row, :velocity_count]
            if group["affine"]:
                start = 6 if group["floating"] else 0
                drive_vector = final_data["reduced_vectors"][
                    row, start : start + storage.specs[world].k
                ]
            elif group["floating"]:
                drive_vector = vector[6:]
            else:
                drive_vector = vector
            wrench = entry_data["uploaded"][row, :body_count]
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
                mass=_owned(entry_data["mass"][row, :velocity_count, :velocity_count]),
                linear_jacobian=_owned(entry_data["linear"][row, :body_count, :, :velocity_count]),
                angular_jacobian=_owned(
                    entry_data["angular"][row, :body_count, :, :velocity_count]
                ),
                bias=_owned(entry_data["entry_vectors"][row, :velocity_count, 0]),
                total_external=_owned(vector[:, 1]),
                acceleration=_owned(vector[:, 3]),
                control_iterations=int(final_data["info"][row, 3]),
                control_residual=float(final_data["diag"][row, 1]),
                equation_residual=float(final_data["diag"][row, 2]),
            )
            if self._tree_dynamic_pairs:
                diagnostics["tree_dynamic_pairs"] = True
            if group["affine"] and group["floating"]:
                spec = storage.specs[world]
                diagnostics.update(
                    coordinate_kind="floating_affine",
                    full_dof_count=spec.d,
                    reduced_dof_count=spec.reduced_d,
                    independent_dof_count=spec.k,
                    joint_count=spec.n,
                    body_count=spec.b,
                    velocity_count=spec.d,
                    position_count=spec.q_count,
                    graph_token=spec.graph_token,
                    projection_matrix=spec.projection,
                    offset_vector=spec.offset,
                    constraint_rows=spec.constraints,
                    floating_affine_diagnostic=final_data["affine_diagnostics"][world],
                    equation_residual=final_data["affine_diagnostics"][world].equation_residual,
                )
            elif group["affine"]:
                spec = storage.specs[world]
                diagnostics.update(
                    coordinate_kind="affine",
                    full_dof_count=spec.n,
                    independent_dof_count=spec.k,
                    graph_token=spec.graph_token,
                    projection_matrix=spec.matrix,
                    offset_vector=spec.offset,
                    constraint_rows=spec.constraints,
                    affine_diagnostic=final_data["affine_diagnostics"][world],
                    equation_residual=final_data["affine_diagnostics"][world].equation_residual,
                )
            elif group["floating"]:
                diagnostics.update(
                    coordinate_kind="floating",
                    joint_count=spec.n,
                    body_count=spec.b,
                    velocity_count=spec.d,
                    position_count=spec.q_count,
                    floating_diagnostic=final_data["floating_diagnostics"][world],
                    equation_residual=final_data["floating_diagnostics"][world].equation_residual,
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
                entry_data["entry_vectors"][row, :velocity_count, 1],
                drive_vector[:, 11:14],
                loads.wrench_fp64,
                wrench,
                final_data["projected"][row, :velocity_count],
                drive_vector[:, 2],
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

    def _contact_stats(self, result):
        counts = self._contact_counts
        result["device_owned_bytes"] += counts["device_owned_bytes"]
        result["device_allocations"] += counts["device_allocations"]
        result.update({("contact_" + key): value for key, value in counts.items()})
        return result

    def _close_contact_mechanics(self):
        self._contact_resources.close()
        self._contact_counts["device_owned_bytes"] = 0
        self._contact_groups.clear()
        self._world_groups.clear()
        self._contact_kernels = None
        self._floating_contact_kernels = None


class ResidentContactPlan(_ResidentContactMechanics, ResidentCollisionPlan):
    """Immutable single-page registration plus temporary load/velocity storage."""

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
        tree_dynamic_pairs=False,
    ):
        if sampling != "discrete_entry":
            raise ValueError("resident contact supports only discrete_entry sampling, not CCD")
        if not isinstance(materials_by_pair, Mapping):
            raise TypeError("contact materials must be a complete pair mapping")
        bindings = tuple(bindings)
        if not all(
            type(b)
            in (
                ResidentTreeBinding,
                ResidentAffineTreeBinding,
                ResidentFloatingTreeBinding,
                ResidentFloatingAffineTreeBinding,
                ResidentStaticBinding,
            )
            for b in bindings
        ):
            raise ValueError(
                "contact binding requires a supported resident tree or finite static shape"
            )
        self._contact_bindings = MappingProxyType({b.key: b for b in bindings})
        self._materials = MappingProxyType(dict(materials_by_pair))
        self._site_config = site_capacity
        self._initialize_contact_mechanics()
        super().__init__(
            batch,
            model,
            bindings,
            candidate_capacity=candidate_capacity,
            tree_dynamic_pairs=tree_dynamic_pairs,
        )

    def _prepare_topology(self, slots):
        super()._prepare_topology(slots)
        self._prepare_contact_registration()

    def _prepare_device(self, view, slots, paths):
        super()._prepare_device(view, slots, paths)
        self._prepare_contact_device()

    def stats(self):
        return self._contact_stats(super().stats())

    def close(self):
        try:
            super().close()
        finally:
            self._close_contact_mechanics()
