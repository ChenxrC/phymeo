"""Floating banks sharing one existing TYX runtime, lock, active index and epoch.

Stage methods never publish or perform HOST mechanics. Every readback is explicit
and counted. Only the owning batch calls finish/commit after one successful swap.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..solvers.floating_tree import FloatingRobotInputs
from ._resident_floating_tree import (
    FloatingTreeDiagnostic,
    FloatingTreeInputs,
    FloatingTreeKinematics,
    FloatingTreeWorldSpec,
    floating_packet,
)
from ._resident_tree_storage import STATUS_NAMES
from ._resident_tree_world import _owned


@dataclass(frozen=True, eq=False)
class FloatingControlStage:
    owner: object
    drive: np.ndarray
    effort: np.ndarray
    loads: np.ndarray


@dataclass(frozen=True, eq=False)
class FloatingRestoreStage:
    owner: object
    mask: np.ndarray
    ticks: np.ndarray
    root: np.ndarray
    state: np.ndarray
    applied: np.ndarray
    controls: FloatingControlStage
    validity: Mapping
    status_codes: Mapping


@dataclass(frozen=True, eq=False)
class FloatingTreeGeometryView:
    owner: object
    topology_generation: int
    state_epoch: int
    active: int
    rows: Mapping
    specs: Mapping
    meta: object
    parents: object
    kinds: object
    anchor: object
    root: object
    state: object
    fk: object

    def write_bounds(
        self,
        launch,
        kernels,
        *,
        count,
        rows,
        groups,
        offsets,
        points,
        geometry,
        ticks,
        bounds,
        stamp,
    ):
        launch(
            kernels[self.active],
            count,
            rows,
            groups,
            offsets,
            points,
            geometry,
            self.meta,
            self.parents,
            self.kinds,
            self.anchor,
            self.root,
            self.state,
            self.fk,
            ticks,
            bounds,
            stamp,
        )


class FloatingTreeDeviceStorage:
    """B=n+1 bodies, D=n+6 velocities and n motors; no dummy program or root motor.

    meta int32[T,4] stores global slot,n,delay,budget. properties[T,B,18]
    stores offset3,axis3,COM3,I3,mass,kp,kd,cap,armature,passive damping; root has no gains. Root state
    [T,13], child state[T,max(1,n),7], generalized load[T,D], FK[T,B,18],
    COM/world twists[T,B,6] and energy[T] are all paired by one active index.
    """

    def __init__(self, batch, specs_by_world):
        from ._tyx_resident_floating_kernels import make_floating_tree_kernels

        if not isinstance(specs_by_world, Mapping) or not specs_by_world:
            raise ValueError("floating storage requires a nonempty compiled mapping")
        self.batch = batch
        self.specs = MappingProxyType(dict(specs_by_world))
        if not all(isinstance(s, FloatingTreeWorldSpec) for s in self.specs.values()):
            raise TypeError("floating storage needs FloatingTreeWorldSpec entries")
        self.worlds = tuple(self.specs)
        self.rows = MappingProxyType({w: i for i, w in enumerate(self.worlds)})
        self.count = len(self.worlds)
        self.nmax = max(s.n for s in self.specs.values())
        self.npad = max(1, self.nmax)
        self.bmax, self.dmax = self.nmax + 1, self.nmax + 6
        self._owner = object()
        self._closed = False
        self._dirty = False
        self._pending_info = None
        self._valid = {w: False for w in self.worlds}
        self.status_codes = MappingProxyType({w: "INITIALIZED" for w in self.worlds})
        self.kernels = make_floating_tree_kernels()
        for category in (
            "floating_topology",
            "floating_initial",
            "floating_control",
            "floating_restore",
            "floating_state",
            "floating_fk",
            "floating_summary",
            "floating_diagnostic",
            "floating_status",
        ):
            for suffix in ("uploaded_bytes", "downloaded_bytes", "host_read_bytes"):
                batch._counts.setdefault(category + "_" + suffix, 0)
        for name in (
            "dynamics_kernel_launches",
            "fk_kernel_launches",
            "energy_kernel_launches",
            "restore_kernel_launches",
            "commit_kernel_launches",
            "diagnostic_reads",
            "state_reads",
            "fk_reads",
            "summary_reads",
            "status_reads",
        ):
            batch._counts.setdefault("floating_" + name, 0)
        t, n, b, d = self.count, self.npad, self.bmax, self.dmax
        meta = np.zeros((t, 4), np.int32)
        parents = np.full((t, b), -1, np.int32)
        kinds = np.full((t, b), 2, np.int32)
        properties, zero, principal = (
            np.zeros((t, b, 18), np.float32),
            np.zeros((t, b, 9), np.float32),
            np.zeros((t, b, 9), np.float32),
        )
        gravity, anchor = np.zeros((t, 3), np.float32), np.zeros((t, 3), np.float32)
        roots, states = np.zeros((t, 13), np.float32), np.zeros((t, n, 7), np.float32)
        applied = np.zeros((t, d), np.float32)
        drive, effort, loads = (
            np.zeros((t, n, 3), np.float32),
            np.zeros((t, n), np.float32),
            np.zeros((t, b, 6), np.float32),
        )
        for w, spec in self.specs.items():
            r, size = self.rows[w], spec.b
            meta[r] = (batch._slots[w], spec.n, spec.delay, spec.control_max_iterations)
            parents[r, :size], kinds[r, :size] = spec.parents, spec.kinds
            for start, values in (
                (0, spec.offsets),
                (3, spec.local_axis),
                (6, spec.com),
                (9, spec.inertia),
            ):
                properties[r, :size, start : start + 3] = values
            properties[r, :size, 12] = spec.mass
            properties[r, 1:size, 16] = spec.joint_armature
            properties[r, 1:size, 17] = spec.joint_damping
            for column, values in ((13, spec.kp), (14, spec.kd), (15, spec.cap)):
                properties[r, 1:size, column] = values
            zero[r, :size] = spec.zero_rotation.reshape(size, 9)
            principal[r, :size] = spec.inertia_frame.reshape(size, 9)
            gravity[r], anchor[r] = spec.gravity, spec.anchor
            roots[r], states[r, : max(1, spec.n)] = spec.initial_root, spec.initial_state
            applied[r, : spec.d] = spec.initial_generalized
            drive[r, : spec.n], effort[r, : spec.n], loads[r, :size] = (
                spec.initial_drive,
                spec.initial_effort,
                spec.initial_loads,
            )
        self._held_drive, self._held_effort, self._held_loads = (
            _owned(drive),
            _owned(effort),
            _owned(loads),
        )
        self._meta_host = _owned(meta, np.int32)
        with batch._device():

            def put(a, category="floating_topology"):
                return batch._buffer(a.shape, a.dtype, a, category)

            def blank(shape, dtype=np.float32):
                return batch._buffer(shape, dtype, np.zeros(shape, dtype), "floating_initial")

            self.meta, self.parents, self.kinds = put(meta), put(parents), put(kinds)
            self.properties, self.zero, self.principal = put(properties), put(zero), put(principal)
            gains = np.zeros((t, n, 4), np.float32)
            gains[:, : self.nmax, :3] = properties[:, 1:b, 13:16]
            gains[:, : self.nmax, 3] = properties[:, 1:b, 17]
            self.motor_gains = put(gains)
            self.physical_root_com = put(properties[:, 0, 6:9].copy())
            self.gravity, self.anchor = put(gravity), put(anchor)
            self.commands = put(drive, "floating_initial")
            self.effort = put(effort, "floating_initial")
            self.loads = put(loads, "floating_initial")
            self.roots = (put(roots, "floating_initial"), put(roots, "floating_initial"))
            self.states = (put(states, "floating_initial"), put(states, "floating_initial"))
            self.applied = (put(applied, "floating_initial"), put(applied, "floating_initial"))
            self.fk = (blank((t, b, 18)), blank((t, b, 18)))
            self.twist = (blank((t, b, 6)), blank((t, b, 6)))
            self.energy = (blank((t,)), blank((t,)))
            self.frame, self.origins = blank((t, b, 9)), blank((t, b, 3))
            self.flags = blank((t,), np.int32)
            self.linear, self.angular = blank((t, b, 3, d)), blank((t, b, 3, d))
            self.spatial, self.inertial = blank((t, b, 18)), blank((t, b, 9))
            self.mass, self.factor, self.mobility = (
                blank((t, d, d)),
                blank((t, d, d)),
                blank((t, d, d)),
            )
            self.vector, self.entry_pose = blank((t, d, 14)), blank((t, 7))
            self.info, self.diag = blank((t, 4), np.int32), blank((t, 3))
            self.restore_mask, self.restore_ticks = blank((t,), np.int32), blank((t,), np.int32)
            self.restore_root, self.restore_state, self.restore_applied = (
                blank((t, 13)),
                blank((t, n, 7)),
                blank((t, d)),
            )
            for bank in (0, 1):
                self._geometry(bank)
                self._energy(bank)
            batch._sync()
            self.check_restore()
        batch._resources.callback(self.close)

    def _check(self):
        if self._closed:
            raise RuntimeError("floating storage is closed")
        self.batch._check_open()

    def _read(self, value, category):
        result = self.batch._read(value, category)
        if self.batch._mode == "interpret":
            self.batch._counts[category + "_host_read_bytes"] += result.nbytes
        return result

    def _geometry(self, bank):
        self.batch._launch(
            self.kernels["geometry"][bank],
            self.count,
            self.meta,
            self.parents,
            self.kinds,
            self.properties,
            self.zero,
            self.roots[bank],
            self.states[bank],
            self.fk[bank],
            self.frame,
            self.flags,
        )
        self.batch._counts["floating_fk_kernel_launches"] += 1

    def _energy(self, bank):
        self.batch._launch(
            self.kernels["energy"][bank],
            self.count,
            self.meta,
            self.parents,
            self.kinds,
            self.properties,
            self.principal,
            self.roots[bank],
            self.states[bank],
            self.fk[bank],
            self.origins,
            self.twist[bank],
            self.energy[bank],
            self.flags,
        )
        self.batch._counts["floating_energy_kernel_launches"] += 1

    def validate_controls(self, by_world):
        self._check()
        if not isinstance(by_world, Mapping):
            raise TypeError("floating controls must be a mapping")
        drive, effort, loads = (
            self._held_drive.copy(),
            self._held_effort.copy(),
            self._held_loads.copy(),
        )
        for world, packet in by_world.items():
            if world not in self.rows:
                raise KeyError(world)
            row, spec = self.rows[world], self.specs[world]
            a, b, c = floating_packet(packet, spec.n)
            drive[row, : spec.n], effort[row, : spec.n], loads[row, : spec.b] = a, b, c
        return FloatingControlStage(self._owner, _owned(drive), _owned(effort), _owned(loads))

    def commit_controls(self, stage):
        if not isinstance(stage, FloatingControlStage) or stage.owner is not self._owner:
            raise ValueError("floating controls belong to another storage")
        changed = any(
            not np.array_equal(a, b)
            for a, b in zip(
                (self._held_drive, self._held_effort, self._held_loads),
                (stage.drive, stage.effort, stage.loads),
                strict=True,
            )
        )
        self._held_drive, self._held_effort, self._held_loads = (
            stage.drive,
            stage.effort,
            stage.loads,
        )
        self._dirty = self._dirty or changed

    def upload_controls(self):
        if self._dirty:
            self.batch._upload(self.commands, self._held_drive, "floating_control")
            self.batch._upload(self.effort, self._held_effort, "floating_control")
            self.batch._upload(self.loads, self._held_loads, "floating_control")

    def controls_uploaded(self):
        self._dirty = False

    def invalidate_diagnostics(self):
        self._valid = {w: False for w in self.worlds}
        self._pending_info = None

    def stage_mechanics(self, src):
        self._check()
        self.batch._launch(
            self.kernels["mechanics"][src],
            self.count,
            self.meta,
            self.parents,
            self.kinds,
            self.properties,
            self.principal,
            self.gravity,
            self.effort,
            self.loads,
            self.roots[src],
            self.states[src],
            self.fk[src],
            self.linear,
            self.angular,
            self.spatial,
            self.inertial,
            self.mass,
            self.vector,
            self.entry_pose,
        )
        self.batch._counts["floating_dynamics_kernel_launches"] += 1

    def stage_solve(self, h, src, dst, tick_src):
        self._check()
        self.batch._launch(
            self.kernels["solve"][src],
            self.count,
            h,
            self.meta,
            self.motor_gains,
            self.physical_root_com,
            self.commands,
            self.roots[src],
            self.states[src],
            self.fk[src],
            self.applied[src],
            tick_src,
            self.roots[dst],
            self.states[dst],
            self.applied[dst],
            self.mass,
            self.factor,
            self.mobility,
            self.vector,
            self.info,
            self.diag,
        )
        self.batch._counts["floating_dynamics_kernel_launches"] += 1

    def stage_candidate_geometry(self, dst):
        self._geometry(dst)
        self._energy(dst)

    def stage_finalize(self, src, dst, tick_src, tick_dst, accepted):
        self.batch._launch(
            self.kernels["finalize"][src],
            self.count,
            self.meta,
            self.roots[src],
            self.roots[dst],
            self.states[src],
            self.states[dst],
            self.applied[src],
            self.applied[dst],
            self.fk[src],
            self.fk[dst],
            self.twist[src],
            self.twist[dst],
            self.energy[src],
            self.energy[dst],
            tick_src,
            tick_dst,
            accepted,
            self.info,
            self.flags,
        )
        self.batch._counts["floating_commit_kernel_launches"] += 1

    def stage_advance(self, h, src, dst, tick_src, tick_dst, accepted):
        self._check()
        self.invalidate_diagnostics()
        self.stage_mechanics(src)
        self.stage_solve(h, src, dst, tick_src)
        self.stage_candidate_geometry(dst)
        self.stage_finalize(src, dst, tick_src, tick_dst, accepted)

    def check_advance(self, status):
        self.batch._counts["floating_status_reads"] += 1
        info = self._read(self.info, "floating_status")
        valid = {}
        names = {}
        for world, row in self.rows.items():
            slot = self.batch._slots[world]
            accepted = bool(status[world]) if isinstance(status, Mapping) else bool(status[slot])
            code = int(info[row, 2])
            if code not in STATUS_NAMES or accepted != (code == 0):
                raise RuntimeError("floating status disagrees with batch acceptance")
            valid[world], names[world] = accepted, STATUS_NAMES[code]
        self._pending_info = (valid, MappingProxyType(names))

    def finish_advance(self, status):
        self._valid, self.status_codes = self._pending_info
        self._pending_info = None

    def validate_restore(self, by_world):
        self._check()
        root = np.zeros((self.count, 13), np.float32)
        state = np.zeros((self.count, self.npad, 7), np.float32)
        applied = np.zeros((self.count, self.dmax), np.float32)
        masks, ticks = np.zeros(self.count, np.int32), np.zeros(self.count, np.int32)
        packets = {}
        for world, checkpoint in by_world.items():
            if world not in self.rows:
                raise KeyError(world)
            row, spec = self.rows[world], self.specs[world]
            r, s, a, packet, tick = spec.restore_row(checkpoint, world)
            root[row], state[row, : max(1, spec.n)], applied[row, : spec.d] = r, s, a
            masks[row], ticks[row], packets[world] = 1, tick, packet
        valid = MappingProxyType(
            {w: False if masks[r] else self._valid[w] for w, r in self.rows.items()}
        )
        names = MappingProxyType(
            {w: "RESTORED" if masks[r] else self.status_codes[w] for w, r in self.rows.items()}
        )
        return FloatingRestoreStage(
            self._owner,
            _owned(masks, np.int32),
            _owned(ticks, np.int32),
            _owned(root),
            _owned(state),
            _owned(applied),
            self.validate_controls(packets),
            valid,
            names,
        )

    def stage_restore(self, stage, src, dst, tick_src, tick_dst):
        if not isinstance(stage, FloatingRestoreStage) or stage.owner is not self._owner:
            raise ValueError("restore stage belongs to another floating storage")
        for buffer, value in (
            (self.restore_mask, stage.mask),
            (self.restore_root, stage.root),
            (self.restore_state, stage.state),
            (self.restore_applied, stage.applied),
            (self.restore_ticks, stage.ticks),
        ):
            self.batch._upload(buffer, value, "floating_restore")
        self.batch._launch(
            self.kernels["restore"][src],
            self.count,
            self.meta,
            self.restore_mask,
            self.restore_root,
            self.restore_state,
            self.restore_applied,
            self.restore_ticks,
            self.roots[src],
            self.roots[dst],
            self.states[src],
            self.states[dst],
            self.applied[src],
            self.applied[dst],
            tick_src,
            tick_dst,
        )
        self.batch._counts["floating_restore_kernel_launches"] += 1
        self.stage_candidate_geometry(dst)

    def check_restore(self):
        self.batch._counts["floating_status_reads"] += 1
        flags = self._read(self.flags, "floating_status")
        if flags.shape != (self.count,) or np.any(flags != 1):
            raise ValueError("floating restored/initial FK, twist or kinetic energy is invalid")

    def commit_restore(self, stage):
        self.commit_controls(stage.controls)
        self._valid, self.status_codes = dict(stage.validity), stage.status_codes
        self._pending_info = None

    def _packet(self, world):
        row, n = self.rows[world], self.specs[world].n
        return FloatingTreeInputs(
            self._held_drive[row, :n],
            FloatingRobotInputs(
                np.asarray(self._held_effort[row, :n], dtype=float),
                np.asarray(self._held_loads[row, : n + 1, :3], dtype=float),
                np.asarray(self._held_loads[row, : n + 1, 3:], dtype=float),
            ),
        )

    def snapshot(self, keys, active, ticks):
        self._check()
        if not keys:
            return {}
        with self.batch._device():
            self.batch._sync()
            root, state, applied, time = [
                self._read(a, "floating_state")
                for a in (self.roots[active], self.states[active], self.applied[active], ticks)
            ]
        self.batch._counts["floating_state_reads"] += 1
        return {
            w: self.specs[w].checkpoint(
                w,
                root[self.rows[w]],
                state[self.rows[w]],
                applied[self.rows[w]],
                time[self.batch._slots[w]],
                self._packet(w),
            )
            for w in keys
        }

    def kinematics(self, keys, active, ticks):
        self._check()
        if not keys:
            return {}
        with self.batch._device():
            self.batch._sync()
            root, state, fk, twist, time = [
                self._read(a, "floating_fk")
                for a in (
                    self.roots[active],
                    self.states[active],
                    self.fk[active],
                    self.twist[active],
                    ticks,
                )
            ]
        self.batch._counts["floating_fk_reads"] += 1
        out = {}
        for world in keys:
            row, spec = self.rows[world], self.specs[world]
            body = fk[row, : spec.b]
            position = np.r_[
                spec.anchor + root[row, :3].astype(float), root[row, 3:7], state[row, : spec.n, 0]
            ]
            velocity = np.r_[root[row, 7:13], state[row, : spec.n, 1]]
            out[world] = FloatingTreeKinematics(
                int(time[self.batch._slots[world]]),
                spec.joint_types,
                position,
                velocity,
                body[:, :3].astype(float) + spec.anchor,
                body[:, 3:12].reshape(spec.b, 3, 3),
                body[:, 15:18].astype(float) + spec.anchor,
                twist[row, : spec.b, :3],
                twist[row, : spec.b, 3:],
                spec.n,
                spec.b,
                spec.d,
                spec.q_count,
            )
        return out

    def diagnostics(self, keys):
        self._check()
        out = {w: None for w in keys}
        valid = [w for w in keys if self._valid[w]]
        if not valid:
            return out
        with self.batch._device():
            self.batch._sync()
            linear, angular, mass, vector, pose, info, diag = [
                self._read(a, "floating_diagnostic")
                for a in (
                    self.linear,
                    self.angular,
                    self.mass,
                    self.vector,
                    self.entry_pose,
                    self.info,
                    self.diag,
                )
            ]
        self.batch._counts["floating_diagnostic_reads"] += 1
        for world in valid:
            row, spec = self.rows[world], self.specs[world]
            v = vector[row, : spec.d]
            q = np.r_[spec.anchor + pose[row, :3].astype(float), pose[row, 3:7], v[6:, 4]]
            out[world] = FloatingTreeDiagnostic(
                int(info[row, 0]),
                int(info[row, 1]),
                float(diag[row, 0]),
                q,
                v[:, 5],
                linear[row, : spec.b, :, : spec.d],
                angular[row, : spec.b, :, : spec.d],
                mass[row, : spec.d, : spec.d],
                v[:, 0],
                v[:, 1],
                v[6:, 2],
                v[:, 3],
                int(info[row, 3]),
                float(diag[row, 1]),
                float(diag[row, 2]),
            )
        return out

    def summary(self, keys, active):
        self._check()
        if not keys:
            return {}
        with self.batch._device():
            self.batch._sync()
            energy = self._read(self.energy[active], "floating_summary")
        if not np.isfinite(energy).all():
            raise FloatingPointError("accepted floating kinetic energy is not finite")
        self.batch._counts["floating_summary_reads"] += 1
        return {w: {"kinetic_energy_j": float(energy[self.rows[w]])} for w in keys}

    def borrow_geometry(self, active):
        self._check()
        return FloatingTreeGeometryView(
            self.batch._collision_owner,
            self.batch._topology_generation,
            self.batch._state_epoch,
            active,
            self.rows,
            self.specs,
            self.meta,
            self.parents,
            self.kinds,
            self.anchor,
            self.roots[active],
            self.states[active],
            self.fk[active],
        )

    def close(self):
        # Actual buffers belong to the parent's existing resource stack.
        self._closed = True
