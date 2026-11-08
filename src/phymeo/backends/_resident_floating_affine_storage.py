"""Full floating physics and reduced affine banks in one batch transaction.

This wrapper shares the original runtime, global slots, active swap and epoch.
It owns only coordinate/drive banks and their diagnostics. No HOST dynamics,
extra root integration or independent publication occurs here.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..solvers.floating_tree import FloatingRobotInputs
from ._resident_affine_storage import AFFINE_STATUS_NAMES
from ._resident_floating_affine_tree import (
    FloatingAffineTreeDiagnostic,
    FloatingAffineTreeInputs,
    FloatingAffineTreeKinematics,
    FloatingAffineTreeWorldSpec,
)
from ._resident_floating_storage import FloatingTreeDeviceStorage
from ._resident_floating_tree import FloatingTreeInputs
from ._resident_tree_world import _owned


@dataclass(frozen=True, eq=False)
class FloatingAffineControlStage:
    owner: object
    drive: np.ndarray
    effort: np.ndarray
    loads: np.ndarray
    physical: object


@dataclass(frozen=True, eq=False)
class FloatingAffineRestoreStage:
    owner: object
    mask: np.ndarray
    ticks: np.ndarray
    state: np.ndarray
    applied: np.ndarray
    controls: FloatingAffineControlStage
    physical: object
    validity: Mapping
    status_codes: Mapping


class FloatingAffineTreeDeviceStorage:
    """Compose a full C9 physical tree and independent root+K algebraic coordinates.

    The real root[13] banks live only in physical. This wrapper has K-by-7
    scalar state and (6+K) applied-load banks. Physical metadata always contains
    n; solve metadata contains K but retains the same global slot/delay/budget.
    """

    is_floating = True
    is_affine = True

    def __init__(self, batch, specs_by_world):
        from ._tyx_resident_floating_affine import make_floating_affine_kernels

        if not isinstance(specs_by_world, Mapping) or not specs_by_world:
            raise ValueError("floating affine storage needs a nonempty compiled mapping")
        self.batch = batch
        self.specs = MappingProxyType(dict(specs_by_world))
        if not all(isinstance(s, FloatingAffineTreeWorldSpec) for s in self.specs.values()):
            raise TypeError("floating affine storage needs FloatingAffineTreeWorldSpec entries")
        self.worlds = tuple(self.specs)
        self.rows = MappingProxyType({w: i for i, w in enumerate(self.worlds)})
        self.count = len(self.worlds)
        self.nmax = max(s.n for s in self.specs.values())
        self.kmax = max(s.k for s in self.specs.values())
        self.rmax = max(s.r for s in self.specs.values())
        self.bmax, self.dmax, self.reduced_dmax = self.nmax + 1, self.nmax + 6, self.kmax + 6
        self._owner, self._closed, self._dirty = object(), False, False
        self._pending_info = None
        self._valid = {w: False for w in self.worlds}
        self.status_codes = MappingProxyType({w: "INITIALIZED" for w in self.worlds})
        for category in (
            "floating_affine_topology", "floating_affine_initial", "floating_affine_control",
            "floating_affine_restore", "floating_affine_state", "floating_affine_fk",
            "floating_affine_diagnostic", "floating_affine_status",
        ):
            for suffix in ("uploaded_bytes", "downloaded_bytes", "host_read_bytes"):
                batch._counts.setdefault(category + "_" + suffix, 0)
        for name in (
            "projection_kernel_launches", "solve_kernel_launches", "expand_kernel_launches",
            "commit_kernel_launches", "restore_kernel_launches", "validation_kernel_launches",
            "state_reads", "fk_reads", "diagnostic_reads", "status_reads",
        ):
            batch._counts.setdefault("floating_affine_" + name, 0)
        self.physical = FloatingTreeDeviceStorage(batch, {w: s.physical for w, s in self.specs.items()})
        if tuple(self.physical.rows.items()) != tuple(self.rows.items()):
            raise RuntimeError("floating affine and physical storage row order differs")
        self.kernels = make_floating_affine_kernels()
        t, n, k, r = self.count, self.nmax, self.kmax, self.rmax
        b, df, dr = self.bmax, self.dmax, self.reduced_dmax
        meta = np.zeros((t, 4), np.int32)
        matrix, projection, offset = np.zeros((t, n, k), np.float32), np.zeros((t, df, dr), np.float32), np.zeros((t, n), np.float32)
        independent = np.zeros((t, k), np.int32)
        indices, reverse = np.zeros((t, r, 2), np.int32), np.zeros((t, r), np.int32)
        values = np.zeros((t, r, 2), np.float32)
        full_limits, limits = np.zeros((t, n, 3), np.float32), np.zeros((t, k, 3), np.float32)
        gains, root_com = np.zeros((t, k, 3), np.float32), np.zeros((t, 3), np.float32)
        state, applied = np.zeros((t, k, 7), np.float32), np.zeros((t, dr), np.float32)
        drive, effort, loads = np.zeros((t, k, 3), np.float32), np.zeros((t, n), np.float32), np.zeros((t, b, 6), np.float32)
        for world, spec in self.specs.items():
            row = self.rows[world]
            meta[row] = (batch._slots[world], spec.k, spec.delay, spec.control_max_iterations)
            matrix[row, :spec.n, :spec.k], offset[row, :spec.n] = spec.matrix, spec.offset
            projection[row, :spec.d, :spec.reduced_d] = spec.projection
            independent[row, :spec.k] = spec.independent
            indices[row, :spec.r], values[row, :spec.r], reverse[row, :spec.r] = spec.row_indices, spec.row_values, spec.reverse_rows
            full_limits[row, :spec.n], limits[row, :spec.k] = spec.full_limits, spec.limits
            gains[row, :spec.k, 0], gains[row, :spec.k, 1], gains[row, :spec.k, 2] = spec.kp, spec.kd, spec.cap
            root_com[row] = spec.physical.com[0]
            state[row, :spec.k], applied[row, :spec.reduced_d] = spec.initial_state, spec.initial_generalized
            drive[row, :spec.k], effort[row, :spec.n], loads[row, :spec.b] = spec.initial_drive, spec.initial_effort, spec.initial_loads
        self._held_drive, self._held_effort, self._held_loads = _owned(drive), _owned(effort), _owned(loads)
        self._meta_host = _owned(meta, np.int32)
        with batch._device():
            def put(value, category="floating_affine_topology"):
                return batch._buffer(value.shape, value.dtype, value, category)

            def blank(shape, dtype=np.float32):
                return put(np.zeros(shape, dtype), "floating_affine_initial")

            self.meta, self.matrix, self.projection, self.offset = put(meta), put(matrix), put(projection), put(offset)
            self.independent = put(independent)
            self.row_indices, self.row_values, self.reverse_rows = put(indices), put(values), put(reverse)
            self.full_limits, self.limits = put(full_limits), put(limits)
            self.motor_gains, self.physical_root_com = put(gains), put(root_com)
            self.commands = put(drive, "floating_affine_initial")
            self.states = (put(state, "floating_affine_initial"), put(state, "floating_affine_initial"))
            self.applied = (put(applied, "floating_affine_initial"), put(applied, "floating_affine_initial"))
            self.mass, self.factor, self.mobility = blank((t, dr, dr)), blank((t, dr, dr)), blank((t, dr, dr))
            self.vector = blank((t, dr, 14))
            self.linear, self.angular = blank((t, b, 3, dr)), blank((t, b, 3, dr))
            self.reaction, self.row_receipt = blank((t, df, 3)), blank((t, r, 2))
            self.metrics = blank((t, 4))
            self.projection_flags, self.restore_flags = blank((t,), np.int32), blank((t,), np.int32)
            self.restore_state, self.restore_applied = blank((t, k, 7)), blank((t, dr))
            for bank in (0, 1):
                self._validate_state(bank)
                batch._sync()
                self.check_restore()
        batch._resources.callback(self.close)

    def _check(self):
        if self._closed:
            raise RuntimeError("floating affine storage is closed")
        self.batch._check_open()

    def _read(self, value, category):
        result = self.batch._read(value, category)
        if self.batch._mode == "interpret":
            self.batch._counts[category + "_host_read_bytes"] += result.nbytes
        return result

    def _control_stage(self, drive, effort, loads):
        packets = {}
        for world, spec in self.specs.items():
            row = self.rows[world]
            packets[world] = FloatingTreeInputs(
                np.zeros((spec.n, 3)),
                FloatingRobotInputs(
                    np.asarray(effort[row, :spec.n], dtype=float),
                    np.asarray(loads[row, :spec.b, :3], dtype=float),
                    np.asarray(loads[row, :spec.b, 3:], dtype=float),
                ),
            )
        return FloatingAffineControlStage(
            self._owner, _owned(drive), _owned(effort), _owned(loads), self.physical.validate_controls(packets),
        )

    def validate_controls(self, by_world):
        self._check()
        if not isinstance(by_world, Mapping):
            raise TypeError("floating affine controls must be a mapping")
        drive, effort, loads = self._held_drive.copy(), self._held_effort.copy(), self._held_loads.copy()
        for world, packet in by_world.items():
            if world not in self.rows:
                raise KeyError(world)
            row, spec = self.rows[world], self.specs[world]
            a, b, c = spec.packet(packet)
            drive[row, :spec.k], effort[row, :spec.n], loads[row, :spec.b] = a, b, c
        return self._control_stage(drive, effort, loads)

    def commit_controls(self, stage):
        if not isinstance(stage, FloatingAffineControlStage) or stage.owner is not self._owner:
            raise ValueError("floating affine control stage belongs to another storage")
        if stage.drive.shape != self._held_drive.shape or stage.effort.shape != self._held_effort.shape or stage.loads.shape != self._held_loads.shape:
            raise ValueError("floating affine held-control dimensions changed")
        self.physical.commit_controls(stage.physical)
        self._dirty = self._dirty or not np.array_equal(stage.drive, self._held_drive)
        self._held_drive, self._held_effort, self._held_loads = stage.drive, stage.effort, stage.loads

    def upload_controls(self):
        self.physical.upload_controls()
        if self._dirty:
            self.batch._upload(self.commands, self._held_drive, "floating_affine_control")

    def controls_uploaded(self):
        self.physical.controls_uploaded()
        self._dirty = False

    def _packet(self, world):
        row, spec = self.rows[world], self.specs[world]
        return FloatingAffineTreeInputs(
            spec.graph_token, self._held_drive[row, :spec.k],
            FloatingRobotInputs(
                np.asarray(self._held_effort[row, :spec.n], dtype=float),
                np.asarray(self._held_loads[row, :spec.b, :3], dtype=float),
                np.asarray(self._held_loads[row, :spec.b, 3:], dtype=float),
            ),
        )

    def invalidate_diagnostics(self):
        self.physical.invalidate_diagnostics()
        self._valid = {w: False for w in self.worlds}
        self._pending_info = None

    def stage_mechanics(self, src):
        self._check()
        self.physical.stage_mechanics(src)

    def stage_solve_from_mechanics(self, h, src, dst, tick_src):
        """Project complete current loads once, then use the one shared root integrator."""
        self._check()
        p, launch = self.physical, self.batch._launch
        launch(
            self.kernels["project"][src], self.count, p.meta, self.meta, self.projection,
            self.states[src], p.mass, p.vector, p.linear, p.angular,
            self.mass, self.vector, self.linear, self.angular, self.projection_flags,
        )
        self.batch._counts["floating_affine_projection_kernel_launches"] += 1
        launch(
            p.kernels["solve"][src], self.count, h, self.meta, self.motor_gains,
            self.physical_root_com, self.commands, p.roots[src], self.states[src], p.fk[src],
            self.applied[src], tick_src, p.roots[dst], self.states[dst], self.applied[dst],
            self.mass, self.factor, self.mobility, self.vector, p.info, p.diag,
        )
        self.batch._counts["floating_affine_solve_kernel_launches"] += 1
        launch(
            self.kernels["expand"][src], self.count, p.meta, self.meta, self.matrix,
            self.projection, self.offset, self.independent, self.row_indices, self.row_values,
            self.reverse_rows, self.full_limits, self.limits, p.states[src], p.states[dst],
            self.states[dst], p.applied[src], p.applied[dst], p.mass, p.vector, self.vector,
            self.reaction, self.row_receipt, p.info, self.metrics, self.projection_flags,
        )
        self.batch._counts["floating_affine_expand_kernel_launches"] += 1

    def stage_candidate_geometry(self, dst):
        self._check()
        self.physical.stage_candidate_geometry(dst)

    def stage_finalize_and_retain(self, src, dst, tick_src, tick_dst, accepted):
        self._check()
        self.physical.stage_finalize(src, dst, tick_src, tick_dst, accepted)
        self.batch._launch(
            self.kernels["retain"][src], self.count, self.meta, self.states[src], self.states[dst],
            self.applied[src], self.applied[dst], accepted,
        )
        self.batch._counts["floating_affine_commit_kernel_launches"] += 1

    def stage_advance(self, h, src, dst, tick_src, tick_dst, accepted):
        self._check()
        self.invalidate_diagnostics()
        self.stage_mechanics(src)
        self.stage_solve_from_mechanics(h, src, dst, tick_src)
        self.stage_candidate_geometry(dst)
        self.stage_finalize_and_retain(src, dst, tick_src, tick_dst, accepted)

    def check_advance(self, status):
        self.batch._counts["floating_affine_status_reads"] += 1
        info = self._read(self.physical.info, "floating_affine_status")
        if info.shape != (self.count, 4):
            raise RuntimeError("floating affine status shape changed")
        valid, names = {}, {}
        for world, row in self.rows.items():
            accepted = bool(status[world]) if isinstance(status, Mapping) else bool(status[self.batch._slots[world]])
            code = int(info[row, 2])
            if code not in AFFINE_STATUS_NAMES or accepted != (code == 0):
                raise RuntimeError("floating affine status disagrees with full-world acceptance")
            valid[world], names[world] = accepted, AFFINE_STATUS_NAMES[code]
        self._pending_info = valid, MappingProxyType(names)

    def finish_advance(self, status):
        if self._pending_info is None:
            raise RuntimeError("floating affine status was not checked before publication")
        self._valid, self.status_codes = self._pending_info
        self._pending_info = None

    def validate_restore(self, by_world):
        self._check()
        if not isinstance(by_world, Mapping):
            raise TypeError("floating affine checkpoints must be a mapping")
        state, applied = np.zeros((self.count, self.kmax, 7), np.float32), np.zeros((self.count, self.reduced_dmax), np.float32)
        drive, effort, loads = self._held_drive.copy(), self._held_effort.copy(), self._held_loads.copy()
        physical_points = {}
        for world, checkpoint in by_world.items():
            if world not in self.rows:
                raise KeyError(world)
            row, spec = self.rows[world], self.specs[world]
            root, reduced, full, reduced_applied, full_applied, packet, tick = spec.restore_row(checkpoint, world)
            a, b, c = spec.packet(packet)
            state[row, :spec.k], applied[row, :spec.reduced_d] = reduced, reduced_applied
            drive[row, :spec.k], effort[row, :spec.n], loads[row, :spec.b] = a, b, c
            full_packet = FloatingTreeInputs(np.zeros((spec.n, 3)), packet.external)
            physical_points[world] = spec.physical.checkpoint(world, root, full, full_applied, tick, full_packet)
        physical = self.physical.validate_restore(physical_points)
        valid = MappingProxyType({w: False if physical.mask[r] else self._valid[w] for w, r in self.rows.items()})
        names = MappingProxyType({w: "RESTORED" if physical.mask[r] else self.status_codes[w] for w, r in self.rows.items()})
        return FloatingAffineRestoreStage(
            self._owner, physical.mask, physical.ticks, _owned(state), _owned(applied),
            self._control_stage(drive, effort, loads), physical, valid, names,
        )

    def _validate_state(self, bank):
        p = self.physical
        self.batch._launch(
            self.kernels["validate_state"][bank], self.count, p.meta, self.meta, self.matrix,
            self.projection, self.offset, self.independent, self.row_indices, self.row_values,
            self.full_limits, self.limits, p.states[bank], self.states[bank], p.applied[bank],
            self.applied[bank], self.restore_flags,
        )
        self.batch._counts["floating_affine_validation_kernel_launches"] += 1

    def stage_restore(self, stage, src, dst, tick_src, tick_dst):
        self._check()
        if not isinstance(stage, FloatingAffineRestoreStage) or stage.owner is not self._owner:
            raise ValueError("floating affine restore stage belongs to another storage")
        p = self.physical
        p.stage_restore(stage.physical, src, dst, tick_src, tick_dst)
        self.batch._upload(self.restore_state, stage.state, "floating_affine_restore")
        self.batch._upload(self.restore_applied, stage.applied, "floating_affine_restore")
        self.batch._launch(
            self.kernels["restore"][src], self.count, self.meta, p.restore_mask,
            self.restore_state, self.restore_applied, self.states[src], self.states[dst],
            self.applied[src], self.applied[dst],
        )
        self.batch._counts["floating_affine_restore_kernel_launches"] += 1
        self._validate_state(dst)

    def check_restore(self):
        self.physical.check_restore()
        self.batch._counts["floating_affine_status_reads"] += 1
        flags = self._read(self.restore_flags, "floating_affine_status")
        if flags.shape != (self.count,) or np.any(flags != 1):
            raise ValueError("floating affine restored/initial full and independent states disagree")

    def commit_restore(self, stage):
        self.commit_controls(stage.controls)
        self.physical.commit_restore(stage.physical)
        self._valid, self.status_codes = dict(stage.validity), stage.status_codes
        self._pending_info = None

    def snapshot(self, keys, active, ticks):
        self._check()
        if not keys:
            return {}
        p = self.physical
        with self.batch._device():
            self.batch._sync()
            root, full, full_applied, reduced, applied, time = [
                self._read(a, "floating_affine_state") for a in (
                    p.roots[active], p.states[active], p.applied[active], self.states[active],
                    self.applied[active], ticks,
                )
            ]
        self.batch._counts["floating_affine_state_reads"] += 1
        return {
            w: self.specs[w].checkpoint(
                w, root[self.rows[w]], reduced[self.rows[w]], full[self.rows[w]],
                applied[self.rows[w]], full_applied[self.rows[w]], time[self.batch._slots[w]], self._packet(w),
            ) for w in keys
        }

    def kinematics(self, keys, active, ticks):
        self._check()
        if not keys:
            return {}
        full = self.physical.kinematics(keys, active, ticks)
        with self.batch._device():
            self.batch._sync()
            reduced = self._read(self.states[active], "floating_affine_fk")
        self.batch._counts["floating_affine_fk_reads"] += 1
        result = {}
        for world in keys:
            spec, row, geometry = self.specs[world], self.rows[world], full[world]
            result[world] = FloatingAffineTreeKinematics(
                geometry.tick, geometry.joint_types, geometry.position, geometry.velocity,
                geometry.link_position_m, geometry.link_rotation, geometry.com_position_m,
                geometry.linear_velocity_m_s, geometry.angular_velocity_rad_s,
                geometry.joint_count, geometry.body_count, geometry.velocity_count, geometry.configuration_count,
                np.r_[geometry.position[:7], reduced[row, :spec.k, 0]],
                np.r_[geometry.velocity[:6], reduced[row, :spec.k, 1]], spec.joint_names, spec.full_joint_names,
            )
        return result

    def diagnostics(self, keys):
        self._check()
        output = {world: None for world in keys}
        selected = [world for world in keys if self._valid[world]]
        if selected:
            output.update(self._read_diagnostics(selected))
        return output

    def _read_diagnostics(self, selected):
        """Read complete same-attempt buffers before or after the single publication."""
        self._check()
        if not selected:
            return {}
        p = self.physical
        with self.batch._device():
            self.batch._sync()
            full = {name: self._read(value, "floating_affine_diagnostic") for name, value in (
                ("linear", p.linear), ("angular", p.angular), ("mass", p.mass), ("vector", p.vector),
                ("entry_pose", p.entry_pose), ("info", p.info), ("diag", p.diag),
            )}
            reduced = {name: self._read(value, "floating_affine_diagnostic") for name, value in (
                ("linear", self.linear), ("angular", self.angular), ("mass", self.mass),
                ("vector", self.vector), ("reaction", self.reaction), ("row_receipt", self.row_receipt),
                ("metrics", self.metrics),
            )}
        self.batch._counts["floating_affine_diagnostic_reads"] += 1
        return self.diagnostics_from_arrays(selected, full, reduced)

    def diagnostics_from_arrays(self, worlds, full_arrays, reduced_arrays):
        """Pure owned records from caller's same-attempt readback; no IO or physics."""
        self._check()
        worlds = tuple(worlds)
        if len(set(worlds)) != len(worlds) or any(w not in self.rows for w in worlds):
            raise ValueError("floating affine diagnostic worlds must be unique storage members")
        if not worlds:
            return {}
        t, b, df, dr = self.count, self.bmax, self.dmax, self.reduced_dmax
        full_shapes = {
            "linear": (t, b, 3, df), "angular": (t, b, 3, df), "mass": (t, df, df),
            "vector": (t, df, 14), "entry_pose": (t, 7), "info": (t, 4), "diag": (t, 3),
        }
        reduced_shapes = {
            "linear": (t, b, 3, dr), "angular": (t, b, 3, dr), "mass": (t, dr, dr),
            "vector": (t, dr, 14), "reaction": (t, df, 3), "row_receipt": (t, self.rmax, 2),
            "metrics": (t, 4),
        }

        def arrays(values, shapes):
            if not isinstance(values, Mapping) or not shapes.keys() <= values.keys():
                raise ValueError("floating affine diagnostic readback is incomplete")
            result = {}
            for name, shape in shapes.items():
                value = np.asarray(values[name])
                dtype = np.int32 if name == "info" else np.float32
                if value.dtype != np.dtype(dtype) or value.shape != shape:
                    raise ValueError("floating affine diagnostic dtype/dimensions changed: " + name)
                result[name] = value
            return result

        full, reduced = arrays(full_arrays, full_shapes), arrays(reduced_arrays, reduced_shapes)
        output = {}
        for world in worlds:
            row, spec = self.rows[world], self.specs[world]
            info, diag = full["info"][row], full["diag"][row]
            if info[2] != 0:
                raise ValueError("floating affine diagnostic requested a rejected attempt")
            if int(info[0]) < 0 or int(info[1]) != int(info[0]) + 1:
                raise ValueError("floating affine diagnostic ticks are not one physical step")
            f, v = full["vector"][row, :spec.d], reduced["vector"][row, :spec.reduced_d]
            pose = full["entry_pose"][row]
            metrics = reduced["metrics"][row]
            used = (
                f, v, pose, diag, metrics,
                full["linear"][row, :spec.b, :, :spec.d],
                full["angular"][row, :spec.b, :, :spec.d],
                full["mass"][row, :spec.d, :spec.d],
                reduced["linear"][row, :spec.b, :, :spec.reduced_d],
                reduced["angular"][row, :spec.b, :, :spec.reduced_d],
                reduced["mass"][row, :spec.reduced_d, :spec.reduced_d],
                reduced["reaction"][row, :spec.d],
                reduced["row_receipt"][row, :spec.r],
            )
            if any(not np.isfinite(value).all() for value in used):
                raise ValueError("floating affine diagnostic contains nonfinite selected data")
            try:
                with np.errstate(over="raise", invalid="raise"):
                    prefix = np.r_[spec.anchor + pose[:3].astype(float), pose[3:7]]
                    q = np.r_[prefix, v[6:, 4]]
                    full_q = np.r_[prefix, f[6:, 4]]
            except FloatingPointError as error:
                raise ValueError("floating affine diagnostic world position is nonfinite") from error
            if not np.isfinite(q).all() or not np.isfinite(full_q).all():
                raise ValueError("floating affine diagnostic world position is nonfinite")
            output[world] = FloatingAffineTreeDiagnostic(
                entry_tick=int(info[0]), exit_tick=int(info[1]), dt_s=float(diag[0]),
                entry_position=q, entry_velocity=v[:, 5],
                linear_jacobian=full["linear"][row, :spec.b, :, :spec.d],
                angular_jacobian=full["angular"][row, :spec.b, :, :spec.d],
                mass=full["mass"][row, :spec.d, :spec.d], bias=f[:, 0], external=f[:, 1],
                drive=v[6:, 2], acceleration=v[:, 3], iterations=int(info[3]),
                control_residual=float(diag[1]), equation_residual=max(float(diag[2]), float(metrics[0])),
                full_entry_position=full_q, full_entry_velocity=f[:, 5],
                reduced_mass=reduced["mass"][row, :spec.reduced_d, :spec.reduced_d],
                reduced_bias=v[:, 0], reduced_external=v[:, 1],
                reduced_linear_jacobian=reduced["linear"][row, :spec.b, :, :spec.reduced_d],
                reduced_angular_jacobian=reduced["angular"][row, :spec.b, :, :spec.reduced_d],
                full_acceleration=f[:, 3], constraint_effort_full=reduced["reaction"][row, :spec.d, 0],
                row_multipliers=reduced["row_receipt"][row, :spec.r, 0],
                row_powers_w=reduced["row_receipt"][row, :spec.r, 1], constraint_power_w=float(metrics[1]),
                root_constraint_residual=reduced["reaction"][row, :6, 0],
            )
        return output

    def summary(self, keys, active):
        self._check()
        return self.physical.summary(keys, active)

    def borrow_geometry(self, active):
        self._check()
        return self.physical.borrow_geometry(active)

    def close(self):
        # All buffers are on the original batch stack. No borrowed runtime close.
        self._closed = True
        self.physical.close()
