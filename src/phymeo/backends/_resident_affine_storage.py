"""N physical plus K independent banks in one existing TYX batch transaction.

The parent owns lock/runtime/resources, active swap, ticks and epoch. No method
here publishes physics. All HOST calculations are control/restore declarations
or explicit counted readback, never current-step dynamics or bounds.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ._resident_affine_tree import AffineTreeDiagnostic, AffineTreeKinematics, AffineTreeWorldSpec
from ._resident_tree_storage import STATUS_NAMES, TreeDeviceStorage
from ._resident_tree_world import _owned

AFFINE_STATUS_NAMES = {**STATUS_NAMES, 7: "AFFINE_CONSTRAINT_RESIDUAL", 8: "AFFINE_FULL_EQUATION_RESIDUAL"}


@dataclass(frozen=True, eq=False)
class AffineControlStage:
    owner: object
    drive: np.ndarray
    loads: np.ndarray
    physical: np.ndarray
    commands: np.ndarray


@dataclass(frozen=True, eq=False)
class AffineRestoreStage:
    owner: object
    mask: np.ndarray
    ticks: np.ndarray
    state: np.ndarray
    controls: AffineControlStage
    physical: object
    validity: Mapping
    status_codes: Mapping


class AffineTreeDeviceStorage:
    """Compose the C4 physical storage and algebraic independent coordinates.

    reduced meta int32[A,4] is global slot,K,delay,budget. P[A,N,K], c[A,N],
    original rows and independent selection are immutable static data. K state
    has exactly the old algebraic 7-column layout and its own two handles.
    The C4 solve kernel receives K dimensions; C4 full mechanics never does.
    """

    def __init__(self, batch, specs_by_world):
        from ._tyx_resident_affine_kernels import make_affine_kernels

        if not isinstance(specs_by_world, Mapping) or not specs_by_world:
            raise ValueError("affine storage requires a nonempty compiled mapping")
        self.batch = batch
        self.specs = MappingProxyType(dict(specs_by_world))
        if not all(isinstance(spec, AffineTreeWorldSpec) for spec in self.specs.values()):
            raise TypeError("affine storage needs AffineTreeWorldSpec entries")
        self.worlds = tuple(self.specs)
        self.rows = MappingProxyType({world: i for i, world in enumerate(self.worlds)})
        self.count = len(self.worlds)
        self.nmax = max(spec.n for spec in self.specs.values())
        self.kmax = max(spec.k for spec in self.specs.values())
        self.rmax = max(spec.r for spec in self.specs.values())
        self._owner = object()
        self._closed = False
        self._dirty = False
        self._pending_info = None
        self._valid = {world: False for world in self.worlds}
        self.status_codes = MappingProxyType({world: "INITIALIZED" for world in self.worlds})
        for category in ("affine_topology", "affine_initial", "affine_control", "affine_restore",
                         "affine_state", "affine_fk", "affine_diagnostic", "affine_status"):
            for suffix in ("uploaded_bytes", "downloaded_bytes", "host_read_bytes"):
                batch._counts.setdefault(category + "_" + suffix, 0)
        for name in ("affine_projection_kernel_launches", "affine_solve_kernel_launches",
                     "affine_expand_kernel_launches", "affine_commit_kernel_launches",
                     "affine_restore_kernel_launches", "affine_validation_kernel_launches",
                     "affine_state_reads", "affine_fk_reads", "affine_diagnostic_reads", "affine_status_reads"):
            batch._counts.setdefault(name, 0)
        self.physical = TreeDeviceStorage(batch, {w: s.physical for w, s in self.specs.items()})
        self.kernels = make_affine_kernels()
        a, n, k, r = self.count, self.nmax, self.kmax, self.rmax
        meta = np.zeros((a, 4), np.int32)
        matrix, offset = np.zeros((a, n, k), np.float32), np.zeros((a, n), np.float32)
        independent = np.zeros((a, k), np.int32)
        row_indices, reverse_rows = np.zeros((a, r, 2), np.int32), np.zeros((a, r), np.int32)
        row_values = np.zeros((a, r, 2), np.float32)
        full_limits, limits = np.zeros((a, n, 3), np.float32), np.zeros((a, k, 3), np.float32)
        properties, states = np.zeros((a, k, 16), np.float32), np.zeros((a, k, 7), np.float32)
        held_drive, held_loads = np.zeros((a, k, 3), np.float32), np.zeros((a, n, 7), np.float32)
        for world, spec in self.specs.items():
            i = self.rows[world]
            meta[i] = (batch._slots[world], spec.k, spec.delay, spec.control_max_iterations)
            matrix[i, :spec.n, :spec.k], offset[i, :spec.n] = spec.matrix, spec.offset
            independent[i, :spec.k] = spec.independent
            row_indices[i, :spec.r], reverse_rows[i, :spec.r] = spec.row_indices, spec.reverse_rows
            row_values[i, :spec.r] = spec.row_values
            full_limits[i, :spec.n], limits[i, :spec.k] = spec.full_limits, spec.limits
            properties[i, :spec.k, 13], properties[i, :spec.k, 14], properties[i, :spec.k, 15] = spec.kp, spec.kd, spec.cap
            states[i, :spec.k], held_drive[i, :spec.k], held_loads[i, :spec.n] = spec.initial_state, spec.initial_drive, spec.initial_loads
        self._held_drive, self._held_loads = _owned(held_drive), _owned(held_loads)
        commands = np.zeros((a, k, 10), np.float32)
        commands[:, :, :3] = held_drive
        self._held_commands = _owned(commands)
        with batch._device():
            def put(value, category="affine_topology"):
                return batch._buffer(value.shape, value.dtype, value, category)
            def blank(shape, dtype=np.float32):
                return put(np.zeros(shape, dtype), "affine_initial")
            self.meta, self.matrix, self.offset = put(meta), put(matrix), put(offset)
            self.independent = put(independent)
            self.row_indices, self.row_values, self.reverse_rows = put(row_indices), put(row_values), put(reverse_rows)
            self.full_limits, self.limits = put(full_limits), put(limits)
            self.properties = put(properties)
            self.commands = put(commands, "affine_initial")
            self.states = (put(states, "affine_initial"), put(states, "affine_initial"))
            self.mass, self.factor, self.mobility = blank((a, k, k)), blank((a, k, k)), blank((a, k, k))
            self.vector = blank((a, k, 14))
            self.linear, self.angular = blank((a, n, 3, k)), blank((a, n, 3, k))
            self.reaction, self.row_receipt = blank((a, n, 3)), blank((a, r, 2))
            self.metrics = blank((a, 3))
            self.restore_state = blank((a, k, 7))
            self.restore_flags = blank((a,), np.int32)
            for bank in (0, 1):
                self._validate_state(bank)
                batch._sync()
                self.check_restore()
        batch._resources.callback(self.close)

    def _check(self):
        if self._closed:
            raise RuntimeError("affine storage is closed")
        self.batch._check_open()

    def _read(self, array, category):
        value = self.batch._read(array, category)
        if self.batch._mode == "interpret":
            self.batch._counts[category + "_host_read_bytes"] += value.nbytes
        return value

    def _control_stage(self, drive, loads):
        commands = np.zeros((self.count, self.kmax, 10), np.float32)
        commands[:, :, :3] = drive
        full = np.zeros((self.count, self.nmax, 10), np.float32)
        full[:, :, 3:] = loads
        return AffineControlStage(self._owner, _owned(drive), _owned(loads), _owned(full), _owned(commands))

    def validate_controls(self, by_world):
        self._check()
        if not isinstance(by_world, Mapping):
            raise TypeError("affine controls must be a mapping")
        drive, loads = self._held_drive.copy(), self._held_loads.copy()
        for world, value in by_world.items():
            if world not in self.rows:
                raise KeyError(world)
            spec, row = self.specs[world], self.rows[world]
            motor, external = spec.packet(value)
            drive[row, :spec.k], loads[row, :spec.n] = motor, external
        return self._control_stage(drive, loads)

    def commit_controls(self, stage):
        if not isinstance(stage, AffineControlStage) or stage.owner is not self._owner:
            raise ValueError("affine control stage belongs to a different storage")
        if stage.drive.shape != self._held_drive.shape or stage.loads.shape != self._held_loads.shape:
            raise ValueError("affine held-control shape changed")
        self.physical.commit_controls(stage.physical)
        if not np.array_equal(stage.drive, self._held_drive):
            self._dirty = True
        self._held_drive, self._held_loads, self._held_commands = stage.drive, stage.loads, stage.commands

    def upload_controls(self):
        self.physical.upload_controls()
        if self._dirty:
            self.batch._upload(self.commands, self._held_commands, "affine_control")

    def controls_uploaded(self):
        self.physical.controls_uploaded()
        self._dirty = False

    def invalidate_diagnostics(self):
        self.physical.invalidate_diagnostics()
        self._valid = {world: False for world in self.worlds}
        self._pending_info = None

    def stage_advance(self, h, src, dst, tick_src, tick_dst, accepted):
        self._check()
        self.invalidate_diagnostics()
        p, launch = self.physical, self.batch._launch
        p.stage_mechanics(src)
        launch(self.kernels["project"][src], self.count, p.meta, self.meta, self.matrix,
               self.states[src], p.mass, p.vector, p.linear, p.angular,
               self.mass, self.vector, self.linear, self.angular)
        self.batch._counts["affine_projection_kernel_launches"] += 1
        # Original C4 algebraic solve. K is not a fabricated physical tree.
        launch(p.kernels["solve"][src], self.count, h, self.meta, self.properties,
               self.commands, self.states[src], tick_src, self.states[dst],
               self.mass, self.factor, self.mobility, self.vector, p.info, p.diag)
        self.batch._counts["affine_solve_kernel_launches"] += 1
        launch(self.kernels["expand"][src], self.count, p.meta, self.meta, self.matrix,
               self.offset, self.independent, self.row_indices, self.row_values, self.reverse_rows,
               self.full_limits, self.limits, p.states[src], self.states[src], p.states[dst],
               self.states[dst], p.mass, p.vector, self.vector, self.reaction, self.row_receipt,
               p.info, self.metrics)
        self.batch._counts["affine_expand_kernel_launches"] += 1
        p.stage_candidate_geometry(dst)
        p.stage_finalize(src, dst, tick_src, tick_dst, accepted)
        launch(self.kernels["retain"][src], self.count, self.meta, self.states[src], self.states[dst], accepted)
        self.batch._counts["affine_commit_kernel_launches"] += 1

    def check_advance(self, status):
        """Small status read after parent sync, before its only bank swap."""
        self.batch._counts["affine_status_reads"] += 1
        info = self._read(self.physical.info, "affine_status")
        if info.shape != (self.count, 4):
            raise RuntimeError("affine status shape changed")
        valid, names = {}, {}
        for world, row in self.rows.items():
            code = int(info[row, 2])
            accepted = bool(status[world]) if isinstance(status, Mapping) else bool(status[self.batch._slots[world]])
            if code not in AFFINE_STATUS_NAMES or accepted != (code == 0):
                raise RuntimeError("affine status disagrees with full-world acceptance")
            valid[world], names[world] = accepted, AFFINE_STATUS_NAMES[code]
        self._pending_info = valid, MappingProxyType(names)

    def finish_advance(self, status):
        """Pure HOST publication only after the parent has committed all banks."""
        if self._pending_info is None:
            raise RuntimeError("affine accepted status was not checked before publication")
        self._valid, self.status_codes = self._pending_info
        self._pending_info = None

    def validate_restore(self, by_world):
        self._check()
        if not isinstance(by_world, Mapping):
            raise TypeError("affine checkpoints must be a mapping")
        state = np.zeros((self.count, self.kmax, 7), np.float32)
        drive, loads = self._held_drive.copy(), self._held_loads.copy()
        physical_points = {}
        for world, checkpoint in by_world.items():
            if world not in self.rows:
                raise KeyError(world)
            row, spec = self.rows[world], self.specs[world]
            reduced, full, motor, external, tick = spec.restore_row(checkpoint, world)
            state[row, :spec.k], drive[row, :spec.k], loads[row, :spec.n] = reduced, motor, external
            command = np.zeros((spec.n, 10), np.float32)
            command[:, 3:] = external
            physical_points[world] = spec.physical.checkpoint(world, full, tick, command)
        physical = self.physical.validate_restore(physical_points)
        validity = {w: False if physical.mask[row] else self._valid[w] for w, row in self.rows.items()}
        codes = MappingProxyType({w: "RESTORED" if physical.mask[row] else self.status_codes[w] for w, row in self.rows.items()})
        return AffineRestoreStage(self._owner, physical.mask, physical.ticks, _owned(state),
                                  self._control_stage(drive, loads), physical, MappingProxyType(validity), codes)

    def _validate_state(self, bank):
        p = self.physical
        self.batch._launch(self.kernels["validate_state"][bank], self.count, p.meta, self.meta,
            self.matrix, self.offset, self.independent, self.full_limits, self.limits,
            p.states[bank], self.states[bank], self.restore_flags)
        self.batch._counts["affine_validation_kernel_launches"] += 1

    def stage_restore(self, stage, src, dst, tick_src, tick_dst):
        self._check()
        if not isinstance(stage, AffineRestoreStage) or stage.owner is not self._owner:
            raise ValueError("affine restore stage belongs to a different storage")
        p = self.physical
        p.stage_restore(stage.physical, src, dst, tick_src, tick_dst)
        self.batch._upload(self.restore_state, stage.state, "affine_restore")
        # The original copy/restore primitive uses the same mask/ticks but K rows.
        self.batch._launch(p.kernels["restore"][src], self.count, self.meta,
            p.restore_mask, self.restore_state, p.restore_ticks, self.states[src], self.states[dst], tick_src, tick_dst)
        self.batch._counts["affine_restore_kernel_launches"] += 1
        self._validate_state(dst)

    def check_restore(self):
        self.physical.check_restore()
        self.batch._counts["affine_status_reads"] += 1
        flags = self._read(self.restore_flags, "affine_status")
        if flags.shape != (self.count,) or np.any(flags != 1):
            raise ValueError("affine restored/initial independent and physical state disagree")

    def commit_restore(self, stage):
        self.commit_controls(stage.controls)
        self.physical.commit_restore(stage.physical)
        self._valid, self.status_codes = dict(stage.validity), stage.status_codes
        self._pending_info = None

    def snapshot(self, keys, active, ticks):
        self._check()
        if not keys:
            return {}
        with self.batch._device():
            self.batch._sync()
            reduced = self._read(self.states[active], "affine_state")
            full = self.physical._read(self.physical.states[active], "tree_state")
            time = self._read(ticks, "affine_state")
        self.batch._counts["affine_state_reads"] += 1
        self.batch._counts["tree_state_reads"] += 1
        return {w: self.specs[w].checkpoint(w, reduced[self.rows[w]], full[self.rows[w]],
            time[self.batch._slots[w]], self._held_drive[self.rows[w]], self._held_loads[self.rows[w]]) for w in keys}

    def kinematics(self, keys, active, ticks):
        self._check()
        if not keys:
            return {}
        with self.batch._device():
            self.batch._sync()
            reduced = self._read(self.states[active], "affine_fk")
            full = self.physical._read(self.physical.states[active], "tree_fk")
            fk = self.physical._read(self.physical.fk[active], "tree_fk")
            time = self._read(ticks, "affine_fk")
        self.batch._counts["affine_fk_reads"] += 1
        self.batch._counts["tree_fk_reads"] += 1
        output = {}
        for world in keys:
            row, spec = self.rows[world], self.specs[world]
            geom = fk[row, :spec.n]
            output[world] = AffineTreeKinematics(int(time[self.batch._slots[world]]), spec.physical.joint_types,
                full[row, :spec.n, 0], full[row, :spec.n, 1],
                geom[:, :3].astype(float)+spec.physical.anchor, geom[:, 3:12].reshape(spec.n, 3, 3),
                geom[:, 15:18].astype(float)+spec.physical.anchor,
                reduced[row, :spec.k, 0], reduced[row, :spec.k, 1], spec.joint_names, spec.full_joint_names)
        return output

    def diagnostics(self, keys):
        self._check()
        output = {world: None for world in keys}
        selected = [world for world in keys if self._valid[world]]
        if not selected:
            return output
        p = self.physical
        with self.batch._device():
            self.batch._sync()
            full_linear, full_angular, full_mass, full_vectors, info, diag = [p._read(a, "tree_diagnostic")
                for a in (p.linear, p.angular, p.mass, p.vector, p.info, p.diag)]
            linear, angular, mass, vectors, reaction, rows, metrics = [self._read(a, "affine_diagnostic")
                for a in (self.linear, self.angular, self.mass, self.vector, self.reaction, self.row_receipt, self.metrics)]
        self.batch._counts["affine_diagnostic_reads"] += 1
        self.batch._counts["tree_diagnostic_reads"] += 1
        for world in selected:
            row, spec = self.rows[world], self.specs[world]
            n, k, r = spec.n, spec.k, spec.r
            v, f = vectors[row, :k], full_vectors[row, :n]
            output[world] = AffineTreeDiagnostic(
                entry_tick=int(info[row, 0]), exit_tick=int(info[row, 1]), dt_s=float(diag[row, 0]),
                entry_position=v[:, 4], entry_velocity=v[:, 5],
                linear_jacobian=full_linear[row, :n, :, :n], angular_jacobian=full_angular[row, :n, :, :n],
                mass=full_mass[row, :n, :n], bias=f[:, 0], external=f[:, 1], drive=v[:, 2], acceleration=v[:, 3],
                iterations=int(info[row, 3]), control_residual=float(diag[row, 1]),
                equation_residual=max(float(diag[row, 2]), float(metrics[row, 0])),
                full_entry_position=f[:, 4], full_entry_velocity=f[:, 5],
                reduced_mass=mass[row, :k, :k], reduced_bias=v[:, 0], reduced_external=v[:, 1],
                reduced_linear_jacobian=linear[row, :n, :, :k], reduced_angular_jacobian=angular[row, :n, :, :k],
                full_acceleration=f[:, 3], constraint_effort_full=reaction[row, :n, 0],
                row_multipliers=rows[row, :r, 0], row_powers_w=rows[row, :r, 1], constraint_power_w=float(metrics[row, 1]))
        return output

    def summary(self, keys, active):
        self._check()
        return self.physical.summary(keys, active)

    def borrow_geometry(self, active):
        self._check()
        return self.physical.borrow_geometry(active)

    def close(self):
        # Device buffers belong to the batch resource stack, including both banks.
        # Close never destroys the borrowed runtime or changes active state.
        self._closed = True
        self.physical.close()
