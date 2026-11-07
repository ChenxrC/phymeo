"""Tree banks owned by one existing TyxWorldBatch, never a second runtime.

All stage/read/borrow calls require the owner's lock. Only the owner changes
active, ticks' active handle, epoch, or public state. Commit methods do no IO.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ._resident_tree_world import TreeDiagnostic, TreeKinematics, TreeWorldSpec, _owned, tree_packet

STATUS_NAMES = {
    0: "ACCEPTED",
    1: "M_NOT_SPD",
    2: "NONFINITE",
    3: "M_RESIDUAL",
    4: "PD_NOT_CONVERGED",
    5: "STATE_DOMAIN",
    6: "FK_NOT_FINITE",
    7: "SITE_CAPACITY_EXCEEDED",
}


@dataclass(frozen=True, eq=False)
class TreeRestoreStage:
    owner: object
    mask: np.ndarray
    state: np.ndarray
    held: np.ndarray
    ticks: np.ndarray
    validity: Mapping
    status_codes: Mapping


@dataclass(frozen=True, eq=False)
class TreeGeometryView:
    """A same-lock loan of one accepted tree bank in anchor-relative world axes."""

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
            self.state,
            self.fk,
            ticks,
            bounds,
            stamp,
        )


class TreeDeviceStorage:
    """Persistent tree arrays allocated through the parent's counted resource stack.

    meta int32[T,4] = global slot,n,delay,budget; parents/kinds int32[T,N].
    static float32[T,N,16] = offset3,axis3,COM3,I3,mass,kp,kd,cap.
    State float32[T,N,7], FK float32[T,N,18] and energy float32[T] have two
    handles selected by the parent's single active index. Padding is zero.
    """

    def __init__(self, batch, specs_by_world):
        from ._tyx_resident_tree_kernels import make_tree_kernels

        if not isinstance(specs_by_world, Mapping) or not specs_by_world:
            raise ValueError("tree storage requires a nonempty compiled-world mapping")
        self.batch = batch
        self.specs = MappingProxyType(dict(specs_by_world))
        if not all(isinstance(s, TreeWorldSpec) for s in self.specs.values()):
            raise TypeError("tree storage requires compiled TreeWorldSpec objects")
        self.worlds = tuple(self.specs)
        self.rows = MappingProxyType({w: i for i, w in enumerate(self.worlds)})
        self.count = len(self.worlds)
        self.nmax = max(s.n for s in self.specs.values())
        self._owner = object()
        self._closed = False
        self._dirty = False
        self._valid = {w: False for w in self.worlds}
        self.status_codes = MappingProxyType({w: "INITIALIZED" for w in self.worlds})
        self._pending_info = None
        self.kernels = make_tree_kernels()
        for category in (
            "tree_topology",
            "tree_initial",
            "tree_control",
            "tree_restore",
            "tree_state",
            "tree_fk",
            "tree_summary",
            "tree_diagnostic",
            "tree_status",
        ):
            for suffix in ("uploaded_bytes", "downloaded_bytes", "host_read_bytes"):
                batch._counts.setdefault(category + "_" + suffix, 0)
        for name in (
            "tree_dynamics_kernel_launches",
            "tree_fk_kernel_launches",
            "tree_energy_kernel_launches",
            "tree_restore_kernel_launches",
            "tree_commit_kernel_launches",
            "tree_diagnostic_reads",
            "tree_state_reads",
            "tree_fk_reads",
            "tree_summary_reads",
            "tree_status_reads",
        ):
            batch._counts.setdefault(name, 0)
        t, n = self.count, self.nmax
        meta = np.zeros((t, 4), np.int32)
        parents = np.full((t, n), -1, np.int32)
        kinds = np.zeros((t, n), np.int32)
        static, zero = np.zeros((t, n, 16), np.float32), np.zeros((t, n, 9), np.float32)
        gravity, anchor = np.zeros((t, 3), np.float32), np.zeros((t, 3), np.float32)
        states, held = np.zeros((t, n, 7), np.float32), np.zeros((t, n, 10), np.float32)
        for world, spec in self.specs.items():
            row, size = self.rows[world], spec.n
            meta[row] = [batch._slots[world], size, spec.delay, spec.control_max_iterations]
            parents[row, :size], kinds[row, :size] = spec.parents, spec.kinds
            static[row, :size, :3] = spec.offsets
            static[row, :size, 3:6] = spec.local_axis
            static[row, :size, 6:9] = spec.com
            static[row, :size, 9:12] = spec.inertia
            for column, values in ((12, spec.mass), (13, spec.kp), (14, spec.kd), (15, spec.cap)):
                static[row, :size, column] = values
            zero[row, :size] = spec.zero_rotation.reshape(size, 9)
            gravity[row], anchor[row] = spec.gravity, spec.anchor
            states[row, :size], held[row, :size] = spec.initial_state, spec.initial_command
        self._held = held
        self._meta_host = _owned(meta, np.int32)
        with batch._device():

            def put(a, category="tree_topology"):
                return batch._buffer(a.shape, a.dtype, a, category)

            def blank(shape, dtype=np.float32):
                return batch._buffer(shape, dtype, np.zeros(shape, dtype), "tree_initial")

            self.meta, self.parents, self.kinds = put(meta), put(parents), put(kinds)
            self.static, self.zero, self.gravity, self.anchor = (
                put(static),
                put(zero),
                put(gravity),
                put(anchor),
            )
            self.commands = put(held, "tree_initial")
            self.states = (put(states, "tree_initial"), put(states, "tree_initial"))
            self.fk = (blank((t, n, 18)), blank((t, n, 18)))
            self.energy = (blank((t,)), blank((t,)))
            self.frame = blank((t, n, 9))
            self.flags = blank((t,), np.int32)
            self.linear, self.angular = blank((t, n, 3, n)), blank((t, n, 3, n))
            self.spatial = blank((t, n, 18))
            self.mass, self.factor, self.mobility = (
                blank((t, n, n)),
                blank((t, n, n)),
                blank((t, n, n)),
            )
            self.vector = blank((t, n, 14))
            self.info, self.diag = blank((t, 4), np.int32), blank((t, 3))
            self.restore_mask, self.restore_ticks = blank((t,), np.int32), blank((t,), np.int32)
            self.restore_state = blank((t, n, 7))
            for bank in (0, 1):
                self._geometry(bank)
                self._energy(bank)
            batch._sync()
            self.check_restore()
        batch._resources.callback(self.close)

    def _check(self):
        if self._closed:
            raise RuntimeError("tree storage is closed")
        self.batch._check_open()

    def _read(self, array, category):
        value = self.batch._read(array, category)
        if self.batch._mode == "interpret":
            self.batch._counts[category + "_host_read_bytes"] += value.nbytes
        return value

    def _geometry(self, bank):
        self.batch._launch(
            self.kernels["geometry"][bank],
            self.count,
            self.meta,
            self.parents,
            self.kinds,
            self.static,
            self.zero,
            self.states[bank],
            self.fk[bank],
            self.frame,
            self.flags,
        )
        self.batch._counts["tree_fk_kernel_launches"] += 1

    def _energy(self, bank):
        self.batch._launch(
            self.kernels["energy"][bank],
            self.count,
            self.meta,
            self.parents,
            self.kinds,
            self.static,
            self.states[bank],
            self.fk[bank],
            self.spatial,
            self.energy[bank],
            self.flags,
        )
        self.batch._counts["tree_energy_kernel_launches"] += 1

    def validate_controls(self, by_world):
        self._check()
        if not isinstance(by_world, Mapping):
            raise TypeError("tree controls must be a mapping")
        held = self._held.copy()
        for world, packet in by_world.items():
            if world not in self.rows:
                raise KeyError(world)
            spec = self.specs[world]
            held[self.rows[world], : spec.n] = tree_packet(packet, spec.domain, spec.n)
        return _owned(held)

    def commit_controls(self, held):
        if held.shape != self._held.shape:
            raise ValueError("tree held-control shape changed")
        if not np.array_equal(held, self._held):
            self._held = held
            self._dirty = True

    def upload_controls(self):
        if self._dirty:
            self.batch._upload(self.commands, self._held, "tree_control")

    def controls_uploaded(self):
        self._dirty = False

    def invalidate_diagnostics(self):
        self._valid = {world: False for world in self.worlds}
        self._pending_info = None

    def stage_mechanics(self, src):
        """Fill full physical M/b/J at the accepted bank; no solve or publication."""
        self._check()
        self.batch._launch(
            self.kernels["mechanics"][src],
            self.count,
            self.meta,
            self.parents,
            self.kinds,
            self.static,
            self.gravity,
            self.commands,
            self.states[src],
            self.fk[src],
            self.linear,
            self.angular,
            self.spatial,
            self.mass,
            self.vector,
        )
        self.batch._counts["tree_dynamics_kernel_launches"] += 1

    def stage_candidate_geometry(self, dst):
        """Rebuild full candidate FK/energy without changing the active bank."""
        self._check()
        self._geometry(dst)
        self._energy(dst)

    def stage_finalize(self, src, dst, tick_src, tick_dst, accepted):
        """Retain or reject the full candidate using its completed physical gates."""
        self._check()
        self.batch._launch(
            self.kernels["finalize"][src],
            self.count,
            self.meta,
            self.states[src],
            self.states[dst],
            self.fk[src],
            self.fk[dst],
            self.energy[src],
            self.energy[dst],
            tick_src,
            tick_dst,
            accepted,
            self.info,
            self.flags,
        )
        self.batch._counts["tree_commit_kernel_launches"] += 1

    def stage_solve(self, h, src, dst, tick_src):
        """Solve the already assembled full mechanics without publishing."""
        self._check()
        self.batch._launch(
            self.kernels["solve"][src],
            self.count,
            h,
            self.meta,
            self.static,
            self.commands,
            self.states[src],
            tick_src,
            self.states[dst],
            self.mass,
            self.factor,
            self.mobility,
            self.vector,
            self.info,
            self.diag,
        )
        self.batch._counts["tree_dynamics_kernel_launches"] += 1

    def stage_advance(self, h, src, dst, tick_src, tick_dst, accepted):
        self._check()
        self.invalidate_diagnostics()
        self.stage_mechanics(src)
        self.stage_solve(h, src, dst, tick_src)
        self.stage_candidate_geometry(dst)
        self.stage_finalize(src, dst, tick_src, tick_dst, accepted)

    def check_advance(self, status):
        """After parent's sync/status and BEFORE its active swap; may raise."""
        self.batch._counts["tree_status_reads"] += 1
        info = self._read(self.info, "tree_status")
        for world, row in self.rows.items():
            slot = self.batch._slots[world]
            valid = bool(status[world]) if isinstance(status, Mapping) else bool(status[slot])
            if int(info[row, 2]) not in STATUS_NAMES or valid != (int(info[row, 2]) == 0):
                raise RuntimeError("tree status code disagrees with batch acceptance")
        self._pending_info = (
            {w: int(info[r, 2]) == 0 for w, r in self.rows.items()},
            MappingProxyType({w: STATUS_NAMES[int(info[r, 2])] for w, r in self.rows.items()}),
        )

    def finish_advance(self, status):
        """Pure HOST publication after the owner's one successful swap; no IO."""
        self._valid, self.status_codes = self._pending_info
        self._pending_info = None

    def validate_restore(self, by_world):
        self._check()
        state = np.zeros((self.count, self.nmax, 7), np.float32)
        masks, ticks = np.zeros(self.count, np.int32), np.zeros(self.count, np.int32)
        held = self._held.copy()
        for world, checkpoint in by_world.items():
            if world not in self.rows:
                raise KeyError(world)
            row, spec = self.rows[world], self.specs[world]
            values, control, tick = spec.restore_row(checkpoint, world)
            state[row, : spec.n], held[row, : spec.n] = values, control
            masks[row], ticks[row] = 1, tick
        validity = {w: False if masks[r] else self._valid[w] for w, r in self.rows.items()}
        codes = MappingProxyType(
            {w: "RESTORED" if masks[r] else self.status_codes[w] for w, r in self.rows.items()}
        )
        return TreeRestoreStage(
            self._owner,
            _owned(masks, np.int32),
            _owned(state),
            _owned(held),
            _owned(ticks, np.int32),
            validity,
            codes,
        )

    def stage_restore(self, stage, src, dst, tick_src, tick_dst):
        if not isinstance(stage, TreeRestoreStage) or stage.owner is not self._owner:
            raise ValueError("restore stage belongs to a different tree storage")
        self.batch._upload(self.restore_mask, stage.mask, "tree_restore")
        self.batch._upload(self.restore_state, stage.state, "tree_restore")
        self.batch._upload(self.restore_ticks, stage.ticks, "tree_restore")
        self.batch._launch(
            self.kernels["restore"][src],
            self.count,
            self.meta,
            self.restore_mask,
            self.restore_state,
            self.restore_ticks,
            self.states[src],
            self.states[dst],
            tick_src,
            tick_dst,
        )
        self.batch._counts["tree_restore_kernel_launches"] += 1
        self._geometry(dst)
        self._energy(dst)

    def check_restore(self):
        self.batch._counts["tree_status_reads"] += 1
        flags = self._read(self.flags, "tree_status")
        if flags.shape != (self.count,) or np.any(flags != 1):
            raise ValueError("tree restored/initial FK or kinetic energy is not finite")

    def commit_restore(self, stage):
        self.commit_controls(stage.held)
        self._valid, self.status_codes = stage.validity, stage.status_codes

    def snapshot(self, keys, active, ticks):
        self._check()
        if not keys:
            return {}
        with self.batch._device():
            self.batch._sync()
            state, time = (
                self._read(self.states[active], "tree_state"),
                self._read(ticks, "tree_state"),
            )
        self.batch._counts["tree_state_reads"] += 1
        return {
            w: self.specs[w].checkpoint(
                w, state[self.rows[w]], time[self.batch._slots[w]], self._held[self.rows[w]]
            )
            for w in keys
        }

    def kinematics(self, keys, active, ticks):
        self._check()
        if not keys:
            return {}
        with self.batch._device():
            self.batch._sync()
            state, fk, time = (
                self._read(self.states[active], "tree_fk"),
                self._read(self.fk[active], "tree_fk"),
                self._read(ticks, "tree_fk"),
            )
        self.batch._counts["tree_fk_reads"] += 1
        out = {}
        for w in keys:
            r, spec = self.rows[w], self.specs[w]
            data = fk[r, : spec.n]
            out[w] = TreeKinematics(
                int(time[self.batch._slots[w]]),
                spec.joint_types,
                state[r, : spec.n, 0],
                state[r, : spec.n, 1],
                data[:, :3].astype(float) + spec.anchor,
                data[:, 3:12].reshape(spec.n, 3, 3),
                data[:, 15:18].astype(float) + spec.anchor,
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
            linear, angular, mass, vectors, info, diag = [
                self._read(a, "tree_diagnostic")
                for a in (self.linear, self.angular, self.mass, self.vector, self.info, self.diag)
            ]
        self.batch._counts["tree_diagnostic_reads"] += 1
        for w in valid:
            r, n = self.rows[w], self.specs[w].n
            v = vectors[r, :n]
            out[w] = TreeDiagnostic(
                int(info[r, 0]),
                int(info[r, 1]),
                float(diag[r, 0]),
                v[:, 4],
                v[:, 5],
                linear[r, :n, :, :n],
                angular[r, :n, :, :n],
                mass[r, :n, :n],
                v[:, 0],
                v[:, 1],
                v[:, 2],
                v[:, 3],
                int(info[r, 3]),
                float(diag[r, 1]),
                float(diag[r, 2]),
            )
        return out

    def summary(self, keys, active):
        self._check()
        if not keys:
            return {}
        with self.batch._device():
            self.batch._sync()
            energy = self._read(self.energy[active], "tree_summary")
        if not np.isfinite(energy).all():
            raise FloatingPointError("accepted tree kinetic energy is not finite")
        self.batch._counts["tree_summary_reads"] += 1
        return {w: {"kinetic_energy_j": float(energy[self.rows[w]])} for w in keys}

    def borrow_geometry(self, active):
        self._check()
        return TreeGeometryView(
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
            self.states[active],
            self.fk[active],
        )

    def close(self):
        # Every device buffer belongs to batch._resources. Never close its runtime
        # here and never try to free a borrowed active handle independently.
        self._closed = True
