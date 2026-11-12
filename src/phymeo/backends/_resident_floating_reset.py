"""Floating-family frozen reset bindings; allocation/publication belong to the reset plan."""

from types import MappingProxyType

import numpy as np

from ._resident_floating_tree import floating_packet
from ._tyx_resident_reset_kernels import make_floating_reset_kernels


class FloatingResetBinding:
    def __init__(self, storage, allocate):
        self.storage = storage
        batch = storage.batch
        self.rows = np.array([batch._slots[w] for w in storage.worlds], np.intp)
        self.mask = np.zeros(storage.count, bool)
        root = np.zeros((storage.count, 13), np.float32)
        state = np.zeros((storage.count, storage.npad, 7), np.float32)
        applied = np.zeros((storage.count, storage.dmax), np.float32)
        drive = np.zeros_like(storage._held_drive)
        effort = np.zeros_like(storage._held_effort)
        loads = np.zeros_like(storage._held_loads)
        for key, row in storage.rows.items():
            spec = storage.specs[key]
            r, s, a, packet, _ = spec.restore_row(batch._initial[key], key)
            root[row], state[row, : max(1, spec.n)], applied[row, : spec.d] = r, s, a
            d, e, f = floating_packet(packet, spec.n)
            drive[row, : spec.n], effort[row, : spec.n], loads[row, : spec.b] = d, e, f
        self.controls = (
            ("_held_drive", drive, self.mask[:, None, None]),
            ("_held_effort", effort, self.mask[:, None]),
            ("_held_loads", loads, self.mask[:, None, None]),
        )
        self.initial = (
            allocate(root),
            allocate(state),
            allocate(applied),
            allocate(np.zeros((storage.count, storage.bmax, 18), np.float32)),
            allocate(np.zeros((storage.count, storage.bmax, 6), np.float32)),
            allocate(np.zeros(storage.count, np.float32)),
        )
        r, s, _, fk, twist, energy = self.initial
        batch._launch(
            storage.kernels["geometry"][0],
            storage.count,
            storage.meta,
            storage.parents,
            storage.kinds,
            storage.properties,
            storage.zero,
            r,
            s,
            fk,
            storage.frame,
            storage.flags,
        )
        batch._launch(
            storage.kernels["energy"][0],
            storage.count,
            storage.meta,
            storage.parents,
            storage.kinds,
            storage.properties,
            storage.principal,
            r,
            s,
            fk,
            storage.origins,
            twist,
            energy,
            storage.flags,
        )
        batch._counts["floating_fk_kernel_launches"] += 1
        batch._counts["floating_energy_kernel_launches"] += 1
        self.kernels = make_floating_reset_kernels()
        self.validity = [{w: False for w in storage.worlds} for _ in (0, 1)]
        self.codes = [{w: "RESTORED" for w in storage.worlds} for _ in (0, 1)]
        self.code_views = tuple(MappingProxyType(c) for c in self.codes)

    def launch(self, bank, selected_ticks, src, dst):
        s = self.storage
        s.batch._launch(
            self.kernels[bank],
            s.count,
            s.meta,
            selected_ticks,
            *self.initial,
            s.roots[src],
            s.states[src],
            s.applied[src],
            s.fk[src],
            s.twist[src],
            s.energy[src],
            s.roots[dst],
            s.states[dst],
            s.applied[dst],
            s.fk[dst],
            s.twist[dst],
            s.energy[dst],
        )

    def select(self, mask):
        np.take(mask, self.rows, out=self.mask, mode="clip")

    def prepare_metadata(self, bank):
        for key, row in self.storage.rows.items():
            selected = self.mask[row]
            self.validity[bank][key] = False if selected else self.storage._valid[key]
            self.codes[bank][key] = "RESTORED" if selected else self.storage.status_codes[key]

    def publish(self, bank):
        self.storage._valid = self.validity[bank]
        self.storage.status_codes = self.code_views[bank]
        self.storage._pending_info = None
