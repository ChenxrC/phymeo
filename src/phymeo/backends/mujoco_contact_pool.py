"""Opt-in CPU contact capture into bounded, leased numeric slabs.

Only native numeric values are copied. No native object or pointer escapes the
capture call. Exported arrays pin their slab, even when retained independently
of the packet. Slabs are recycled only after every exported view is released.
"""

import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType

import numpy as np

from .mujoco_recording import (
    _LAYOUT,
    CONTACT_COLUMN_SCHEMA,
    CONTACT_ROW_BYTES,
    MuJoCoContactColumns,
)

_NATIVE = {
    "geometry_ids": "geom",
    "flex_ids": "flex",
    "element_ids": "elem",
    "vertex_ids": "vert",
    "dimension": "dim",
    "distance": "dist",
    "position": "pos",
    "frame": "frame",
    "include_margin": "includemargin",
    "friction": "friction",
    "solref": "solref",
    "solref_friction": "solreffriction",
    "solimp": "solimp",
    "regularized_mu": "mu",
    "cone_hessian": "H",
    "excluded": "exclude",
    "constraint_address": "efc_address",
}
# Match _contacts/_frozen: inactive Hessians and scalar fields are not filtered.
_FINITE = frozenset(
    {
        "position",
        "frame",
        "friction",
        "solref",
        "solref_friction",
        "solimp",
        "wrench_contact_frame",
    }
)


class ContactPoolCapacityError(RuntimeError):
    """The numeric slab cap cannot accommodate another unmodified capture."""


class _Block:
    def __init__(self, capacity):
        self.capacity = capacity
        self.arrays = {
            name: np.empty((capacity, *shape), dtype=dtype) for name, dtype, shape in _LAYOUT
        }


class _Lease:
    def __init__(self, pool, block):
        self.pool = weakref.ref(pool)
        self.block = block

    def __del__(self):
        pool = self.pool()
        if pool is not None:
            pool._release(self.block)


class _ReadOnlyExport:
    """NumPy retains this exporter as .base, including through derived views."""

    def __init__(self, array, lease):
        self._array = array
        self._lease = lease

    @property
    def __array_interface__(self):
        interface = self._array.__array_interface__.copy()
        interface["data"] = (interface["data"][0], True)
        return interface


@dataclass(frozen=True, slots=True)
class PooledMuJoCoContacts(Sequence):
    """Read-only leased columns; individual MuJoCoContact records are lazy.

    Standard ndarray views, np.asarray and memoryview retain the lease. Keep
    the packet/arrays until a background writer finishes. Do not access private
    exporters or mutate their storage. Use detach() for independent immutable
    bytes-backed storage when indefinite retention should not pin pool capacity.
    """

    count: int = field()
    arrays: object
    schema: str = CONTACT_COLUMN_SCHEMA

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self.contact(i) for i in range(*index.indices(self.count)))
        if isinstance(index, (int, np.integer)) and not isinstance(index, bool) and index < 0:
            index += self.count
        return self.contact(index)

    def contact(self, index):
        return MuJoCoContactColumns(self.count, self.arrays).contact(index)

    @property
    def nbytes(self):
        return self.count * CONTACT_ROW_BYTES

    def detach(self):
        arrays = {
            name: np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)
            for name, a in self.arrays.items()
        }
        return MuJoCoContactColumns(self.count, MappingProxyType(arrays))


class MuJoCoContactPool:
    """Bound persistent numeric storage, with explicit failure instead of loss.

    max_bytes counts allocated slab capacity, including free and pinned slabs;
    it excludes Python metadata and short-lived vector validation/index arrays.
    Native MjData must be exclusively owned by the caller during capture_native.
    A pool lock protects allocation/publication, not concurrent native stepping.
    """

    def __init__(self, *, max_bytes=128 * 1024**2, block_rows=4096):
        if type(max_bytes) is not int or max_bytes < CONTACT_ROW_BYTES:
            raise ValueError("max_bytes must hold at least one complete contact row")
        if type(block_rows) is not int or block_rows <= 0:
            raise ValueError("block_rows must be a positive integer")
        self.max_bytes = max_bytes
        self.block_rows = block_rows
        self._lock = RLock()
        self._active = None
        self._used = 0
        self._free = []
        self._closed = False
        self._allocated = self._peak = self._allocations = self._reuses = 0
        self._captures = self._rows = 0
        self._empty = PooledMuJoCoContacts(
            0,
            MappingProxyType(
                {
                    name: np.frombuffer(b"", dtype=dtype).reshape((0, *shape))
                    for name, dtype, shape in _LAYOUT
                }
            ),
        )

    @property
    def stats(self):
        with self._lock:
            return MappingProxyType(
                dict(
                    allocated_bytes=self._allocated,
                    peak_bytes=self._peak,
                    max_bytes=self.max_bytes,
                    allocations=self._allocations,
                    reuses=self._reuses,
                    captures=self._captures,
                    rows=self._rows,
                    free_blocks=len(self._free),
                    closed=self._closed,
                )
            )

    def _release(self, block):
        with self._lock:
            if self._closed:
                self._allocated -= block.capacity * CONTACT_ROW_BYTES
            else:
                self._free.append(block)

    def flush(self):
        """Retire the current slab; outstanding views continue to own it."""
        with self._lock:
            self._active = None
            self._used = 0

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._allocated -= sum(b.capacity * CONTACT_ROW_BYTES for b in self._free)
            self._free.clear()
            self.flush()

    def _reserve(self, count):
        if self._active is not None and self._used + count <= self._active.block.capacity:
            return self._active, self._used
        self.flush()
        fits = [b for b in self._free if b.capacity >= count]
        if fits:
            block = min(fits, key=lambda b: b.capacity)
            self._free.remove(block)
            self._reuses += 1
        else:
            # Discard only unleased free blocks. Live exports are never evicted.
            self._allocated -= sum(b.capacity * CONTACT_ROW_BYTES for b in self._free)
            self._free.clear()
            available = (self.max_bytes - self._allocated) // CONTACT_ROW_BYTES
            if count > available:
                raise ContactPoolCapacityError(
                    "contact pool exhausted; release views or detach; no rows truncated"
                )
            capacity = min(max(count, self.block_rows), available)
            block = _Block(capacity)
            self._allocated += capacity * CONTACT_ROW_BYTES
            self._peak = max(self._peak, self._allocated)
            self._allocations += 1
        self._active = _Lease(self, block)
        return self._active, 0

    def capture_native(self, mj, model, data):
        """Copy all 19 fields directly from native contacts; publish atomically."""
        with self._lock:
            if self._closed:
                raise RuntimeError("contact pool is closed")
            count = int(data.ncon)
            if count < 0:
                raise ValueError("negative native contact count")
            if count == 0:
                self._captures += 1
                return self._empty
            if count * CONTACT_ROW_BYTES > self.max_bytes:
                raise ContactPoolCapacityError(
                    "native batch exceeds contact pool cap; no rows truncated"
                )
            lease, start = self._reserve(count)
            columns = {name: a[start : start + count] for name, a in lease.block.arrays.items()}
            columns["index"][:] = np.arange(count, dtype=np.int64)
            for name, dtype, shape in _LAYOUT:
                if name not in _NATIVE:
                    continue
                source = np.asarray(getattr(data.contact, _NATIVE[name]))
                expected_shape = (count, 9) if name == "frame" else (count, *shape)
                if source.shape != expected_shape or (
                    source.dtype != np.dtype("f8")
                    if dtype == "f8"
                    else source.dtype.kind != "i" or source.dtype.itemsize > 8
                ):
                    raise ValueError(f"unexpected native contact schema for {name}")
                np.copyto(columns[name], source.reshape(columns[name].shape), casting="safe")
            for index in range(count):
                mj.mj_contactForce(model, data, index, columns["wrench_contact_frame"][index])
            for name in _FINITE:
                if not np.all(np.isfinite(columns[name])):
                    raise RuntimeError(f"nonfinite native contact array: {name}")
            arrays = {name: np.asarray(_ReadOnlyExport(a, lease)) for name, a in columns.items()}
            packet = PooledMuJoCoContacts(count, MappingProxyType(arrays))
            self._used = start + count
            self._captures += 1
            self._rows += count
            return packet


def retained_contact_bytes(packets):
    """Count unique numeric allocations pinned by column packets for queue admission.

    Includes unused slab rows; excludes Python metadata and unrelated arrays.
    This conservative charge may count a slab in more than one queued record.
    It never permits a queue to charge only the visible slice of a larger slab.
    """
    allocations = {}
    for packet in packets:
        if not isinstance(packet, (PooledMuJoCoContacts, MuJoCoContactColumns)):
            raise TypeError("expected a contact-column packet")
        for array in packet.arrays.values():
            root = array
            while isinstance(root.base, np.ndarray):
                root = root.base
            if isinstance(root.base, _ReadOnlyExport):
                block = root.base._lease.block
                allocations[id(block)] = block.capacity * CONTACT_ROW_BYTES
            elif isinstance(root.base, bytes):
                allocations[id(root.base)] = len(root.base)
            else:
                raise ValueError("contact array has an unrecognized ownership chain")
    return sum(allocations.values())
