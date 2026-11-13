"""Derived current-state forward frames; no integration or authoritative bank."""

from types import MappingProxyType

import numpy as np
import tyx as tx

from ._physolve_validation import _capacity, _finite
from .physolve_initial import INTEGRATION_FIELDS

FIELDS = frozenset(
    {
        "qpos",
        "qvel",
        "xpos",
        "xquat",
        "xmat",
        "xipos",
        "ximat",
        "xanchor",
        "xaxis",
        "geom_xpos",
        "geom_xmat",
        "site_xpos",
        "site_xmat",
        "subtree_com",
        "cdof",
        "cvel",
        "qM",
        "qfrc_bias",
        "qfrc_passive",
        "sensordata",
        "contact_geom",
        "contact_worldid",
        "nacon",
    }
)


@tx.kernel(gradient="unsupported")
def _publish(source: tx.Array[tx.f32], target: tx.Array[tx.f32], flags: tx.Array[tx.i32]):
    for i in tx.range(target.shape[0]):
        if flags[0] == 0:
            target[i] = source[i]


def _field(data, name):
    return getattr(data.contact, name[8:]) if name.startswith("contact_") else getattr(data, name)


INTEGER_FIELDS = frozenset({"contact_geom", "contact_worldid", "nacon"})


@tx.kernel(gradient="unsupported")
def _publish_int(source: tx.Array[tx.i32], target: tx.Array[tx.i32], flags: tx.Array[tx.i32]):
    for i in tx.range(target.shape[0]):
        if flags[0] == 0:
            target[i] = source[i]


@tx.kernel(gradient="unsupported")
def _contact_rows(count: tx.Array[tx.i32], world: tx.Array[tx.i32], geom: tx.Array[tx.i32],
                  rows: tx.Array[tx.i32], counts: tx.Array[tx.i32], flags: tx.Array[tx.i32],
                  capacity: tx.i32, ngeom: tx.i32):
    for i in tx.range(world.shape[0]):
        if flags[0] == 0 and i < count[0]:
            w = world[i]
            if w < 0 or w >= counts.shape[0] or geom[2*i] < 0 or geom[2*i] >= ngeom or geom[2*i+1] < 0 or geom[2*i+1] >= ngeom:
                tx.atomic_or(flags[0], 65536)
            else:
                slot = tx.atomic_add(counts[w], 1)
                if slot < capacity:
                    rows[(w*capacity+slot)*2] = geom[2*i]
                    rows[(w*capacity+slot)*2+1] = geom[2*i+1]
                else:
                    tx.atomic_or(flags[0], 65536)


@tx.kernel(gradient="unsupported")
def _dense_mass(
    source: tx.Array[tx.f32],
    target: tx.Array[tx.f32],
    flags: tx.Array[tx.i32],
    indices: tx.Array[tx.i32],
    source_width: tx.i32,
):
    for i in tx.range(target.shape[0]):
        if flags[0] == 0:
            world = i // indices.shape[0]
            address = indices[i % indices.shape[0]]
            value = 0.0
            if address >= 0:
                value = source[world * source_width + address]
            target[i] = value


class CurrentForwardFrame:
    """One disposable native scratch workspace and two captured source bindings.

    Source integration inputs are copied on device. The shared physical model
    and accepted/candidate states belong exclusively to PhySolveRuntime.
    Outputs are provisional until that runtime's completion gate succeeds.
    This first contract requires the stream to remain fixed until frame close.
    """

    contract_version = 2
    sample = "current state forward; provisional until execution completion"

    def __init__(self, runtime, fields):
        fields = tuple(fields)
        if not fields or len(set(fields)) != len(fields) or set(fields) - FIELDS:
            raise ValueError("select distinct supported current-forward fields")
        self.runtime = runtime
        self.fields = fields
        self.data = self.handle = None
        self.graphs = []
        self.arrays = MappingProxyType({})
        self.ready = False
        self.closed = False
        self.idle_pending = False
        self.refreshes = 0
        self.copy_bytes = 0
        self.logical_arrays = MappingProxyType({})
        self.contact_rows = self.contact_counts = None

    def prepare(self):
        import mujoco

        r = self.runtime
        self.data = r._frontend.put_data(
            r._cpu_model,
            mujoco.MjData(r._cpu_model),
            nworld=r.nworld,
            nconmax=r.nconmax,
            njmax=r.njmax,
        )
        r._frontend.prepare_workspace(r._model, self.data)
        self.handle = r._native._create_runtime(r._model, self.data, "tyx", "tyx")
        r._native._prepare_runtime(self.handle)
        self.sources = {
            name: _field(self.data, name).view(tx.f32).flatten() for name in self.fields
        }
        self.arrays = MappingProxyType(
            {name: tx.zeros_like(_field(self.data, name)) for name in self.fields}
        )
        logical = dict(self.arrays)
        if {"contact_geom", "contact_worldid", "nacon"}.issubset(self.fields):
            self.contact_rows = tx.zeros((r.nworld, self.data.naconmax, 2), tx.i32, device=r._device)
            self.contact_counts = tx.zeros(r.nworld, tx.i32, device=r._device)
            logical["contact_pairs"] = tx.zeros_like(self.contact_rows)
            logical["contact_valid_count"] = tx.zeros((r.nworld, 1), tx.i32, device=r._device)
        if "qM" in logical:
            nv = r._cpu_model.nv
            if self.data.qM.shape[1:] == (1, r._cpu_model.nM):
                mapping = np.full((nv, nv), -1, np.int32)
                for row in range(nv):
                    address, col = int(r._cpu_model.dof_Madr[row]), row
                    while col >= 0:
                        mapping[row, col] = mapping[col, row] = address
                        address, col = address + 1, int(r._cpu_model.dof_parentid[col])
            elif self.data.qM.shape[1] >= nv and self.data.qM.shape[1] == self.data.qM.shape[2]:
                padded = self.data.qM.shape[1]
                mapping = np.arange(nv, dtype=np.int32)[:, None] * padded + np.arange(nv, dtype=np.int32)
            else:
                raise NotImplementedError("unknown native mass matrix storage")
            self.mass_indices = tx.Array(mapping.reshape(-1), device=r._device)
            self.mass_source_width = self.data.qM.size // r.nworld
            logical["qM"] = tx.zeros((r.nworld, nv, nv), tx.f32, device=r._device)
        self.logical_arrays = MappingProxyType(logical)
        self.targets = {name: value.view(tx.f32).flatten() for name, value in self.arrays.items()}
        self.copies = [
            tuple(
                (getattr(bank, name), getattr(self.data, name))
                for name in INTEGRATION_FIELDS
                if getattr(bank, name).size
            )
            for bank in r._banks
        ]
        self.copy_bytes = sum(
            source.size if source.dtype == tx.bool1 else source.view(tx.f32).size * 4
            for source, _ in self.copies[0]
        )
        self._record(r.active_bank)
        r._validation.read(r._stream)
        for index in (0, 1):
            graph = tx.Graph(stream=r._stream, mode="cuda_stream")
            self.graphs.append(graph)
            with graph:
                self._record(index)
        self.ready = True

    def _record(self, bank):
        r = self.runtime
        for source, target in self.copies[bank]:
            target.assign(source, stream=r._stream)
        r._native._forward_runtime(self.handle)
        flags = r._validation.flags
        for name, limit in (
            ("nacon", self.data.naconmax),
            ("ncollision", self.data.naconmax),
            ("nefc", self.data.njmax),
        ):
            _capacity(
                getattr(self.data, name).view(tx.i32).flatten(),
                flags,
                limit,
                mode="cuda",
                stream=r._stream,
            )
        for name, source in self.sources.items():
            if source.size and name not in {"contact_geom", "contact_worldid", "nacon"}:
                _finite(source, flags, 4194304, mode="cuda", stream=r._stream)
        if self.contact_rows is not None:
            self.contact_rows.fill_(-1, stream=r._stream)
            self.contact_counts.zero_(stream=r._stream)
            _contact_rows(self.data.nacon, self.data.contact.worldid.flatten(), self.data.contact.geom.view(tx.i32).flatten(),
                          self.contact_rows.flatten(), self.contact_counts, flags, self.data.naconmax,
                          r._cpu_model.ngeom, mode="cuda", stream=r._stream)
            _publish_int(self.contact_rows.flatten(), self.logical_arrays["contact_pairs"].flatten(), flags, mode="cuda", stream=r._stream)
            _publish_int(self.contact_counts, self.logical_arrays["contact_valid_count"].flatten(), flags, mode="cuda", stream=r._stream)
        for name, source in self.sources.items():
            if source.size:
                if name in INTEGER_FIELDS:
                    _publish_int(source.view(tx.i32), self.targets[name].view(tx.i32), flags, mode="cuda", stream=r._stream)
                else:
                    _publish(source, self.targets[name], flags, mode="cuda", stream=r._stream)
        if "qM" in self.sources and self.logical_arrays["qM"].size:
            _dense_mass(
                self.sources["qM"],
                self.logical_arrays["qM"].flatten(),
                flags,
                self.mass_indices,
                self.mass_source_width,
                mode="cuda",
                stream=r._stream,
            )

    def refresh(self):
        r = self.runtime
        with r._scope():
            if not self.ready:
                raise RuntimeError("current-forward frame is closed or unprepared")
            if r._pending is not None and r._pending["phase"] == "completed":
                raise RuntimeError("current forward cannot append work after execution completion")
            if r._pending is None:
                self.idle_pending = True
            self.graphs[r.observation_bank].replay()
            self.refreshes += 1
        return self

    def complete(self):
        """Validate an idle refresh before beginning a physical transaction.

        Enqueued-candidate refreshes use the runtime's complete_execution gate.
        An idle observation error discards this sample, never a physical state.
        """
        r = self.runtime
        with r._scope():
            if not self.ready or not self.idle_pending or r._pending is not None:
                raise RuntimeError("completion requires an idle pending current-forward sample")
            try:
                r._validation.read(r._stream)
            except Exception as error:
                try:
                    r._complete()
                except Exception as cleanup:
                    r._closing = True
                    error.runtime = r
                    raise error from cleanup
                r._validation.clear(r._stream)
                self.idle_pending = False
                raise
            r._validation.clear(r._stream)
            self.idle_pending = False

    def close(self):
        r = self.runtime
        with r._lock:
            if self.closed:
                return
            self.ready = False
            try:
                with tx.use_device(r._device), tx.use_stream(r._stream):
                    r._complete()
                    while self.graphs:
                        self.graphs[-1].destroy()
                        self.graphs.pop()
                    if self.handle is not None:
                        r._native._close_runtime(self.handle, True)
                        self.handle = None
                    if self.idle_pending:
                        r._validation.clear(r._stream)
            except Exception as error:
                r._closing = True
                error.runtime = r
                raise
            self.idle_pending = False
            self.closed = True
            self.data = None
            self.arrays = MappingProxyType({})
            self.logical_arrays = MappingProxyType({})
            self.contact_rows = self.contact_counts = None
            self.mass_indices = None
            self.sources = self.targets = {}
            self.copies = ()
