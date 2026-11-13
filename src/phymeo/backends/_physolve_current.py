"""Derived current-state forward frames; no integration or authoritative bank."""

from types import MappingProxyType

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
    }
)


@tx.kernel(gradient="unsupported")
def _publish(source: tx.Array[tx.f32], target: tx.Array[tx.f32], flags: tx.Array[tx.i32]):
    for i in tx.range(target.shape[0]):
        if flags[0] == 0:
            target[i] = source[i]


class CurrentForwardFrame:
    """One disposable native scratch workspace and two captured source bindings.

    Source integration inputs are copied on device. The shared physical model
    and accepted/candidate states belong exclusively to PhySolveRuntime.
    Outputs are provisional until that runtime's completion gate succeeds.
    This first contract requires the stream to remain fixed until frame close.
    """

    contract_version = 1
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
            name: getattr(self.data, name).view(tx.f32).flatten() for name in self.fields
        }
        self.arrays = MappingProxyType(
            {name: tx.zeros_like(getattr(self.data, name)) for name in self.fields}
        )
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
            if source.size:
                _finite(source, flags, 4194304, mode="cuda", stream=r._stream)
        for name, source in self.sources.items():
            if source.size:
                _publish(source, self.targets[name], flags, mode="cuda", stream=r._stream)

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
            self.sources = self.targets = {}
            self.copies = ()
