"""Owned PhySolve runtime with explicit candidate execution and publication.

Two persistent device banks make publication transactional. Synchronous helpers
use the same begin/enqueue/complete/accept contract. Completion reads a compact
device validation result; this is not Graph or P2PE C ABI qualification. CPU MuJoCo prepares
initial data. Native coordinates are preserved, including WXYZ free-root qpos.
"""

from contextlib import contextmanager
from numbers import Integral
from threading import RLock
from types import MappingProxyType

import numpy as np

from ..builders.mujoco_compiler import MuJoCoCompilation
from ..builders.mujoco_scene import CompiledMuJoCoWorld
from .physolve_initial import INTEGRATION_FIELDS, NativeInitialState
from .physolve_payload import PhySolvePayload

OBSERVATION_FIELDS = (
    "qpos",
    "qvel",
    "qacc",
    "time",
    "ctrl",
    "qfrc_applied",
    "qfrc_actuator",
    "qfrc_constraint",
    "xpos",
    "xquat",
)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class PhySolveRuntime:
    """Own model/banks/handles/stream; borrow the immutable compilation source.

    Contacts and xpos are solver evaluation-time results, not a separate
    endpoint mj_forward recomputation. reset restores compile-time initial
    state (no keyframe selected), then performs native forward.
    """

    def __init__(
        self,
        compilation,
        *,
        payload,
        nworld=1,
        nconmax=64,
        njmax=256,
        device="cuda:0",
        initial_state=None,
    ):
        world = compilation if isinstance(compilation, CompiledMuJoCoWorld) else None
        if world is not None:
            if initial_state is not None or world.binding_version != 1:
                raise ValueError("compiled world owns its versioned initial state")
            compilation = world.compilation
        if not isinstance(compilation, MuJoCoCompilation):
            raise TypeError("a Phymeo MuJoCoCompilation is required")
        if not isinstance(payload, PhySolvePayload):
            raise TypeError("an explicit frozen PhySolvePayload is required")
        self.nworld = _positive(nworld, "nworld")
        self.nconmax = _positive(nconmax, "nconmax")
        self.njmax = _positive(njmax, "njmax")
        import mujoco
        import physolve_tyx as frontend
        import tyx as tx

        # Initialize/validate before loading the native library or allocating GPU owners.
        if world is None:
            cpu = compilation.fresh_native_model()
            initial = mujoco.MjData(cpu)
            initial.qpos[:] = compilation.native_initial_qpos
            initial.qvel[:] = compilation.native_initial_qvel
        else:
            cpu, initial = world.fresh_native_data()
        if initial_state is not None and not isinstance(initial_state, NativeInitialState):
            raise TypeError("initial_state requires NativeInitialState")
        initial_state = initial_state or NativeInitialState.from_data(compilation, initial)
        initial_state.apply(compilation, initial)

        self._lock = RLock()
        self._tx, self._frontend = tx, frontend
        self._native = payload.load()
        self._closed, self._closing, self._active = False, False, 0
        self._banks, self._handles = [], []
        self._stream = self._model = None
        self._reset = None
        self._validation = self._storage = None
        self._storage_generation = 1
        self._pending = None
        self._execution_epoch = 0
        self._compilation = compilation
        self._cpu_model = cpu
        cpu = self._cpu_model
        if cpu.nuserdata or cpu.nplugin or cpu.nflex or cpu.nsensor:
            raise NotImplementedError(
                "PhySolve profile excludes userdata, plugins, flex and sensors"
            )
        if cpu.opt.integrator not in (
            mujoco.mjtIntegrator.mjINT_EULER, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        ):
            raise NotImplementedError("this direct qualification profile requires Euler or implicitfast")
        if cpu.opt.solver != mujoco.mjtSolver.mjSOL_NEWTON:
            raise NotImplementedError("PhySolve requires the Newton constraint solve algorithm")
        # This first profile deliberately fixes dense Jacobians; sparse capacity
        # accounting must be qualified before allowing a sparse request.
        if cpu.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE:
            raise NotImplementedError("sparse Jacobians are not qualified in this profile")
        cpu.opt.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
        selected = tx.get_device(device)
        if not selected.is_cuda or selected.arch != 120:
            raise RuntimeError("frozen payload is qualified only on CUDA sm_120")
        self._device = selected
        try:
            with tx.use_device(selected):
                self._stream = tx.CudaStream(device=selected.ordinal)
                with tx.use_stream(self._stream):
                    mujoco.mj_forward(cpu, initial)
                    if initial.ncon > self.nconmax or initial.nefc > self.njmax:
                        raise ValueError("initial model exceeds contact/constraint capacity")
                    self._model = frontend.put_model(cpu)
                    self._model.opt.graph_conditional = True
                    for _ in range(2):
                        data = frontend.put_data(
                            cpu, initial, nworld=self.nworld, nconmax=self.nconmax, njmax=self.njmax
                        )
                        self._banks.append(data)
                        # Each bank owns its frontend scratch before Host prepare/capture.
                        frontend.prepare_workspace(self._model, data)
                        handle = self._native._create_runtime(self._model, data, "tyx", "tyx")
                        self._handles.append(handle)
                        self._native._prepare_runtime(handle)
                        self._native._forward_runtime(handle)
                        self._validate(data)
                    from ._physolve_reset import PreparedReset

                    self._reset = PreparedReset(
                        self._banks, self.nworld, selected, OBSERVATION_FIELDS
                    )
                    self._reset.prepare(self._stream)
                    from ._physolve_binding import bind_storage
                    from ._physolve_validation import PreparedValidation

                    self._validation = PreparedValidation(self._banks, selected)
                    self._validation.prepare(self._stream)
                    self._storage = bind_storage(
                        self, self._banks, self._stream, selected, self._storage_generation
                    )
                    self._complete()
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup:
                error.runtime = self  # Retain owner for an explicit cleanup retry.
                raise error from cleanup
            raise

    @contextmanager
    def _scope(self):
        with self._lock:
            if self._closed or self._closing:
                raise RuntimeError("PhySolve runtime is closing or closed")
            with self._tx.use_device(self._device), self._tx.use_stream(self._stream):
                yield

    def _complete(self):
        for handle in self._handles:
            if handle is not None:
                self._native._synchronize_runtime_stream(
                    handle, int(self._device.context), int(self._stream.handle)
                )
                return
        if self._stream is not None:
            self._stream.synchronize()

    def _validate(self, data):
        if self._validation is not None:
            self._validation.read(self._stream)
            return
        # Cold construction validation before persistent device checks exist.
        self._complete()
        if (
            int(data.nacon.numpy()[0]) > data.naconmax
            or int(data.ncollision.numpy()[0]) > data.naconmax
            or np.any(data.nefc.numpy() > data.njmax)
        ):
            raise RuntimeError("PhySolve contact/constraint capacity exceeded; not published")
        for name in ("qpos", "qvel", "qacc", "time"):
            if not np.isfinite(getattr(data, name).numpy()).all():
                raise RuntimeError(f"non-finite PhySolve {name}; not published")

    def _candidate(self):
        index = 1 - self._active
        data, previous = self._banks[index], self._banks[self._active]
        for name in INTEGRATION_FIELDS:
            getattr(data, name).assign(getattr(previous, name), stream=self._stream)
        return index, data

    def _controls(self, controls):
        result = {}
        shapes = {
            "ctrl": (self.nworld, self._cpu_model.nu),
            "qfrc_applied": (self.nworld, self._cpu_model.nv),
        }
        for name, value in controls.items():
            if value is None:
                continue
            expected = shapes[name]
            array = np.asarray(value)
            if array.shape != expected or array.dtype.kind not in "fiu":
                raise ValueError(f"{name} requires numeric shape {expected}")
            with np.errstate(over="ignore", invalid="ignore"):
                array = np.array(array, dtype=np.float32, order="C", copy=True)
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite in float32")
            result[name] = array
        return result

    def step(self, *, ctrl=None, qfrc_applied=None):
        """Advance one compiled timestep; controls persist until changed/reset."""
        with self._scope():
            controls = self._controls(dict(ctrl=ctrl, qfrc_applied=qfrc_applied))
            with self._synchronous_execution() as ticket:
                for name, value in controls.items():
                    getattr(self._banks[ticket[1]], name).assign(value, stream=self._stream)
                self.enqueue_step(ticket)

    def forward(self):
        """Refresh derived values without advancing time or changing qpos/qvel."""
        with self._scope():
            with self._synchronous_execution() as ticket:
                self.enqueue_forward(ticket)

    def reset(self, mask=None):
        """Reset selected worlds; keep unselected integration and snapshot values."""
        with self._scope():
            self._require_idle()
            if mask is not None:
                mask = np.asarray(mask)
                if mask.dtype != np.bool_ or mask.shape != (self.nworld,):
                    raise ValueError("reset mask must be bool[nworld]")
                if not mask.any():
                    return
            with self._synchronous_execution() as ticket:
                self.enqueue_reset(ticket, mask)

    @contextmanager
    def _synchronous_execution(self):
        # Refuse someone else's pending transaction before entering cleanup scope.
        self._require_idle()
        try:
            ticket = self.begin_execution()
            yield ticket
            self.complete_execution(ticket)
            self.accept_execution(ticket)
        except Exception as primary:
            ticket = self.pending_execution
            if ticket is not None:
                try:
                    self.discard_execution(ticket)
                except Exception as cleanup:
                    raise primary from cleanup
            raise

    @property
    def storage_generation(self):
        return self._storage_generation if not self._closing else -1

    @property
    def active_bank(self):
        return self._active

    @property
    def pending_execution(self):
        """Recovery ticket, including a failure during candidate preparation."""
        return self._pending["ticket"] if self._pending else None

    def execution_costs(self):
        """Explicit validation transfers; native Host internal work is additional."""
        with self._scope():
            return {"validation_readbacks": self._validation.readbacks,
                    "validation_d2h_bytes": 4 * self._validation.readbacks,
                    "native_direct_conditional_sync": True}

    @property
    def writable_bank(self):
        pending = self._pending
        return pending["ticket"][1] if pending and pending["phase"] == "prepared" else None

    def storage_binding(self, *, contract_version=1):
        """Borrow prepared fields/stream; no native model, Data or Host escapes."""
        with self._scope():
            if contract_version != 1:
                raise ValueError("unsupported runtime storage contract")
            return self._storage

    def _require_idle(self):
        if self._pending is not None:
            raise RuntimeError("an execution is pending; accept or discard it first")

    def _require_ticket(self, ticket, phase=None):
        if self._pending is None or self._pending["ticket"] is not ticket:
            raise RuntimeError("stale or foreign execution ticket")
        if phase is not None and self._pending["phase"] != phase:
            raise RuntimeError(f"execution requires {phase} phase")
        return ticket[1]

    def begin_execution(self):
        """Copy integration once; caller may now enqueue candidate field writes."""
        with self._scope():
            self._require_idle()
            self._execution_epoch += 1
            ticket = (self._execution_epoch, 1 - self._active)
            self._pending = {"ticket": ticket, "phase": "preparing"}
            # On failure retain the candidate until discard/close confirms completion.
            self._validation.clear(self._stream)
            self._candidate()
            self._pending["phase"] = "prepared"
            return ticket

    def enqueue_step(self, ticket, *, steps=1):
        """Advance one candidate for all substeps; no publication or host readback here.

        Native direct conditional loops may synchronize internally; this API
        separates publication and does not promise a nonblocking Host solver.
        """
        steps = _positive(steps, "steps")
        with self._scope():
            index = self._require_ticket(ticket, "prepared")
            self._pending["phase"] = "enqueued"
            try:
                for _ in range(steps):
                    self._native._step_runtime(self._handles[index])
                    self._validation.enqueue(index, self._stream)
            except Exception:
                self._pending["phase"] = "failed"
                raise

    def enqueue_forward(self, ticket):
        with self._scope():
            index = self._require_ticket(ticket, "prepared")
            self._pending["phase"] = "enqueued"
            try:
                self._native._forward_runtime(self._handles[index])
                self._validation.enqueue(index, self._stream)
            except Exception:
                self._pending["phase"] = "failed"
                raise

    def enqueue_reset(self, ticket, mask=None):
        with self._scope():
            index = self._require_ticket(ticket, "prepared")
            if mask is not None:
                mask = np.asarray(mask)
                if mask.dtype != np.bool_ or mask.shape != (self.nworld,):
                    raise ValueError("reset mask must be bool[nworld]")
            self._pending["phase"] = "enqueued"
            try:
                self._reset.restore(index, mask, self._stream)
                self._native._forward_runtime(self._handles[index])
                # Error bits latch before unselected observations overwrite qacc.
                self._validation.enqueue(index, self._stream)
                if mask is not None and not mask.all():
                    self._reset.preserve_observations(index, self._stream)
            except Exception:
                self._pending["phase"] = "failed"
                raise

    def complete_execution(self, ticket):
        """Wait for the canonical stream and check 4 bytes; do not publish."""
        with self._scope():
            index = self._require_ticket(ticket, "enqueued")
            try:
                self._validate(self._banks[index])
            except Exception:
                self._pending["phase"] = "failed"
                raise
            self._pending["phase"] = "completed"

    def accept_execution(self, ticket):
        """Publish only this completed candidate, once; no GPU work."""
        with self._scope():
            index = self._require_ticket(ticket, "completed")
            self._active = index
            self._pending = None

    def discard_execution(self, ticket):
        """Keep the accepted bank; failed completion retains all owners for retry."""
        with self._scope():
            self._require_ticket(ticket)
            self._complete()
            self._pending = None

    def snapshot(self):
        """Independent read-only host arrays, explicitly synchronous D2H."""
        with self._scope():
            self._complete()
            data = self._banks[self._active]
            fields = {}
            for name in OBSERVATION_FIELDS:
                array = getattr(data, name).numpy(stream=self._stream)
                fields[name] = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(
                    array.shape
                )
            return MappingProxyType(fields)

    def close(self):
        """Complete before release; failed phase retains owners for retry."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            self._complete()
            for index, handle in enumerate(self._handles):
                if handle is not None:
                    self._native._close_runtime(handle, True)
                    self._handles[index] = None
            if self._stream is not None:
                self._stream.destroy()
                self._stream = None
            self._banks.clear()
            if self._storage is not None:
                self._storage._release()
            self._storage = self._validation = self._pending = None
            self._storage_generation += 1
            self._reset = None
            self._model = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
