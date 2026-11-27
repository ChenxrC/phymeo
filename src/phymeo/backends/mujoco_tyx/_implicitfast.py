"""Experimental original IMPLICITFAST with explicit process-local commit.

The resident velocity contract is FP32; qpos, time, M and integration acceleration
retain high/low lanes. This is not a complete engine, reset or learning adapter.
Raw device producers must share the caller's state epoch and scoped loans.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

import numpy as np

from ._mesh_query import _positive
from ._rigid_terms import _split
from ._smooth_terms import SmoothModel, compile_smooth_model


@dataclass(frozen=True)
class ImplicitModel:
    smooth: SmoothModel
    timestep: tuple
    static: object


def compile_implicit_model(mj, model, smooth):
    if not isinstance(smooth, SmoothModel):
        raise TypeError("compiled SmoothModel required")
    check = compile_smooth_model(mj, model, smooth.rigid)
    if check.timestep != smooth.timestep or any(
        not np.array_equal(v, smooth.static[k]) for k, v in check.static.items()
    ):
        raise ValueError("original smooth model identity mismatch")
    if int(model.opt.integrator) != int(mj.mjtIntegrator.mjINT_IMPLICITFAST):
        raise ValueError("original IMPLICITFAST required")
    derivative = -model.dof_damping.copy()
    for a in range(model.nu):
        if model.actuator_biastype[a] == int(mj.mjtBias.mjBIAS_AFFINE):
            j = int(model.actuator_trnid[a, 0])
            d = int(model.jnt_dofadr[j])
            # Native mjd_actuator_vel does not differentiate the force clamps.
            derivative[d] += model.actuator_gear[a, 0] ** 2 * model.actuator_biasprm[a, 2]
    if not np.isfinite(derivative).all() or np.any(derivative > 0):
        raise ValueError("bounded implicit component requires nonpositive velocity derivative")
    joints = np.column_stack((model.jnt_type, model.jnt_qposadr, model.jnt_dofadr)).astype(np.int32)
    static = dict(
        derivative=_split(derivative),
        joints=joints,
        products=smooth.rigid.static["products"].copy(),
    )
    for v in static.values():
        v.setflags(write=False)
    return ImplicitModel(
        smooth, tuple(map(float, _split(model.opt.timestep))), MappingProxyType(static)
    )


@dataclass(frozen=True)
class StepCandidate:
    workspace: object
    attempt: int
    source_generation: int

    def read(self):
        return self.workspace._read_candidate(self)

    def device_view(self):
        return self.workspace._candidate_view(self)


class ImplicitWorkspace:
    """Prepare never overwrites active state; only commit changes its generation.

    Commit is whole-batch and process-local. It is not persistence or rollback of
    already executed physical steps. A failed candidate expires; active state is
    preserved. The caller must bind mass/force and recorded evidence to the same
    state generation; shape checks alone cannot establish producer provenance.
    """

    def __init__(
        self, model, *, worlds=1, mode="cuda", cache_root=None, memory_limit_bytes=256 * 1024**2
    ):
        if not isinstance(model, ImplicitModel):
            raise TypeError("compiled ImplicitModel required")
        if mode not in ("cuda", "interpret"):
            raise ValueError("explicit cuda/interpret mode required")
        import tyx as tx

        from . import _implicitfast_kernels as kernels
        from . import _smooth_force_kernels as solve

        self.model, self.mode, self.worlds = model, mode, _positive(worlds, "worlds", 256)
        w, n, q = self.worlds, model.smooth.rigid.nv, model.smooth.rigid.nq
        self.inputs = dict(
            mass=((w, n, n), np.float32),
            mass_low=((w, n, n), np.float32),
            force=((w, n), np.float32),
            force_low=((w, n), np.float32),
        )
        self.state_layout = dict(
            qpos=((w, q, 2), np.float32), qvel=((w, n), np.float32), time=((w, 2), np.float32)
        )
        layouts = {k: (v.shape, v.dtype) for k, v in model.static.items()}
        for k in ("modified", "modified_low", "solve_mass", "solve_mass_low", "factor"):
            layouts[k] = ((w, n, n), np.float32)
        for k in (
            "rhs",
            "rhs_low",
            "forward",
            "correction",
            "acc",
            "acc_low",
            "total",
            "total_low",
        ):
            layouts[k] = ((w, n), np.float32)
        layouts.update(
            status=((w,), np.int32), balance=((w,), np.float32), scratch=((w, 11, 2), np.float32)
        )
        for slot in range(2):
            layouts.update({f"{k}{slot}": v for k, v in self.state_layout.items()})
        required = sum(max(1, int(np.prod(s))) * np.dtype(t).itemsize for s, t in layouts.values())
        if required > _positive(memory_limit_bytes, "memory_limit_bytes", 2**63 - 1):
            raise MemoryError(f"implicit workspace requires {required} bytes")
        self.stats = dict(
            owned_bytes=required, attempted=0, committed=0, discarded=0, status_download_bytes=0
        )
        self.generation = 0
        self._attempt = 0
        self._active = 0
        self._state_loans = 0
        self._candidate_loans = 0
        self._lock = RLock()
        self.closed = False
        self.initialized = False
        self._pending = None
        self._buffers = {}
        self.kernels, self.solve = kernels, solve
        self.runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        try:
            with self.runtime.activate():
                for k, (s, t) in layouts.items():
                    v = model.static[k].copy() if k in model.static else np.zeros(s, t)
                    self._buffers[k] = tx.cuda.array(v) if mode == "cuda" else v
        except BaseException:
            self.close()
            raise

    def _ready(self, *, prepare=False):
        if self.closed:
            raise RuntimeError("implicit workspace closed")
        if self._candidate_loans or (self._state_loans and not prepare):
            raise RuntimeError("implicit state/candidate borrowed")

    def initialize(self, qpos, qvel, time):
        with self._lock:
            self._ready()
            if self.initialized:
                raise RuntimeError("already initialized; explicit reset adapter required")
            values = {}
            for k, source in dict(qpos=qpos, qvel=qvel, time=time).items():
                v = np.asarray(source)
                shape, _ = self.state_layout[k]
                expected = shape if k == "qvel" else shape[:-1]
                if v.shape != expected or v.dtype.kind != "f" or not np.isfinite(v).all():
                    raise ValueError(f"{k}: finite floating exact shape required")
                packed = v.astype(np.float32) if k == "qvel" else _split(v)
                if not np.isfinite(packed).all():
                    raise ValueError("state outside representable domain")
                values[k] = packed
            with self.runtime.activate():
                for k, v in values.items():
                    if self.mode == "cuda":
                        self._buffers[k + "0"].copy_from(v)
                    else:
                        self._buffers[k + "0"][...] = v
            self.initialized = True

    def _state(self, slot):
        return {k: self._buffers[k + str(slot)] for k in self.state_layout}

    def _download(self, k):
        a = self._buffers[k]
        return a.to_numpy() if self.mode == "cuda" else a.copy()

    def read_state(self):
        with self._lock:
            if self.closed or not self.initialized:
                raise RuntimeError("no live initialized state")
            return dict(
                generation=self.generation,
                **{k: self._download(k + str(self._active)) for k in self.state_layout},
            )

    @contextmanager
    def state_view(self):
        with self._lock:
            self._ready()
            if not self.initialized:
                raise RuntimeError("state not initialized")
            self._state_loans += 1
            try:
                yield MappingProxyType(
                    dict(generation=self.generation, **self._state(self._active))
                )
            finally:
                try:
                    self.runtime.sync()
                finally:
                    self._state_loans -= 1

    def _begin(self, expected_generation):
        self._ready(prepare=True)
        self._attempt += 1
        self._pending = None
        self.stats["attempted"] += 1
        if not self.initialized:
            raise RuntimeError("state not initialized")
        if type(expected_generation) is not int or expected_generation != self.generation:
            raise RuntimeError("expired state generation")

    def _validate_inputs(self, inputs):
        import tyx as tx

        if set(inputs) != set(self.inputs):
            raise ValueError("exact implicit input schema required")
        for k, a in inputs.items():
            shape, dtype = self.inputs[k]
            if self.mode == "cuda" and (not isinstance(a, tx.cuda.Array) or a.device_ordinal != 0):
                raise ValueError("same-device CUDA arrays required")
            if tuple(a.shape) != shape or np.dtype(a.dtype) != np.dtype(dtype):
                raise ValueError(f"{k}: shape/dtype mismatch")

    def prepare_device(self, inputs, *, expected_generation):
        with self._lock:
            self._begin(expected_generation)
            self._validate_inputs(inputs)
            return self._prepare(inputs)

    def prepare_coupled(self, smooth, contacts, acceleration, *, expected_generation):
        """Consume original resident force outputs under scoped producer loans.

        Caller must have computed these results in one current state_view scope.
        Numerical convergence is checked before acceleration buffers are loaned.
        """
        with self._lock:
            self._begin(expected_generation)
            if smooth.workspace.model is not self.model.smooth:
                raise ValueError("same compiled smooth model required")
            for result in (smooth, contacts, acceleration):
                if result.workspace.worlds != self.worlds or result.workspace.mode != self.mode:
                    raise ValueError("producer world/device mismatch")
            if acceleration.workspace.nv != self.model.smooth.rigid.nv:
                raise ValueError("producer nv mismatch")
            if (
                acceleration.workspace.capacity != contacts.workspace.capacity
                or acceleration.workspace.scalar_capacity != self.model.smooth.capacity
            ):
                raise ValueError("producer capacity mismatch")
            with (
                smooth.device_view() as s,
                contacts.device_view() as c,
                acceleration.device_view() as a,
                self.runtime.activate(),
            ):
                w, n = self.worlds, self.model.smooth.rigid.nv
                self.kernels.coupled_force(
                    w,
                    n,
                    contacts.workspace.capacity,
                    self.model.smooth.capacity,
                    c["counts"],
                    a["result_counts"],
                    c["jac"],
                    a["force"],
                    s["scalar_counts"],
                    s["sj"],
                    s["sa"],
                    s["sd"],
                    s["sl"],
                    s["sk"],
                    a["acc"],
                    a["acc_low"],
                    s["rhs"],
                    s["rhs_low"],
                    self._buffers["total"],
                    self._buffers["total_low"],
                    mode=self.mode,
                )
                inputs = dict(
                    mass=s["mass"],
                    mass_low=s["mass_low"],
                    force=self._buffers["total"],
                    force_low=self._buffers["total_low"],
                )
                self._validate_inputs(inputs)
                return self._prepare(inputs)

    def _prepare(self, inputs):
        m = self.model
        w = self.worlds
        n = m.smooth.rigid.nv
        q = m.smooth.rigid.nq
        b = self._buffers
        state = self._state(self._active)
        candidate = self._state(1 - self._active)

        def args(*names):
            return tuple(b[k] for k in names)

        with self.runtime.activate():
            self.kernels.system(
                w,
                n,
                *m.timestep,
                b["derivative"],
                inputs["mass"],
                inputs["mass_low"],
                inputs["force"],
                inputs["force_low"],
                *args("modified", "modified_low", "rhs", "rhs_low", "status"),
                mode=self.mode,
            )
            self.solve.smooth_solve(
                w,
                n,
                *args(
                    "modified",
                    "modified_low",
                    "rhs",
                    "rhs_low",
                    "solve_mass",
                    "solve_mass_low",
                    "factor",
                    "forward",
                    "correction",
                    "acc",
                    "acc_low",
                    "balance",
                    "status",
                ),
                mode=self.mode,
            )
            self.kernels.advance(
                w,
                q,
                n,
                len(m.static["joints"]),
                *m.timestep,
                b["joints"],
                b["products"],
                state["qpos"],
                state["qvel"],
                state["time"],
                b["acc"],
                b["acc_low"],
                candidate["qpos"],
                candidate["qvel"],
                candidate["time"],
                b["scratch"],
                b["status"],
                mode=self.mode,
            )
            self.runtime.sync()
            status = self._download("status")
            self.stats["status_download_bytes"] += status.nbytes if self.mode == "cuda" else 0
        if np.any(status):
            raise ValueError(f"invalid implicit candidate: {status.tolist()}")
        result = StepCandidate(self, self._attempt, self.generation)
        self._pending = result
        return result

    def _valid(self, candidate):
        if (
            self.closed
            or candidate is not self._pending
            or candidate.attempt != self._attempt
            or candidate.source_generation != self.generation
        ):
            raise RuntimeError("implicit candidate expired")

    def commit(self, candidate):
        with self._lock:
            self._ready()
            self._valid(candidate)
            self._active = 1 - self._active
            self.generation += 1
            self._pending = None
            self.stats["committed"] += 1
            return self.generation

    def discard(self, candidate):
        with self._lock:
            self._ready()
            self._valid(candidate)
            self._pending = None
            self.stats["discarded"] += 1

    def _read_candidate(self, candidate):
        with self._lock:
            self._valid(candidate)
            return dict(
                source_generation=candidate.source_generation,
                **{k: self._download(k + str(1 - self._active)) for k in self.state_layout},
                **{
                    k: self._download(k)
                    for k in (
                        "modified",
                        "modified_low",
                        "rhs",
                        "rhs_low",
                        "acc",
                        "acc_low",
                        "balance",
                        "status",
                    )
                },
            )

    @contextmanager
    def _candidate_view(self, candidate):
        with self._lock:
            self._valid(candidate)
            self._candidate_loans += 1
            try:
                yield MappingProxyType(self._state(1 - self._active))
            finally:
                try:
                    self.runtime.sync()
                finally:
                    self._candidate_loans -= 1

    def close(self):
        with self._lock:
            if not self.closed:
                self._ready()
                self.closed = True
                try:
                    self.runtime.close()
                finally:
                    for a in self._buffers.values():
                        if hasattr(a, "free"):
                            a.free()

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *exc):
        self.close()
