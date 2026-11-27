"""Resident noncontact forces, free acceleration and original scalar rows.

This fixed-state component does not enable IMPLICITFAST, reset or training.
Native code is used only by the caller to compile static model fields. No
native forward, captured acceleration/row substitution or CPU solve is used.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

import numpy as np

from ._mesh_query import _positive
from ._rigid_terms import RigidModel, RigidResult, _split, compile_rigid_model


@dataclass(frozen=True)
class SmoothModel:
    rigid: RigidModel
    nu: int
    neq: int
    candidates: int
    capacity: int
    timestep: float
    static: object


def compile_smooth_model(mj, model, rigid):
    """Admit static semantics separately from full-engine integration."""
    if not isinstance(rigid, RigidModel):
        raise TypeError("compiled rigid model required")
    check = compile_rigid_model(mj, model)
    if any(
        getattr(check, k) != getattr(rigid, k)
        for k in ("nq", "nv", "nbody", "ngeom", "nmocap", "nodes")
    ) or any(not np.array_equal(v, rigid.static[k]) for k, v in check.static.items()):
        raise ValueError("original rigid model identity mismatch")
    if model.ntendon or model.na or model.nflex or model.nplugin or model.nu > 128:
        raise ValueError("activation/tendon/flex/plugin or actuator capacity unsupported")
    if model.opt.enableflags or model.opt.disableflags or model.opt.disableactuator:
        raise ValueError("default simulation and actuator-group flags required")
    if (
        model.opt.density
        or model.opt.viscosity
        or np.any(model.body_gravcomp)
        or mj.get_mjcb_passive() is not None
    ):
        raise ValueError("fluid/gravity-compensation/custom passive callback unsupported")
    if model.opt.timestep <= 0 or not np.isfinite(model.opt.timestep):
        raise ValueError("finite positive timestep required")
    if (
        np.any(model.dof_frictionloss < 0)
        or np.any(model.dof_damping < 0)
        or np.any(model.jnt_stiffness < 0)
    ):
        raise ValueError("negative passive/friction parameters unsupported")
    nv, nu = int(model.nv), int(model.nu)
    actors = np.zeros((nu, 4), np.int32)
    params = np.zeros((nu, 9))
    scalar = (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE))
    dofs = np.zeros((nv, 2), np.int32)
    dofs[:, 0] = -1
    passive = np.zeros((nv, 5))
    passive[:, 0] = model.dof_damping
    for j in range(model.njnt):
        qa, da = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        if int(model.jnt_type[j]) not in scalar:
            if model.jnt_stiffness[j] or model.jnt_limited[j] or model.jnt_actfrclimited[j]:
                raise ValueError("FREE spring/limit/total clamp unsupported")
            continue
        dofs[da] = [qa, int(model.jnt_actfrclimited[j])]
        passive[da, 1:3] = [model.jnt_stiffness[j], model.qpos_spring[qa]]
        passive[da, 3:5] = model.jnt_actfrcrange[j]
        if model.jnt_actfrclimited[j] and passive[da, 3] > passive[da, 4]:
            raise ValueError("reversed joint force range")
    for a in range(nu):
        j = int(model.actuator_trnid[a, 0])
        if (
            int(model.actuator_trntype[a]) != int(mj.mjtTrn.mjTRN_JOINT)
            or not 0 <= j < model.njnt
            or int(model.jnt_type[j]) not in scalar
        ):
            raise ValueError("only scalar joint actuator transmissions supported")
        if (
            model.actuator_dyntype[a] != 0
            or model.actuator_gaintype[a] != 0
            or model.actuator_biastype[a] not in (0, 1)
            or np.any(model.actuator_gear[a, 1:] != 0)
        ):
            raise ValueError("fixed gain, none/affine bias, no activation required")
        actors[a] = [
            model.jnt_qposadr[j],
            model.jnt_dofadr[j],
            model.actuator_ctrllimited[a],
            model.actuator_forcelimited[a],
        ]
        params[a, 0:2] = [model.actuator_gear[a, 0], model.actuator_gainprm[a, 0]]
        if model.actuator_biastype[a] == 1:
            params[a, 2:5] = model.actuator_biasprm[a, :3]
        params[a, 5:7] = model.actuator_ctrlrange[a]
        params[a, 7:9] = model.actuator_forcerange[a]
        if (actors[a, 2] and params[a, 5] > params[a, 6]) or (
            actors[a, 3] and params[a, 7] > params[a, 8]
        ):
            raise ValueError("reversed actuator range")
    ids = []
    data = []

    def append(kind, owner, qa, da, qb, db, law, ref, imp):
        if (ref[0] > 0) != (ref[1] > 0):
            raise ValueError("mixed-sign solref unsupported")
        law[12:14] = ref
        law[14:19] = imp
        ids.append([kind, owner, qa, da, qb, db])
        data.append(law)

    for e in range(model.neq):
        if int(model.eq_type[e]) != int(mj.mjtEq.mjEQ_JOINT):
            raise ValueError("only scalar joint polynomial equality supported")
        a, b = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if (
            not 0 <= a < model.njnt
            or b < -1
            or b >= model.njnt
            or b == a
            or int(model.jnt_type[a]) not in scalar
            or (b >= 0 and int(model.jnt_type[b]) not in scalar)
        ):
            raise ValueError("invalid scalar equality object")
        qa, da = int(model.jnt_qposadr[a]), int(model.jnt_dofadr[a])
        qb, db = (-1, -1) if b < 0 else (int(model.jnt_qposadr[b]), int(model.jnt_dofadr[b]))
        law = np.zeros(19)
        law[:5] = model.eq_data[e, :5]
        law[5] = model.qpos0[qa]
        law[10] = model.dof_invweight0[da]
        if b >= 0:
            law[6] = model.qpos0[qb]
            law[10] += model.dof_invweight0[db]
        append(0, e, qa, da, qb, db, law, model.eq_solref[e], model.eq_solimp[e])
    for d in np.flatnonzero(model.dof_frictionloss > 0):
        law = np.zeros(19)
        law[10] = model.dof_invweight0[d]
        law[11] = model.dof_frictionloss[d]
        append(1, int(d), -1, int(d), -1, -1, law, model.dof_solref[d], model.dof_solimp[d])
    for j in np.flatnonzero(model.jnt_limited):
        qa, da = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        for side, sign in ((0, 1), (1, -1)):
            law = np.zeros(19)
            law[7] = model.jnt_margin[j]
            law[8] = model.jnt_range[j, side]
            law[9] = sign
            law[10] = model.dof_invweight0[da]
            append(3, int(j), qa, da, -1, -1, law, model.jnt_solref[j], model.jnt_solimp[j])
    count = len(ids)
    cap = max(1, count)
    if count > 128:
        raise ValueError("scalar capacity exceeds 128")
    static = dict(
        actors=actors,
        params=_split(params),
        dofs=dofs,
        passive_params=_split(passive),
        row_ids=np.asarray(ids, np.int32).reshape(count, 6),
        row_data=_split(np.asarray(data).reshape(count, 19)),
    )
    for v in static.values():
        if not np.isfinite(v).all():
            raise ValueError("finite static parameters required")
        v.setflags(write=False)
    return SmoothModel(
        rigid, nu, int(model.neq), count, cap, float(model.opt.timestep), MappingProxyType(static)
    )


@dataclass(frozen=True)
class SmoothResult:
    workspace: object
    generation: int

    def read(self):
        return self.workspace._read(self)

    def device_view(self):
        return self.workspace._device_view(self)


class SmoothWorkspace:
    """Owned snapshots and scoped device loans; no stale upstream aliases.

    Forces are world-coordinate wrenches at each body COM, with force before
    torque (MuJoCo xfrc_applied convention). Inputs are packed high/low FP32
    except integer eq_active. Only status words are downloaded in device mode.
    """

    OUTPUT = (
        "mass",
        "mass_low",
        "smooth",
        "smooth_low",
        "scalar_counts",
        "sj",
        "sa",
        "sd",
        "sl",
        "sk",
        "owner",
        "position",
        "margin",
        "velocity",
        "R",
        "actuator_force",
        "actuator",
        "actuator_low",
        "passive",
        "external",
        "rhs",
        "rhs_low",
        "balance",
        "status",
    )

    def __init__(
        self, model, *, worlds=1, mode="cuda", cache_root=None, memory_limit_bytes=256 * 1024**2
    ):
        if not isinstance(model, SmoothModel):
            raise TypeError("compiled SmoothModel required")
        if mode not in ("cuda", "interpret"):
            raise ValueError("explicit cuda/interpret mode required")
        import tyx as tx

        from . import _scalar_terms_kernels as sk
        from . import _smooth_force_kernels as fk

        self.model, self.mode = model, mode
        self.worlds = w = _positive(worlds, "worlds", 256)
        d, b, u, e, c = model.rigid.nv, model.rigid.nbody, model.nu, model.neq, model.capacity
        self.inputs = dict(
            ctrl=((w, u, 2), np.float32),
            applied=((w, d, 2), np.float32),
            body_force=((w, b, 6, 2), np.float32),
            eq_active=((w, e), np.int32),
        )
        self.layouts = {k: (v.shape, v.dtype) for k, v in model.static.items()}
        self.layouts.update(self.inputs)
        self.layouts.update(
            mass=((w, d, d), np.float32),
            mass_low=((w, d, d), np.float32),
            factor=((w, d, d), np.float32),
            actuator_force=((w, u), np.float32),
            scalar_counts=((w,), np.int32),
            sj=((w, c, d), np.float32),
            sk=((w, c), np.int32),
            owner=((w, c), np.int32),
            balance=((w,), np.float32),
            status=((w,), np.int32),
        )
        for k in (
            "smooth",
            "smooth_low",
            "forward",
            "correction",
            "actuator",
            "actuator_low",
            "passive",
            "external",
            "rhs",
            "rhs_low",
        ):
            self.layouts[k] = ((w, d), np.float32)
        for k in ("sa", "sd", "sl", "position", "margin", "velocity", "R"):
            self.layouts[k] = ((w, c), np.float32)
        required = sum(
            max(1, int(np.prod(s))) * np.dtype(t).itemsize for s, t in self.layouts.values()
        )
        if required > _positive(memory_limit_bytes, "memory_limit_bytes", 2**63 - 1):
            raise MemoryError(f"smooth workspace requires {required} bytes")
        self.stats = dict(
            owned_bytes=required, evaluations=0, launches=0, uploaded_bytes=0, downloaded_bytes=0
        )
        self._lock, self._generation, self._borrowed, self.closed = RLock(), 0, 0, False
        self._buffers = {}
        self.runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        self._force, self._rows = fk, sk
        try:
            with self.runtime.activate():
                for name, (shape, dtype) in self.layouts.items():
                    value = (
                        model.static[name].copy()
                        if name in model.static
                        else np.zeros(shape, dtype)
                    )
                    self._buffers[name] = tx.cuda.array(value) if mode == "cuda" else value
                    self.stats["uploaded_bytes"] += value.nbytes if mode == "cuda" else 0
        except BaseException:
            self.close()
            raise

    def _ready(self):
        if self.closed:
            raise RuntimeError("smooth workspace closed")
        if self._borrowed:
            raise RuntimeError("smooth outputs borrowed")

    def evaluate(self, rigid, ctrl, applied, body_force, eq_active):
        """Explicit host convenience upload of controls and applied forces only."""
        with self._lock:
            self._ready()
            self._generation += 1
            for name, value in dict(
                ctrl=ctrl, applied=applied, body_force=body_force, eq_active=eq_active
            ).items():
                source = np.asarray(value)
                shape, dtype = self.inputs[name]
                expected = shape if name == "eq_active" else shape[:-1]
                if source.shape != expected or not np.isfinite(source).all():
                    raise ValueError(f"{name}: finite exact shape required")
                if name == "eq_active":
                    if source.dtype.kind not in "biu" or np.any((source != 0) & (source != 1)):
                        raise ValueError("binary equality activation required")
                    packed = source.astype(np.int32)
                else:
                    if source.dtype.kind != "f":
                        raise ValueError("floating controls/forces required")
                    packed = _split(source)
                with self.runtime.activate():
                    if self.mode == "cuda":
                        self._buffers[name].copy_from(packed)
                        self.stats["uploaded_bytes"] += packed.nbytes
                    else:
                        self._buffers[name][...] = packed
            return self._evaluate(rigid, {k: self._buffers[k] for k in self.inputs})

    def evaluate_device(self, rigid, inputs):
        import tyx as tx

        with self._lock:
            self._ready()
            self._generation += 1
            if set(inputs) != set(self.inputs):
                raise ValueError("exact smooth input schema required")
            for name, value in inputs.items():
                shape, dtype = self.inputs[name]
                if self.mode == "cuda" and (
                    not isinstance(value, tx.cuda.Array) or value.device_ordinal != 0
                ):
                    raise ValueError("same-device TYX CUDA arrays required")
                if tuple(value.shape) != shape or np.dtype(value.dtype) != np.dtype(dtype):
                    raise ValueError(f"{name}: shape/dtype mismatch")
            return self._evaluate(rigid, inputs)

    def _evaluate(self, rigid, inputs):
        if (
            not isinstance(rigid, RigidResult)
            or rigid.workspace.model is not self.model.rigid
            or rigid.workspace.mode != self.mode
            or rigid.workspace.worlds != self.worlds
        ):
            raise ValueError("same original rigid model/worlds/device required")
        m, w = self.model, self.worlds
        d, b = m.rigid.nv, m.rigid.nbody
        bank = {**self._buffers, **inputs}

        def args(*names):
            return tuple(bank[k] for k in names)

        with rigid.device_view() as r, self.runtime.activate():
            self._force.forces(
                w,
                d,
                b,
                m.nu,
                *args("actors", "params", "dofs", "passive_params"),
                r["qpos"],
                r["qvel"],
                r["body_jac"],
                r["body_jac_low"],
                r["bias"],
                r["bias_low"],
                *args(
                    "ctrl",
                    "applied",
                    "body_force",
                    "actuator_force",
                    "actuator",
                    "actuator_low",
                    "passive",
                    "external",
                    "rhs",
                    "rhs_low",
                    "status",
                ),
                mode=self.mode,
            )
            self._force.smooth_solve(
                w,
                d,
                r["mass"],
                r["mass_low"],
                *args(
                    "rhs",
                    "rhs_low",
                    "mass",
                    "mass_low",
                    "factor",
                    "forward",
                    "correction",
                    "smooth",
                    "smooth_low",
                    "balance",
                    "status",
                ),
                mode=self.mode,
            )
            self._rows.rows(
                w,
                d,
                m.candidates,
                m.capacity,
                m.neq,
                *args("row_ids", "row_data"),
                m.timestep,
                r["qpos"],
                r["qvel"],
                *args(
                    "eq_active",
                    "scalar_counts",
                    "sj",
                    "sa",
                    "sd",
                    "sl",
                    "sk",
                    "owner",
                    "position",
                    "margin",
                    "velocity",
                    "R",
                    "status",
                ),
                mode=self.mode,
            )
            self.stats["launches"] += 3
            self.runtime.sync()
            status = self._download("status")
            if np.any(status):
                raise ValueError(f"invalid smooth/scalar device state: {status.tolist()}")
        self.stats["evaluations"] += 1
        return SmoothResult(self, self._generation)

    def _download(self, name):
        a = self._buffers[name]
        value = a.to_numpy() if self.mode == "cuda" else a.copy()
        self.stats["downloaded_bytes"] += value.nbytes if self.mode == "cuda" else 0
        return value

    def _valid(self, result):
        if self.closed or result.workspace is not self or result.generation != self._generation:
            raise RuntimeError("smooth result expired")

    def _read(self, result):
        with self._lock:
            self._valid(result)
            return {k: self._download(k) for k in self.OUTPUT}

    @contextmanager
    def _device_view(self, result):
        with self._lock:
            self._ready()
            self._valid(result)
            self._borrowed += 1
            try:
                yield MappingProxyType({k: self._buffers[k] for k in self.OUTPUT})
            finally:
                try:
                    self.runtime.sync()
                finally:
                    self._borrowed -= 1

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
