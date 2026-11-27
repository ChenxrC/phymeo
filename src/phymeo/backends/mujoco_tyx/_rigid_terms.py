"""Bounded instantaneous FREE/HINGE/SLIDE rigid terms on TYX.

This component does not admit a model for integration. Equality constraints,
actuation, contact response, passive forces and integration remain separate.
Only static model fields are compiled on CPU. All state-dependent poses, body
COM Jacobians, physical M (including armature) and gravity/Coriolis bias are
computed on the selected device, with no native forward or CPU fallback.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType

import numpy as np

from ._mesh_query import _positive


def _split(value):
    value = np.asarray(value, dtype=np.float64)
    high = value.astype(np.float32)
    return np.stack((high, (value - high.astype(np.float64)).astype(np.float32)), axis=-1)


@dataclass(frozen=True)
class RigidModel:
    nbody: int
    nq: int
    nv: int
    ngeom: int
    nmocap: int
    nodes: int
    static: object


def compile_rigid_model(mj, model):
    """Compile original static transforms; reject unimplemented tree domains."""
    nb, nq, nv, ng, nm = map(int, (model.nbody, model.nq, model.nv, model.ngeom, model.nmocap))
    if not (1 <= nb <= 128 and 1 <= nq <= 128 and 1 <= nv <= 64 and ng <= 8192):
        raise ValueError("bounded rigid component dimensions exceeded")
    if np.any(model.body_jntnum > 1):
        raise ValueError("at most one joint per body is supported")
    if model.nflex or model.nplugin:
        raise ValueError("flex and plugin dynamics are outside this rigid component")
    if (
        np.any(model.body_mass < 0)
        or np.any(model.body_inertia < 0)
        or np.any(model.dof_armature < 0)
    ):
        raise ValueError("nonnegative physical mass, inertia and armature required")
    types = {
        int(mj.mjtJoint.mjJNT_HINGE): 1,
        int(mj.mjtJoint.mjJNT_SLIDE): 2,
        int(mj.mjtJoint.mjJNT_FREE): 3,
    }
    if any(int(t) not in types for t in model.jnt_type):
        raise ValueError("only FREE/HINGE/SLIDE joints are supported")
    if int(model.opt.disableflags) & int(mj.mjtDisableBit.mjDSBL_GRAVITY):
        raise ValueError("disabled gravity needs an explicit separate component contract")
    ids, local, joints = [], [], []

    def node(parent, kind=0, index=0, pos=(0, 0, 0), quat=(1, 0, 0, 0), axis=(0, 0, 0), ref=0):
        ids.append([parent, kind, index])
        local.append([*pos, *quat])
        joints.append([*axis, ref])
        return len(ids) - 1

    node(-1)
    bodies = np.zeros((nb, 5), np.int32)
    dofs = np.zeros((nv, 3), np.int32)
    ax = np.zeros((nv, 3, 3), np.float32)
    ancestor = np.zeros((nb, nv), np.int32)
    inertia = np.zeros((nb, 3, 3), np.float64)
    for b in range(1, nb):
        p = int(model.body_parentid[b])
        if not 0 <= p < b:
            raise ValueError("topologically ordered parent tree required")
        ancestor[b] = ancestor[p]
        j = int(model.body_jntadr[b])
        kind = types[int(model.jnt_type[j])] if model.body_jntnum[b] else 0
        qa = int(model.jnt_qposadr[j]) if kind else 0
        da = int(model.jnt_dofadr[j]) if kind else 0
        mc = int(model.body_mocapid[b])
        if (mc >= 0 or kind == 3) and p != 0:
            raise ValueError("FREE and mocap must be world children")
        if mc >= 0 and kind:
            raise ValueError("mocap bodies cannot have joints")
        if mc >= 0:
            bn = node(-1, 4, mc)
            kind = 4
        elif kind == 3:
            bn = node(-1, 3, qa)
            for k in range(6):
                dofs[da + k] = [int(k >= 3), b, bn if k >= 3 else 0]
                ax[da + k, k % 3, 0] = 1
            ancestor[b, da : da + 6] = 1
        else:
            pre = node(bodies[p, 0], pos=model.body_pos[b], quat=model.body_quat[b])
            bn = pre
            if kind:
                anchor = model.jnt_pos[j]
                axis = model.jnt_axis[j]
                bn = node(
                    pre,
                    kind,
                    qa,
                    pos=anchor if kind == 1 else (0, 0, 0),
                    axis=axis,
                    ref=model.qpos0[qa],
                )
                if kind == 1:
                    bn = node(bn, pos=-anchor)
                dofs[da] = [int(kind == 1), b, pre]
                ax[da, :, 0] = axis
                ax[da, :, 1:] = _split(anchor)
                ancestor[b, da] = 1
        bodies[b] = [bn, 0, p, kind, da]
        rotation = np.empty(9, np.float64)
        mj.mju_quat2Mat(rotation, model.body_iquat[b])
        rotation = rotation.reshape(3, 3)
        inertia[b] = (rotation * model.body_inertia[b]) @ rotation.T
    for b in range(nb):
        bodies[b, 1] = node(bodies[b, 0], pos=model.body_ipos[b])
    geom_nodes = np.array(
        [
            node(
                bodies[int(model.geom_bodyid[g]), 0], pos=model.geom_pos[g], quat=model.geom_quat[g]
            )
            for g in range(ng)
        ],
        np.int32,
    )
    products = np.array(
        [
            [(0, 0, 1), (1, 1, -1), (2, 2, -1), (3, 3, -1)],
            [(0, 1, 1), (1, 0, 1), (2, 3, 1), (3, 2, -1)],
            [(0, 2, 1), (1, 3, -1), (2, 0, 1), (3, 1, 1)],
            [(0, 3, 1), (1, 2, 1), (2, 1, -1), (3, 0, 1)],
        ],
        np.int32,
    )
    terms = [
        [(0, 0, 1), (1, 1, 1), (2, 2, -1), (3, 3, -1)],
        [(1, 2, 2), (0, 3, -2)],
        [(1, 3, 2), (0, 2, 2)],
        [(1, 2, 2), (0, 3, 2)],
        [(0, 0, 1), (1, 1, -1), (2, 2, 1), (3, 3, -1)],
        [(2, 3, 2), (0, 1, -2)],
        [(1, 3, 2), (0, 2, -2)],
        [(2, 3, 2), (0, 1, 2)],
        [(0, 0, 1), (1, 1, -1), (2, 2, -1), (3, 3, 1)],
    ]
    matrix = np.array([row + [(0, 0, 0)] * (4 - len(row)) for row in terms], np.int32)
    static = dict(
        ids=np.array(ids, np.int32),
        local=_split(local),
        joint=_split(joints),
        products=products,
        matrix=matrix,
        bodies=bodies,
        dofs=dofs,
        axes_local=ax,
        ancestor=ancestor,
        inertia=inertia.astype(np.float32),
        weight=np.array(model.body_mass, np.float32),
        armature=np.array(model.dof_armature, np.float32),
        gravity=np.array(model.opt.gravity, np.float32),
        geom_nodes=geom_nodes,
    )
    for value in static.values():
        if not np.isfinite(value).all():
            raise ValueError("finite static model required")
        value.setflags(write=False)
    return RigidModel(nb, nq, nv, ng, nm, len(ids), MappingProxyType(static))


@dataclass(frozen=True)
class RigidResult:
    workspace: object
    generation: int

    def read(self):
        return self.workspace._read(self)

    def device_view(self):
        return self.workspace._device_view(self)


class RigidWorkspace:
    """Reusable resident terms. Borrowed outputs expire at every new attempt.

    Packed qpos/mocap have last dimension [high, low] FP32. A device producer
    must retain the low lane; this API cannot recover previously lost bits.
    Synchronous borrowed views must not escape their context.
    """

    OUTPUT = (
        "positions",
        "positions_low",
        "rotations",
        "rotations_low",
        "com",
        "com_low",
        "body_jac",
        "mass",
        "bias",
        "qvel",
    )

    def __init__(
        self, model, *, worlds=1, mode="cuda", cache_root=None, memory_limit_bytes=256 * 1024**2
    ):
        if not isinstance(model, RigidModel):
            raise TypeError("compiled RigidModel required")
        if mode not in ("cuda", "interpret"):
            raise ValueError("explicit cuda or interpret mode required")
        import tyx as tx

        from . import _rigid_pose_kernels, _rigid_terms_kernels

        self.model, self.mode = model, mode
        self.worlds = w = _positive(worlds, "worlds", 256)
        n, b, d, g = model.nodes, model.nbody, model.nv, model.ngeom
        self.inputs = dict(qpos=(w, model.nq, 2), qvel=(w, d), mocap=(w, model.nmocap, 7, 2))
        layouts = {k: (s, np.float32) for k, s in self.inputs.items()}
        layouts.update({k: (v.shape, v.dtype) for k, v in model.static.items()})
        layouts.update(
            state=((w, n, 7, 2), np.float32),
            rotation=((w, n, 9, 2), np.float32),
            pose_scratch=((w, 7, 2), np.float32),
            positions=((w, g, 3), np.float32),
            positions_low=((w, g, 3), np.float32),
            rotations=((w, g, 3, 3), np.float32),
            rotations_low=((w, g, 3, 3), np.float32),
            com=((w, b, 3), np.float32),
            com_low=((w, b, 3), np.float32),
            axes=((w, d, 3), np.float32),
            anchors=((w, d, 3, 2), np.float32),
            body_jac=((w, b, 6, d), np.float32),
            motion=((w, b, 4, 3), np.float32),
            iw=((w, b, 3, 3), np.float32),
            force=((w, b, 6), np.float32),
            scratch=((w, 4, 3), np.float32),
            mass=((w, d, d), np.float32),
            bias=((w, d), np.float32),
            sampled_qvel=((w, d), np.float32),
            status=((w,), np.int32),
        )
        required = sum(max(1, int(np.prod(s))) * np.dtype(t).itemsize for s, t in layouts.values())
        if required > _positive(memory_limit_bytes, "memory_limit_bytes", 2**63 - 1):
            raise MemoryError(f"rigid workspace requires {required} bytes")
        self.stats = dict(
            owned_bytes=required, evaluations=0, launches=0, uploaded_bytes=0, downloaded_bytes=0
        )
        self._lock, self._generation, self._borrowed, self.closed = RLock(), 0, 0, False
        self._buffers = {}
        self.runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        self._pose, self._terms = _rigid_pose_kernels, _rigid_terms_kernels
        try:
            with self.runtime.activate():
                for name, (shape, dtype) in layouts.items():
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
            raise RuntimeError("rigid workspace closed")
        if self._borrowed:
            raise RuntimeError("rigid outputs borrowed")

    def evaluate(self, qpos, qvel, mocap_pos, mocap_quat):
        """Explicit diagnostic host upload; production consumers use evaluate_device."""
        with self._lock:
            self._ready()
            self._generation += 1
            w, m = self.worlds, self.model
            values = []
            for value, shape in (
                (qpos, (w, m.nq)),
                (qvel, (w, m.nv)),
                (mocap_pos, (w, m.nmocap, 3)),
                (mocap_quat, (w, m.nmocap, 4)),
            ):
                value = np.asarray(value)
                if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
                    raise ValueError("finite floating state with exact shape required")
                values.append(value.astype(np.float64, copy=True))
            q, v, p, r = values
            packed = dict(
                qpos=_split(q),
                qvel=v.astype(np.float32),
                mocap=_split(np.concatenate((p, r), axis=-1)),
            )
            with self.runtime.activate():
                for name, value in packed.items():
                    if self.mode == "cuda":
                        self._buffers[name].copy_from(value)
                        self.stats["uploaded_bytes"] += value.nbytes
                    else:
                        self._buffers[name][...] = value
            return self._evaluate({name: self._buffers[name] for name in self.inputs})

    def evaluate_device(self, inputs):
        import tyx as tx

        with self._lock:
            self._ready()
            self._generation += 1
            if set(inputs) != set(self.inputs):
                raise ValueError("exact resident rigid input schema required")
            for name, value in inputs.items():
                if self.mode == "cuda" and (
                    not isinstance(value, tx.cuda.Array) or value.device_ordinal != 0
                ):
                    raise ValueError("device 0 TYX CUDA arrays required")
                if tuple(value.shape) != self.inputs[name] or np.dtype(value.dtype) != np.dtype(
                    np.float32
                ):
                    raise ValueError(f"{name}: shape/dtype mismatch")
            return self._evaluate(inputs)

    def _evaluate(self, inputs):
        b = {**self._buffers, **inputs}
        m, w = self.model, self.worlds

        def args(*names):
            return tuple(b[k] for k in names)

        with self.runtime.activate():
            self._pose.poses(
                w,
                m.nodes,
                *args(
                    "ids",
                    "local",
                    "joint",
                    "products",
                    "matrix",
                    "qpos",
                    "mocap",
                    "state",
                    "rotation",
                    "pose_scratch",
                    "status",
                ),
                mode=self.mode,
            )
            self._pose.gather(
                w,
                m.ngeom,
                m.nbody,
                m.nv,
                *args(
                    "bodies",
                    "geom_nodes",
                    "dofs",
                    "axes_local",
                    "state",
                    "rotation",
                    "positions",
                    "positions_low",
                    "rotations",
                    "rotations_low",
                    "com",
                    "com_low",
                    "axes",
                    "anchors",
                ),
                mode=self.mode,
            )
            self._terms.mechanics(
                w,
                m.nbody,
                m.nv,
                *args(
                    "bodies",
                    "dofs",
                    "ancestor",
                    "inertia",
                    "weight",
                    "armature",
                    "gravity",
                    "qvel",
                    "state",
                    "rotation",
                    "axes",
                    "anchors",
                    "com",
                    "com_low",
                    "body_jac",
                    "motion",
                    "iw",
                    "force",
                    "scratch",
                    "mass",
                    "bias",
                    "sampled_qvel",
                    "status",
                ),
                mode=self.mode,
            )
            self.stats["launches"] += 3
            self.runtime.sync()
            status = self._download("status")
            if np.any(status):
                raise ValueError(f"invalid/unsupported rigid device state: {status.tolist()}")
        self._qvel = b["sampled_qvel"]
        self.stats["evaluations"] += 1
        return RigidResult(self, self._generation)

    def _download(self, name):
        value = self._qvel if name == "qvel" else self._buffers[name]
        result = value.to_numpy() if self.mode == "cuda" else value.copy()
        self.stats["downloaded_bytes"] += result.nbytes if self.mode == "cuda" else 0
        return result

    def _valid(self, result):
        if self.closed or result.workspace is not self or result.generation != self._generation:
            raise RuntimeError("rigid result expired")

    def _read(self, result):
        with self._lock:
            self._valid(result)
            return {name: self._download(name) for name in self.OUTPUT}

    @contextmanager
    def _device_view(self, result):
        with self._lock:
            self._ready()
            self._valid(result)
            self._borrowed += 1
            try:
                yield MappingProxyType(
                    {k: self._qvel if k == "qvel" else self._buffers[k] for k in self.OUTPUT}
                )
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
                    for value in self._buffers.values():
                        if hasattr(value, "free"):
                            value.free()

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *exc):
        self.close()
