"""Selected-world site projection of retained solve-time body measurements."""

import numpy as np
import tyx as tx

from ..builders.native_sites import NativeBodySites


@tx.func
def _rotate(q: tx.vec4f, v: tx.vec3f) -> tx.vec3f:
    axis = tx.vec3f(q[0], q[1], q[2])
    return v + 2.0 * tx.cross(axis, tx.cross(axis, v) + q[3] * v)


@tx.kernel(gradient="unsupported")
def _sample(
    selected: tx.Array[tx.i32],
    count: tx.Array[tx.i32],
    gate: tx.Array[tx.i32],
    native_error: tx.Array[tx.i32],
    bodies: tx.Array[tx.i32],
    anchors: tx.ndarray(tx.f32, 2),
    position: tx.ndarray(tx.f32, 3),
    rotation: tx.ndarray(tx.f32, 3),
    com: tx.ndarray(tx.f32, 3),
    specific: tx.ndarray(tx.f32, 3),
    alpha: tx.ndarray(tx.f32, 3),
    twist: tx.ndarray(tx.f32, 3),
    wrench: tx.ndarray(tx.f32, 3),
    output: tx.ndarray(tx.f32, 2),
    errors: tx.Array[tx.i32],
    rows: tx.i32,
):
    for i in tx.range(rows * bodies.shape[0]):
        if gate[0] == 0 and native_error[0] == 0:
            row, site = i // bodies.shape[0], i % bodies.shape[0]
            if row < count[0]:
                world, body = selected[row], bodies[site]
                # Selector validation belongs to the batch boundary; guard before any indexing.
                if world < 0 or world >= position.shape[0]:
                    tx.atomic_or(errors[0], 1)
                else:
                    q = tx.vec4f(
                        rotation[world, body, 1],
                        rotation[world, body, 2],
                        rotation[world, body, 3],
                        rotation[world, body, 0],
                    )
                    invq = tx.vec4f(-q[0], -q[1], -q[2], q[3])
                    local = tx.vec4f(
                        -anchors[site, 3], -anchors[site, 4], -anchors[site, 5], anchors[site, 6]
                    )
                    p = tx.vec3f(
                        position[world, body, 0], position[world, body, 1], position[world, body, 2]
                    )
                    c = tx.vec3f(com[world, body, 0], com[world, body, 1], com[world, body, 2])
                    r = (
                        p
                        + _rotate(q, tx.vec3f(anchors[site, 0], anchors[site, 1], anchors[site, 2]))
                        - c
                    )
                    a = tx.vec3f(
                        alpha[world, body, 0], alpha[world, body, 1], alpha[world, body, 2]
                    )
                    omega = tx.vec3f(
                        twist[world, body, 3], twist[world, body, 4], twist[world, body, 5]
                    )
                    f = tx.vec3f(
                        specific[world, body, 0], specific[world, body, 1], specific[world, body, 2]
                    )
                    f = _rotate(
                        local,
                        _rotate(invq, f + tx.cross(a, r) + tx.cross(omega, tx.cross(omega, r))),
                    )
                    gyro = _rotate(local, _rotate(invq, omega))
                    force = tx.vec3f(
                        wrench[world, body, 0], wrench[world, body, 1], wrench[world, body, 2]
                    )
                    torque = tx.vec3f(
                        wrench[world, body, 3], wrench[world, body, 4], wrench[world, body, 5]
                    ) - tx.cross(r, force)
                    force = _rotate(local, _rotate(invq, force))
                    torque = _rotate(local, _rotate(invq, torque))
                    for j in range(3):
                        output[i, j] = f[j]
                        output[i, j + 3] = gyro[j]
                        output[i, j + 6] = force[j]
                        output[i, j + 9] = torque[j]
                    for j in range(12):
                        if not tx.isfinite(output[i, j]):
                            tx.atomic_or(errors[0], 1)


class PreparedSiteMeasurements:
    """Owned projection output/scratch; caller retains until stream work completes.

    Values are [selected row, site, accelerometer/gyro/force/torque]. The wrench
    is parent-on-child at site origin in site axes. No physics is advanced.
    """

    contract_version = 1

    def __init__(self, runtime, binding):
        if (
            not isinstance(binding, NativeBodySites)
            or binding.contract_version != 1
            or binding.compilation is not runtime._compilation
        ):
            raise ValueError("site binding belongs to another compilation or contract")
        if runtime._body_measurements is None:
            raise RuntimeError("site projection requires body_measurements=True")
        if (
            not binding.sites
            or len(binding.sites) != len(binding.bodies)
            or any(type(i) is not int or not 0 <= i < runtime._model.nbody for i in binding.bodies)
        ):
            raise ValueError("site binding has invalid native body indices")
        self._closed = False
        self.runtime, self.binding = runtime, binding
        self.storage = runtime.storage_binding()
        self.resources = []
        self.bodies = self._array(np.asarray(binding.bodies, np.int32), tx.i32)
        self.anchors = self._array(
            np.asarray([(*s.position, *s.rotation_xyzw) for s in binding.sites], np.float32), tx.f32
        )
        self.values = tx.empty(
            (runtime.nworld * len(binding.sites), 12), tx.f32, device=self.storage.device
        )
        self.resources.append(self.values)
        self.errors = self._array(np.zeros(1, np.int32), tx.i32)
        fields = (
            "xpos",
            "xquat",
            "measured_body_com_position",
            "measured_body_specific_acceleration",
            "measured_body_angular_acceleration",
            "measured_body_com_twist",
            "measured_body_internal_wrench",
        )
        self.inputs = tuple(tuple(self.storage.borrow(f, bank) for f in fields) for bank in (0, 1))
        self.native_error = self.storage.borrow("execution_error", 0)

    def _array(self, values, dtype):
        array = tx.empty(values.shape, dtype, device=self.storage.device)
        self.resources.append(array)
        array.assign(values, stream=self.storage.stream)
        return array

    def sample(self, selected, count, gate, rows):
        """Enqueue a projection, returning no host result or completion wait.

        Caller prevalidates unique world IDs and count[0]==rows in the same
        stream, retains inputs, and excludes concurrent mutation/close. Inputs
        must not alias this plan's output/error resources. A nonzero gate
        suppresses work. Check errors after completion before publishing values.
        """
        self.storage.require_valid()
        if self._closed:
            raise RuntimeError("site measurement plan is closed")
        if type(rows) is not int or not 0 <= rows <= self.runtime.nworld:
            raise ValueError("site sample rows exceed prepared capacity")
        for array, size in ((selected, rows), (count, 1), (gate, 1)):
            if (
                array.dtype != tx.i32
                or array.ndim != 1
                or array.size < size
                or str(array.device) != self.storage.device
                or tuple(array.strides) != (4,)
            ):
                raise ValueError(
                    "site selection/count/gate require same-device contiguous int32 arrays"
                )
        stream = self.storage.stream
        self.errors.zero_(stream=stream)
        if rows:
            _sample(
                selected,
                count,
                gate,
                self.native_error,
                self.bodies,
                self.anchors,
                *self.inputs[self.runtime.observation_bank],
                self.values,
                self.errors,
                rows,
                mode="cuda",
                stream=stream,
            )

    @property
    def allocated_bytes(self):
        return sum(a.size * np.dtype(a.dtype.numpy_dtype).itemsize for a in self.resources)

    def close(self):
        if self.resources:
            self.storage.stream.synchronize()
        self.resources.clear()
        self.inputs = ()
        self.values = self.bodies = self.anchors = self.errors = None
        self._closed = True
