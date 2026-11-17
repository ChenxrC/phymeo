"""Prepared world-origin pose / world-COM twist to native FREE coordinates."""

import numpy as np
import tyx as tx


@tx.func
def _rotate(
    x: tx.f32, y: tx.f32, z: tx.f32, w: tx.f32, a: tx.f32, b: tx.f32, c: tx.f32, axis: tx.i32
) -> tx.f32:
    result = (
        (1.0 - 2.0 * (y * y + z * z)) * a + 2.0 * (x * y - z * w) * b + 2.0 * (x * z + y * w) * c
    )
    if axis == 1:
        result = (
            2.0 * (x * y + z * w) * a
            + (1.0 - 2.0 * (x * x + z * z)) * b
            + 2.0 * (y * z - x * w) * c
        )
    if axis == 2:
        result = (
            2.0 * (x * z - y * w) * a
            + 2.0 * (y * z + x * w) * b
            + (1.0 - 2.0 * (x * x + y * y)) * c
        )
    return result


@tx.kernel(gradient="unsupported")
def _world_mask(
    selected: tx.Array[tx.bool1],
    worlds: tx.Array[tx.bool1],
    count: tx.i32,
    extra: tx.Array[tx.bool1],
    has_extra: tx.i32,
):
    for world in tx.range(worlds.shape[0]):
        value = False
        if has_extra != 0:
            value = extra[world]
        for body in range(count):
            value = value or selected[world * count + body]
        worlds[world] = value


@tx.kernel(gradient="unsupported")
def _convert(
    values: tx.Array[tx.f32],
    selected: tx.Array[tx.bool1],
    addresses: tx.Array[tx.i32],
    com: tx.Array[tx.f32],
    q: tx.Array[tx.f32],
    v: tx.Array[tx.f32],
    scratch: tx.Array[tx.f32],
    error: tx.Array[tx.i32],
    nq: tx.i32,
    nv: tx.i32,
    fields: tx.i32,
    per_body: tx.Array[tx.i32],
):
    for lane in tx.range(selected.shape[0]):
        if selected[lane]:
            body = lane % (addresses.shape[0] // 2)
            world = lane // (addresses.shape[0] // 2)
            qa = world * nq + addresses[body * 2]
            va = world * nv + addresses[body * 2 + 1]
            base = lane * 13
            enabled_fields = fields
            if fields < 0:
                enabled_fields = per_body[body]
            valid = enabled_fields > 0 and enabled_fields <= 15
            for c in range(13):
                enabled = (
                    (c < 3 and (enabled_fields & 1) != 0)
                    or (c >= 3 and c < 7 and (enabled_fields & 2) != 0)
                    or (c >= 7 and c < 10 and (enabled_fields & 4) != 0)
                    or (c >= 10 and (enabled_fields & 8) != 0)
                )
                if enabled and not tx.isfinite(values[base + c]):
                    valid = False
            x, y, z, w = q[qa + 4], q[qa + 5], q[qa + 6], q[qa + 3]
            nx, ny, nz, nw = x, y, z, w
            if (enabled_fields & 2) != 0:
                nx, ny, nz, nw = (
                    values[base + 3],
                    values[base + 4],
                    values[base + 5],
                    values[base + 6],
                )
                norm = (
                    tx.f64(nx) * tx.f64(nx)
                    + tx.f64(ny) * tx.f64(ny)
                    + tx.f64(nz) * tx.f64(nz)
                    + tx.f64(nw) * tx.f64(nw)
                )
                if not tx.isfinite(norm) or tx.abs(norm - tx.f64(1.0)) > tx.f64(
                    0.00000762939453125
                ):
                    valid = False
                else:
                    scale = tx.cast_f32(1.0 / tx.sqrt(norm))
                    nx, ny, nz, nw = nx * scale, ny * scale, nz * scale, nw * scale
            if valid:
                # qvel stores origin/world linear and body-local angular velocity.
                wx = _rotate(x, y, z, w, v[va + 3], v[va + 4], v[va + 5], 0)
                wy = _rotate(x, y, z, w, v[va + 3], v[va + 4], v[va + 5], 1)
                wz = _rotate(x, y, z, w, v[va + 3], v[va + 4], v[va + 5], 2)
                cx, cy, cz = com[body * 3], com[body * 3 + 1], com[body * 3 + 2]
                rx = _rotate(x, y, z, w, cx, cy, cz, 0)
                ry = _rotate(x, y, z, w, cx, cy, cz, 1)
                rz = _rotate(x, y, z, w, cx, cy, cz, 2)
                vx = v[va] + wy * rz - wz * ry
                vy = v[va + 1] + wz * rx - wx * rz
                vz = v[va + 2] + wx * ry - wy * rx
                if (enabled_fields & 4) != 0:
                    vx, vy, vz = values[base + 7], values[base + 8], values[base + 9]
                if (enabled_fields & 8) != 0:
                    wx, wy, wz = values[base + 10], values[base + 11], values[base + 12]
                rx = _rotate(nx, ny, nz, nw, cx, cy, cz, 0)
                ry = _rotate(nx, ny, nz, nw, cx, cy, cz, 1)
                rz = _rotate(nx, ny, nz, nw, cx, cy, cz, 2)
                for c in range(3):
                    scratch[base + c] = q[qa + c]
                    if (enabled_fields & 1) != 0:
                        scratch[base + c] = values[base + c]
                scratch[base + 3], scratch[base + 4], scratch[base + 5], scratch[base + 6] = (
                    nw,
                    nx,
                    ny,
                    nz,
                )
                scratch[base + 7] = vx - wy * rz + wz * ry
                scratch[base + 8] = vy - wz * rx + wx * rz
                scratch[base + 9] = vz - wx * ry + wy * rx
                scratch[base + 10] = _rotate(-nx, -ny, -nz, nw, wx, wy, wz, 0)
                scratch[base + 11] = _rotate(-nx, -ny, -nz, nw, wx, wy, wz, 1)
                scratch[base + 12] = _rotate(-nx, -ny, -nz, nw, wx, wy, wz, 2)
                if (enabled_fields & 14) == 0:
                    for c in range(6):
                        scratch[base + 7 + c] = v[va + c]
                elif (enabled_fields & 10) == 0:
                    for c in range(3):
                        scratch[base + 10 + c] = v[va + 3 + c]
                for c in range(13):
                    if not tx.isfinite(scratch[base + c]):
                        valid = False
            if not valid:
                tx.atomic_or(error[0], 524288)


@tx.kernel(gradient="unsupported")
def _scatter(
    scratch: tx.Array[tx.f32],
    selected: tx.Array[tx.bool1],
    addresses: tx.Array[tx.i32],
    q: tx.Array[tx.f32],
    v: tx.Array[tx.f32],
    error: tx.Array[tx.i32],
    nq: tx.i32,
    nv: tx.i32,
):
    for lane in tx.range(selected.shape[0]):
        if error[0] == 0 and selected[lane]:
            body = lane % (addresses.shape[0] // 2)
            world = lane // (addresses.shape[0] // 2)
            qa = world * nq + addresses[body * 2]
            va = world * nv + addresses[body * 2 + 1]
            for c in range(7):
                q[qa + c] = scratch[lane * 13 + c]
            for c in range(6):
                v[va + c] = scratch[lane * 13 + 7 + c]


class PreparedFreeBodies:
    """Derived scratch and immutable native bindings, never authoritative state."""

    def __init__(self, model, banks, flags, nworld, device, storage):
        import mujoco

        rows = []
        for body in range(1, model.nbody):
            j = int(model.body_jntadr[body])
            if (
                model.body_parentid[body] == 0
                and model.body_jntnum[body] == 1
                and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            ):
                rows.append(
                    (
                        model.body(body).name,
                        body,
                        int(model.jnt_qposadr[j]),
                        int(model.jnt_dofadr[j]),
                        tuple(map(float, model.body_ipos[body])),
                    )
                )
        self.bindings = tuple(rows)
        self.nworld, self.count = nworld, len(rows)
        self.device, self.flags = device, flags
        self.nq, self.nv = model.nq, model.nv
        self.banks = tuple(
            (bank.qpos.view(tx.f32).flatten(), bank.qvel.view(tx.f32).flatten()) for bank in banks
        )
        self.resources = []
        self.ready = False
        self.protected = []
        for bank in (0, 1):
            for name in storage.fields:
                array = storage.borrow(name, bank)
                if array.size:
                    extent = sum((n - 1) * s for n, s in zip(array.shape, array.strides))
                    self.protected.append((array.ptr, array.ptr + extent + 4))

    def prepare(self, stream):
        def allocate(shape, dtype):
            array = tx.empty(shape, dtype, device=self.device)
            self.resources.append(array)
            return array

        self.addresses = allocate(self.count * 2, tx.i32)
        self.com = allocate(self.count * 3, tx.f32)
        self.scratch = allocate(self.nworld * self.count * 13, tx.f32)
        self.worlds = allocate(self.nworld, tx.bool1)
        self.empty_mask = allocate(self.nworld * self.count, tx.bool1)
        self.addresses.assign(
            np.asarray([row[2:4] for row in self.bindings], np.int32).reshape(-1), stream=stream
        )
        self.com.assign(
            np.asarray([row[4] for row in self.bindings], np.float32).reshape(-1), stream=stream
        )
        self.empty_mask.zero_(stream=stream)
        for bank in (0, 1):
            self.enqueue(bank, self.scratch, self.empty_mask, 15, stream)
        stream.synchronize()
        self.ready = True

    def check(self, values, selected, per_body=None, extra=None):
        inputs = [
            (values, (self.nworld, self.count, 13), tx.f32),
            (selected, (self.nworld, self.count), tx.bool1),
        ]
        if per_body is not None:
            inputs.append((per_body, (self.count,), tx.i32))
        if extra is not None:
            inputs.append((extra, (self.nworld,), tx.bool1))
        spans = []
        for array, shape, dtype in inputs:
            itemsize = 1 if dtype == tx.bool1 else 4
            strides = tuple(
                np.prod(shape[i + 1 :], dtype=int) * itemsize for i in range(len(shape))
            )
            if (
                not isinstance(array, tx.Array)
                or tuple(array.shape) != shape
                or array.dtype != dtype
                or tuple(array.strides) != strides
                or str(array.device) != str(self.device)
            ):
                raise ValueError(
                    "free-body input requires canonical-device contiguous float32[N,F,13], bool[N,F], int32[F] components or bool[N] worlds"
                )
            span = (array.ptr, array.ptr + array.size * itemsize)
            if array.size and any(max(span[0], a) < min(span[1], b) for a, b in self.protected):
                raise ValueError("free-body input aliases runtime storage")
            if array.size and any(max(span[0], a) < min(span[1], b) for a, b in spans):
                raise ValueError("free-body input arrays alias")
            if array.size:
                spans.append(span)

    def enqueue(self, bank, values, selected, fields, stream, per_body=None, extra=None):
        _world_mask(
            selected,
            self.worlds,
            self.count,
            self.worlds if extra is None else extra,
            int(extra is not None),
            mode="cuda",
            stream=stream,
        )
        if self.count:
            _convert(
                values,
                selected,
                self.addresses,
                self.com,
                *self.banks[bank],
                self.scratch,
                self.flags,
                self.nq,
                self.nv,
                fields,
                self.addresses if per_body is None else per_body,
                mode="cuda",
                stream=stream,
            )
            _scatter(
                self.scratch,
                selected,
                self.addresses,
                *self.banks[bank],
                self.flags,
                self.nq,
                self.nv,
                mode="cuda",
                stream=stream,
            )

    @property
    def allocated_bytes(self):
        return sum(
            array.size * np.dtype(array.dtype.numpy_dtype).itemsize for array in self.resources
        )
