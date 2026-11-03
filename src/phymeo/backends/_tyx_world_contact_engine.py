"""Shared TYX buffers for the two contact samples in an accepted physical substep."""

from contextlib import ExitStack

import numpy as np

from ._world_step_layout import as_fp32


class WorldContactEngine:
    def __init__(self, bindings, mode, cache_root):
        import tyx as tx

        from ._tyx_normal_layer import normal_layer_rows
        from ._tyx_world_contact_kernels import map_and_advance, material_velocities

        self.mode, self.n = mode, len(bindings)
        self.tx, self.prepare, self.law, self.advance = (
            tx,
            material_velocities,
            normal_layer_rows,
            map_and_advance,
        )
        self.resources = ExitStack()
        self.runtime = tx.Runtime(
            backend="cuda" if mode == "cuda" else "cpu", cache_root=cache_root
        )
        self.resources.callback(self.runtime.close)
        self.counts = dict.fromkeys(
            (
                "completed_kernels",
                "synchronizations",
                "device_allocations",
                "static_uploaded_bytes",
                "uploaded_bytes",
                "downloaded_bytes",
            ),
            0,
        )
        try:
            with tx.use_runtime(self.runtime):
                n = self.n
                self.axis = self.buffer((n, 3), data=[b.axis for b in bindings])
                self.inverse_mass = self.buffer((n, 4), data=[1 / b.mass for b in bindings])
                self.geometry = self.buffer((2 * n, 13))
                self.rows = self.buffer((2 * n, 13))
                self.forces = self.buffer((2 * n, 5))
                self.base = self.buffer((n, 4, 3))
                self.other = self.buffer((n, 4, 3))
                self.mid = self.buffer((n, 4, 3))
                self.mapped = self.buffer((n, 4, 3))
                self.dx = self.buffer((n, 4, 3))
                self.dv = self.buffer((n, 4, 3))
                self.status = self.buffer((n,), np.int32)
                self.enabled = self.buffer((n,), np.int32)
        except BaseException:
            self.close()
            raise

    def buffer(self, shape, dtype=np.float32, data=None):
        host = None if data is None else as_fp32(data, "static world model", shape)
        if self.mode == "interpret":
            return np.empty(shape, dtype=dtype) if host is None else host
        array = self.tx.cuda_empty(shape, dtype) if host is None else self.tx.cuda_array(host)
        self.resources.callback(array.free)
        self.counts["device_allocations"] += 1
        if host is not None:
            self.counts["static_uploaded_bytes"] += host.nbytes
        return array

    def upload(self, device, host):
        if self.mode == "cuda":
            device.copy_from(host)
            self.counts["uploaded_bytes"] += host.nbytes
        else:
            np.copyto(device, host)

    def read(self, device):
        if self.mode == "cuda":
            array = device.to_numpy()
            self.counts["downloaded_bytes"] += array.nbytes
            return array
        return device.copy()

    def evaluate(self, velocity, stages, enabled, h):
        """One material/force/mapping chain for a sample; two chains for a physical step."""
        n = self.n
        with self.tx.use_runtime(self.runtime), np.errstate(all="ignore"):
            self.upload(self.base, velocity)
            self.upload(self.enabled, enabled)
            for index, (geometry, other) in enumerate(stages):
                stage = 0 if len(stages) == 1 else index + 1
                self.upload(self.geometry, geometry)
                self.upload(self.other, other)
                self.prepare(
                    2 * n,
                    self.geometry,
                    self.mid if stage == 2 else self.base,
                    self.axis,
                    self.rows,
                    mode=self.mode,
                )
                self.law(2 * n, self.rows, self.forces, mode=self.mode)
                self.advance(
                    n,
                    stage,
                    h,
                    self.axis,
                    self.inverse_mass,
                    self.geometry,
                    self.forces,
                    self.base,
                    self.other,
                    self.enabled,
                    self.mapped,
                    self.mid,
                    self.dx,
                    self.dv,
                    self.status,
                    mode=self.mode,
                )
            # Every stage is submitted on the same TYX runtime before synchronizing.
            self.runtime.sync()
            self.counts["synchronizations"] += 1
            self.counts["completed_kernels"] += 3 * len(stages)
            data = [
                self.read(self.rows).reshape(n, 2, 13),
                self.read(self.forces).reshape(n, 2, 5),
                self.read(self.mapped),
                self.read(self.status),
            ]
            if len(stages) == 2:
                data.extend((self.read(self.dx), self.read(self.dv)))
            return data

    def close(self):
        self.resources.close()
