"""Owned buffers for the bounded D1 node producer (no contact response)."""
from contextlib import contextmanager

import numpy as np

from .mujoco_tyx._resources import DeviceResources


def create_runtime(mode, cache_root):
    # Deliberately lazy: registration/resource rejection never initializes TYX.
    import tyx as tx
    options = {"backend": "cuda" if mode == "cuda" else "cpu"}
    if cache_root is not None:
        options["cache_root"] = cache_root
    return tx.Runtime(**options)


class DeformableBuffers:
    def __init__(self, worlds, mode, trace, cache_root=None):
        self.mode = mode
        self.resources = DeviceResources()
        self.worlds = {}
        self.allocations = 0
        self.native_bytes = 0
        self.uploaded_bytes = 0
        self.downloaded_bytes = 0
        self.kernels = 0
        self.syncs = 0
        self.runtime = None
        try:
            self.runtime = create_runtime(mode, cache_root)
            self.resources.set_runtime(self.runtime)
            import tyx as tx
            self.tx = tx
            with self.context():
                for key, model in worlds.items():
                    n, p = model.n, model.p
                    arrays = {}
                    for bank in range(2):
                        arrays[f"x{bank}"] = self.allocate(model.initial_position)
                        arrays[f"v{bank}"] = self.allocate(model.initial_velocity)
                        arrays[f"clock{bank}"] = self.allocate(np.zeros(2, np.float32))
                        arrays[f"tick{bank}"] = self.allocate(np.zeros(1, np.int32))
                    for name in ("predicted_x", "predicted_v", "force"):
                        arrays[name] = self.allocate(np.zeros((n, 3), np.float32))
                    for name, value in (("weights", model.weights), ("gravity", model.gravity),
                                        ("edges", model.edges), ("rest", model.rest),
                                        ("compliance", model.compliance)):
                        arrays[name] = self.allocate(value)
                    for name in ("alpha_tmp", "alpha", "zero", "multipliers"):
                        arrays[name] = self.allocate(np.zeros(p, np.float32))
                    arrays["status"] = self.allocate(np.zeros(1, np.int32))
                    if trace:
                        arrays["trace"] = self.allocate(np.zeros((p, 18), np.float32))
                    self.worlds[key] = arrays
            self.sync()
        except BaseException as primary:
            try:
                self.resources.close()
            except BaseException as cleanup:
                error = BaseExceptionGroup("deformable construction and cleanup failed",
                                           [primary, cleanup])
                error.resource_owner = self
                raise error
            raise

    @contextmanager
    def context(self):
        with self.tx.use_runtime(self.runtime):
            yield

    def allocate(self, value):
        value = np.ascontiguousarray(value)
        if self.mode == "cuda":
            array = self.tx.cuda_array(value)
            self.resources.add_buffer(array)
        else:
            array = value.copy()
        self.allocations += 1
        self.native_bytes += value.nbytes
        self.uploaded_bytes += value.nbytes
        return array

    def upload(self, target, value):
        value = np.ascontiguousarray(value)
        if self.mode == "cuda":
            target.copy_from(value)
        else:
            np.copyto(target, value, casting="no")
        self.uploaded_bytes += value.nbytes

    def read(self, value):
        result = value.to_numpy() if self.mode == "cuda" else value.copy()
        self.downloaded_bytes += result.nbytes
        return result

    def call(self, kernel, *args):
        kernel(*args, mode=self.mode)
        self.kernels += 1

    def sync(self):
        self.runtime.sync()
        self.syncs += 1

    def close(self):
        self.resources.close()
