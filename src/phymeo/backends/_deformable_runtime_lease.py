"""A node buffer owner may borrow one resident scene runtime, never its state."""


class ResidentDeformableRuntimeLease:
    """Internal lifetime edge: child buffers must close before their scene runtime.

    DeviceResources owns this lease as its runtime release object. An ambiguous
    child buffer release therefore retains both this lease and the real runtime.
    No fake runtime identity or cross-runtime device pointer is introduced.
    """

    def __init__(self, scene):
        from .resident_scene import ResidentSceneBatch

        if type(scene) is not ResidentSceneBatch:
            raise TypeError("shared node runtime requires a real resident scene")
        with scene._service.locked():
            scene._check_open()
            self._batch = scene._service._batch
            self.runtime = self._batch._runtime
            self.mode = self._batch._mode
            self._claimed = False
            self._closed = False
            self._token = self._batch._register_boundary_runtime_lease(self)

    def claim(self, mode):
        with self._batch._lock:
            self._batch._check_open()
            if self._closed or self._claimed or mode != self.mode:
                raise ValueError("runtime lease is closed, claimed, or uses another mode")
            if self._batch._runtime is not self.runtime:
                raise RuntimeError("runtime lease no longer names its original runtime")
            self._claimed = True
            return self.runtime

    def close(self):
        with self._batch._lock:
            if not self._closed:
                self._batch._release_boundary_runtime_lease(self._token)
                self._closed = True
