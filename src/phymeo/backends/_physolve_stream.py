"""Cold stream rebinding; physical arrays and their storage generation stay fixed."""

from numbers import Integral


def bind_stream(runtime, handle, owner):
    if isinstance(handle, bool) or not isinstance(handle, Integral) or handle < 0:
        raise ValueError("stream handle must be a nonnegative integer")
    if handle and owner is None:
        raise ValueError("external stream requires a lifetime owner")
    with runtime._scope():
        runtime._require_idle()
        old = runtime._stream
        default = runtime._owned_stream
        pointer = int(handle) or int(default.handle)
        if old.is_capturing:
            raise RuntimeError("stream rebinding is forbidden during capture")
        if pointer == int(old.handle):
            old.validate_context()
            return False
        candidate = default if pointer == int(default.handle) else runtime._tx.CudaStream.from_external(
            pointer, owner=owner, device=runtime._device.ordinal
        )
        try:
            candidate.validate_context()
            if candidate.is_capturing:
                raise RuntimeError("cannot bind a capturing external stream")
            runtime._complete()
        except Exception:
            if candidate is not default:
                candidate.destroy()
            raise
        had_graph = runtime.step_graph_ready
        try:
            runtime._release_step_graphs()
        except Exception as error:
            if candidate is not default:
                candidate.destroy()
            runtime._closing = True
            error.runtime = runtime
            error.stream_bindings_changed = True
            raise
        # Keep both wrappers until native retired sessions and any submitted
        # preparation work have been released. A failed phase keeps this list.
        runtime._stream_retained = [old, candidate]
        try:
            with runtime._tx.use_stream(candidate):
                for native_handle in runtime._handles:
                    runtime._native._rebind_runtime(native_handle, True)
                runtime._synchronize_stream(candidate)
        except Exception as error:
            try:
                runtime._synchronize_stream(candidate)
                with runtime._tx.use_stream(old):
                    for native_handle in runtime._handles:
                        runtime._native._rebind_runtime(native_handle, True)
                    runtime._complete()
                for native_handle in runtime._handles:
                    runtime._native._retire_runtime(native_handle, True)
                if candidate is not default:
                    candidate.destroy()
                runtime._stream_retained.clear()
                if had_graph:
                    runtime.prepare_step_graph()
            except Exception as cleanup:
                runtime._closing = True
                error.runtime = runtime
                error.stream_bindings_changed = True
                raise error from cleanup
            raise
        runtime._stream = runtime._storage.stream = candidate
        runtime._stream_rebindings += 1
        try:
            for native_handle in runtime._handles:
                runtime._native._retire_runtime(native_handle, True)
            if old is not default:
                old.destroy()  # Only our non-owning wrapper, never caller's handle.
            runtime._stream_retained.clear()
        except Exception as error:
            runtime._closing = True
            error.runtime = runtime
            error.stream_bindings_changed = True
            raise
        return True
