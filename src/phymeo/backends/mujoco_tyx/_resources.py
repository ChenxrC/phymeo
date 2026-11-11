"""Retain ambiguous native releases without calling an opaque handle twice.

Buffers precede their runtime. Only a normally returned callback establishes
release completion. An exception inside a callback is sticky UNKNOWN_RELEASE,
not evidence that its side effect did not happen. Quarantine prevents ordinary
in-process garbage collection from invoking a retained owner's destructor;
Python interpreter shutdown/destructor ordering is outside this guarantee.
"""

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ReleaseFailure:
    """Original opaque release failure and the owner whose effect is unknown."""

    kind: str
    owner: Any
    error: BaseException


@dataclass(frozen=True, slots=True)
class _Release:
    owner: Any
    callback: Callable[[], None]


_QUARANTINE: dict[int, "DeviceResources"] = {}
_QUARANTINE_LOCK = RLock()


def quarantined_resources() -> tuple["DeviceResources", ...]:
    """Inspect strong owners retained after unknown release; never retry them.

    No clearing/recovery operation is offered: callers cannot establish native
    release completion from the stored Python exception. Returned owners remain
    sticky even if a test knows more about its injected callback's phase.
    """
    with _QUARANTINE_LOCK:
        return tuple(_QUARANTINE.values())


class DeviceResources:
    """Two-phase, one-attempt cleanup for one device bank's owned resources."""

    def __init__(self):
        self._lock = RLock()
        self._buffers: list[_Release] = []
        self._runtime: _Release | None = None
        self._failures: list[ReleaseFailure] = []
        self._closed = False
        self._closing = False

    @property
    def state(self) -> str:
        with self._lock:
            if self._failures:
                return "UNKNOWN_RELEASE"
            if self._closed:
                return "CLOSED"
            return "CLOSING" if self._closing else "OPEN"

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    @property
    def failures(self) -> tuple[ReleaseFailure, ...]:
        with self._lock:
            return tuple(self._failures)

    @property
    def pending_buffer_count(self) -> int:
        with self._lock:
            return len(self._buffers)

    @property
    def runtime_pending(self) -> bool:
        with self._lock:
            return self._runtime is not None

    def _require_registration(self):
        if self._closed:
            raise RuntimeError("device resource ledger is closed")
        if self._closing or self._failures:
            raise RuntimeError("device resource ledger cannot accept owners during/after cleanup")

    def set_runtime(self, owner):
        """Own the runtime before registering dependent device buffers."""
        with self._lock:
            self._require_registration()
            if self._runtime is not None:
                raise ValueError("device resource runtime is already registered")
            callback = getattr(owner, "close", None)
            if not callable(callback):
                raise TypeError("device runtime owner must expose close()")
            self._runtime = _Release(owner, callback)

    def add_buffer(self, owner):
        """Own one distinct CUDA buffer; take its callback before cleanup."""
        with self._lock:
            self._require_registration()
            if self._runtime is None:
                raise RuntimeError("register the runtime before device buffers")
            if owner is self._runtime.owner:
                raise ValueError("runtime cannot also be a buffer owner")
            if any(item.owner is owner for item in self._buffers):
                raise ValueError("duplicate device buffer owner")
            callback = getattr(owner, "free", None)
            if not callable(callback):
                raise TypeError("device buffer owner must expose free()")
            self._buffers.append(_Release(owner, callback))

    def _unknown(self, kind, release, error):
        self._failures.append(ReleaseFailure(kind, release.owner, error))
        # Strong ownership must outlive a failed runtime/clone being discarded.
        # Lock ordering is ledger -> quarantine; inspection takes only quarantine.
        with _QUARANTINE_LOCK:
            _QUARANTINE[id(self)] = self

    def _raise_unknown(self):
        raise BaseExceptionGroup(
            "device cleanup UNKNOWN_RELEASE; native release is not retried",
            [failure.error for failure in self._failures],
        )

    def close(self):
        """Release pending owners once; keep unknown owners and dependency alive.

        An outer exception before this method is entered leaves PENDING owners
        available to a later call. Once an actual callback throws, subsequent
        calls only report the original unknown outcome and never invoke it again.
        """
        with self._lock:
            if self._closing:
                raise RuntimeError("recursive device resource cleanup is not supported")
            if self._failures:
                self._raise_unknown()
            if self._closed:
                return
            self._closing = True
            try:
                while self._buffers:
                    release = self._buffers.pop()
                    try:
                        release.callback()
                    except BaseException as error:
                        self._unknown("buffer", release, error)
                if self._failures:
                    # The runtime remains owned and uncalled while any dependent
                    # buffer has an ambiguous release outcome.
                    self._raise_unknown()
                if self._runtime is not None:
                    release, self._runtime = self._runtime, None
                    try:
                        release.callback()
                    except BaseException as error:
                        self._unknown("runtime", release, error)
                        self._raise_unknown()
                self._closed = True
            finally:
                self._closing = False