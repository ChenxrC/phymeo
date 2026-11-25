"""Shared limits for the registered TYX device array payload."""

from contextlib import contextmanager
from threading import RLock
from types import MappingProxyType

DEVICE_PAYLOAD_EXCLUSIONS = (
    "context",
    "jit-cache",
    "allocator-overhead",
    "host",
    "json",
    "external-device-controls",
    "external-task-workspaces",
)


def _bytes(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative builtin integer")
    return value


class MuJoCoTyxResourceBudget:
    """One shared, reentrant device payload budget for related runtime owners.

    Native contexts, JIT/cache, allocator overhead, HOST arrays and JSON are
    outside this measured scope. UNKNOWN payload remains part of current_bytes.
    Admission checks do not themselves reserve or double-charge any payload.
    """

    def __init__(self, device_payload_limit_bytes):
        self._limit = _bytes(device_payload_limit_bytes, "device_payload_limit_bytes")
        self._lock = RLock()
        self._current = self._reserved = self._peak = 0
        self._next_owner = 0
        self._owners = {}

    @property
    def device_payload_limit_bytes(self):
        return self._limit

    def _admit(self, additional):
        if self._current + self._reserved + additional > self._limit:
            raise MemoryError("TYX registered device payload budget exceeded")

    @contextmanager
    def admit(self, additional_bytes):
        """Hold the shared RLock while checking an uncharged peak allowance.

        Related leaf reserve/release calls reenter the same lock. Callers must
        acquire this budget before child resource locks; no reservation is
        created here and no actual allocation is inferred from admission.
        """
        additional = _bytes(additional_bytes, "additional_bytes")
        with self._lock:
            self._admit(additional)
            yield

    def reserve(self, payload_bytes, *, backend="cuda"):
        """Reserve a complete leaf manifest before its Runtime/staging creation."""
        size = _bytes(payload_bytes, "payload_bytes")
        if backend not in ("cuda", "musa"):
            raise ValueError("unknown TYX allocation backend")
        with self._lock:
            self._admit(size)
            identity = self._next_owner
            self._next_owner += 1
            self._owners[identity] = dict(
                backend=backend,
                manifest_payload_bytes=size,
                reserved_bytes=size,
                finished=False,
                runtime_release_unknown=False,
                allocations={},
            )
            self._reserved += size
            return _Reservation(self, identity)

    def snapshot(self):
        with self._lock:
            owners = []
            backends = {row["backend"] for row in self._owners.values()}
            scope = "registered-cuda-array-payload-only"
            if "musa" in backends:
                scope = (
                    "registered-musa-array-payload-only"
                    if len(backends) == 1
                    else "registered-device-array-payload-only"
                )
            unknown = 0
            for identity, row in self._owners.items():
                allocations = tuple(
                    MappingProxyType(dict(name=name, **allocation))
                    for name, allocation in row["allocations"].items()
                )
                current = sum(item["nbytes"] for item in allocations if item["state"] != "RELEASED")
                retained = sum(
                    item["nbytes"] for item in allocations if item["state"] == "UNKNOWN_RELEASE"
                )
                unknown += retained
                owners.append(
                    MappingProxyType(
                        dict(
                            owner_id=identity,
                            manifest_payload_bytes=row["manifest_payload_bytes"],
                            current_bytes=current,
                            reserved_bytes=row["reserved_bytes"],
                            unknown_retained_bytes=retained,
                            allocation_finished=row["finished"],
                            runtime_release_unknown=row["runtime_release_unknown"],
                            allocations=allocations,
                        )
                    )
                )
            return MappingProxyType(
                dict(
                    schema="mujoco-tyx-device-payload-budget-v1",
                    scope=scope,
                    device_payload_limit_bytes=self._limit,
                    current_bytes=self._current,
                    peak_bytes=self._peak,
                    reserved_bytes=self._reserved,
                    unknown_retained_bytes=unknown,
                    unknown_is_current_subset=True,
                    owners=tuple(owners),
                    total_device_live_bytes=None,
                    total_device_peak_bytes=None,
                    host_live_bytes=None,
                    host_peak_bytes=None,
                    excluded=DEVICE_PAYLOAD_EXCLUSIONS,
                )
            )


class _Reservation:
    def __init__(self, budget, identity):
        self._budget, self.owner_id = budget, identity

    def allocated(self, name, nbytes):
        size = _bytes(nbytes, "allocation nbytes")
        with self._budget._lock:
            row = self._budget._owners[self.owner_id]
            if row["finished"] or name in row["allocations"]:
                raise RuntimeError("payload reservation allocation identity reused or closed")
            if type(name) is not str or not name or size > row["reserved_bytes"]:
                raise RuntimeError("allocation does not fit its manifest reservation")
            row["allocations"][name] = dict(nbytes=size, state="ALLOCATED")
            row["reserved_bytes"] -= size
            self._budget._reserved -= size
            self._budget._current += size
            self._budget._peak = max(self._budget._peak, self._budget._current)

    def released(self, name):
        with self._budget._lock:
            item = self._budget._owners[self.owner_id]["allocations"][name]
            if item["state"] != "ALLOCATED":
                raise RuntimeError("payload release is not an allocated buffer")
            item["state"] = "RELEASED"
            self._budget._current -= item["nbytes"]

    def unknown(self, name=None):
        with self._budget._lock:
            row = self._budget._owners[self.owner_id]
            if name is None:
                row["runtime_release_unknown"] = True
            else:
                item = row["allocations"][name]
                if item["state"] != "ALLOCATED":
                    raise RuntimeError("unknown payload release is not an allocated buffer")
                item["state"] = "UNKNOWN_RELEASE"

    def finish(self):
        """Cancel only unconsumed reservation; actual/unknown arrays stay charged."""
        with self._budget._lock:
            row = self._budget._owners[self.owner_id]
            self._budget._reserved -= row["reserved_bytes"]
            row["reserved_bytes"] = 0
            row["finished"] = True
