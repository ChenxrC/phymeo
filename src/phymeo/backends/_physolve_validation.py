"""Prepared device validation with one fixed-size host completion result."""

import numpy as np
import tyx as tx


@tx.kernel(gradient="unsupported")
def _finite(values: tx.Array[tx.f32], flags: tx.Array[tx.i32], bit: tx.i32):
    for i in tx.range(values.shape[0]):
        if not tx.isfinite(values[i]):
            tx.atomic_or(flags[0], bit)


@tx.kernel(gradient="unsupported")
def _capacity(values: tx.Array[tx.i32], flags: tx.Array[tx.i32], limit: tx.i32):
    for i in tx.range(values.shape[0]):
        if values[i] < 0 or values[i] > limit:
            tx.atomic_or(flags[0], 1)


class PreparedValidation:
    """Borrow fixed bank views; no per-world dispatch or full-state host readback."""

    finite_fields = ("qpos", "qvel", "qacc", "time", "ctrl", "qfrc_applied")

    def __init__(self, banks, device, extra_banks=None):
        self.flags = tx.zeros(1, tx.i32, device=device)
        self.host = np.zeros(1, dtype=np.int32)
        self.plans = []
        self.readbacks = 0
        self.extra_plans = [(), ()]
        for bank in banks:
            finite = tuple(
                (name, getattr(bank, name).view(tx.f32).flatten(), 1 << (i + 1))
                for i, name in enumerate(self.finite_fields)
                if getattr(bank, name).size
            )
            capacity = tuple(
                (getattr(bank, name).view(tx.i32).flatten(), limit)
                for name, limit in (("nacon", bank.naconmax), ("ncollision", bank.naconmax),
                                    ("nefc", bank.njmax))
            )
            self.plans.append((finite, capacity))
        if extra_banks is not None:
            names = tuple(extra_banks[0])
            self.extra_plans = [tuple((name, bank[name].flatten(), 1 << (i + 1 + len(self.finite_fields)))
                                     for i, name in enumerate(names)) for bank in extra_banks]
            self.finite_fields = (*self.finite_fields, *names)

    def prepare(self, stream):
        # The runtime owns this object before launches can fail in flight.
        self.clear(stream)
        self.enqueue(0, stream)
        self.read(stream)
        self.readbacks = 0

    def clear(self, stream):
        self.flags.zero_(stream=stream)

    def enqueue(self, index, stream, *, include_extra=True):
        finite, capacity = self.plans[index]
        for _, array, bit in finite:
            _finite(array, self.flags, bit, mode="cuda", stream=stream)
        for array, limit in capacity:
            _capacity(array, self.flags, limit, mode="cuda", stream=stream)
        if include_extra:
            self.enqueue_extra(index, stream)

    def enqueue_extra(self, index, stream):
        for _, array, bit in self.extra_plans[index]:
            _finite(array, self.flags, bit, mode="cuda", stream=stream)

    def read(self, stream):
        # TYX fills this persistent caller buffer and waits only for this stream.
        self.flags.copy_to_host(self.host, stream=stream)
        self.readbacks += 1
        bits = int(self.host[0])
        if bits & 1:
            raise RuntimeError("PhySolve contact/constraint capacity exceeded; not published")
        names = [name for i, name in enumerate(self.finite_fields) if bits & (1 << (i + 1))]
        if names:
            raise RuntimeError(f"non-finite PhySolve {', '.join(names)}; not published")
