"""Prepared device-only masked restore of a PhySolve integration snapshot."""

import tyx as tx

from .physolve_initial import INTEGRATION_FIELDS


@tx.kernel(gradient="unsupported")
def _restore_float(
    mask: tx.Array[tx.bool1], source: tx.Array[tx.f32], target: tx.Array[tx.f32], width: tx.i32
):
    for i in tx.range(target.shape[0]):
        if mask[i // width]:
            target[i] = source[i]


@tx.kernel(gradient="unsupported")
def _restore_bool(
    mask: tx.Array[tx.bool1], source: tx.Array[tx.bool1], target: tx.Array[tx.bool1], width: tx.i32
):
    for i in tx.range(target.shape[0]):
        if mask[i // width]:
            target[i] = source[i]


class PreparedReset:
    """One immutable integration backup and persistent bindings for both banks.

    O(N * integration width) device storage; no host snapshots, per-world Python
    dispatch or stream wrappers on reset. Empty fields have no launch/binding.
    """

    def __init__(self, banks, nworld, device):
        self.mask = tx.zeros(nworld, tx.bool1, device=device)
        self.all_mask = tx.full(nworld, True, tx.bool1, device=device)
        self.sources, self.plans = [], [[], []]
        for name in INTEGRATION_FIELDS:
            source = getattr(banks[0], name)
            if source.size == 0:
                continue
            dtype = tx.bool1 if name == "eq_active" else tx.f32
            kernel = _restore_bool if dtype == tx.bool1 else _restore_float
            source = tx.clone(source).view(dtype).flatten()
            if source.size % nworld:
                raise ValueError(f"{name} is not world-major integration storage")
            width = source.size // nworld
            self.sources.append(source)
            for index, bank in enumerate(banks):
                target = getattr(bank, name).view(dtype).flatten()
                self.plans[index].append((kernel, source, target, width))

    def prepare(self, stream):
        # Compile/launch each required signature on the unchanged initial bank.
        # Creation owns completion; first reset must not become a JIT boundary.
        for kernel, source, target, width in self.plans[0]:
            kernel(self.all_mask, source, target, width, mode="cuda", stream=stream)

    def restore(self, index, mask, stream):
        selector = self.all_mask
        if mask is not None:
            self.mask.assign(mask, stream=stream)
            selector = self.mask
        for kernel, source, target, width in self.plans[index]:
            kernel(selector, source, target, width, mode="cuda", stream=stream)
